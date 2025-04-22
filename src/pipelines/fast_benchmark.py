import yaml
from src.pipelines.run_workflow import run_workflow
import pandas as pd
import os
import traceback
from pathlib import Path
import qlib
from multiprocessing import Pool, cpu_count
from functools import partial


def run_single_benchmark_task(task_settings, qlib_init_params):
    """
    Execute a single benchmark task.
    Designed to be called by multiprocessing Pool.
    """
    task_index = task_settings.get(
        "task_index", "N/A")  # add task index to settings for logging
    print(f"--- Worker Process: Starting Benchmark Task {task_index} ---")

    try:
        qlib.init(**qlib_init_params)
        print(f"Worker {task_index}: Qlib initialized successfully.")
    except Exception as init_e:
        print(
            f"Worker {task_index}: FATAL - Error initializing Qlib: {init_e}")
        traceback.print_exc()
        return {
            "task_index": task_index,
            "config_path": task_settings.get("config_path"),
            "experiment_name": task_settings.get("experiment_name", "Unknown"),
            "recorder_id": None, "source_recorder_id": task_settings.get("recorder_id"),
            "factor_subset_used": task_settings.get("factor_subset", "all"),
            "data_range_used": task_settings.get("data_range", "from_config"),
            "status": "error",
            "message": f"Qlib initialization failed in worker: {init_e}",
            "metrics": None
        }

    config_path = task_settings.get("config_path")
    recorder_id = task_settings.get("recorder_id")

    if not config_path:
        print(f"Worker {task_index}: Skipping task - Missing 'config_path'.")
        return {"task_index": task_index, "status": "skipped", "message": "Missing config_path"}
    if not recorder_id:
        print(f"Worker {task_index}: Skipping task - Missing 'recorder_id'.")
        return {
            "task_index": task_index,
            "config_path": config_path,
            "experiment_name": task_settings.get("experiment_name", "Unknown"),
            "recorder_id": None,
            "source_recorder_id": None,
            "mode": "predict",
            "factor_subset_used": task_settings.get("factor_subset", "all"),
            "data_range_used": task_settings.get("data_range", "from_config"),
            "status": "skipped",
            "message": "Missing source recorder_id",
            "metrics": None
        }

    exp_name = task_settings.get("experiment_name")
    if not exp_name:
        try:
            with open(config_path, 'r') as f:
                base_cfg = yaml.safe_load(f)
                exp_name = base_cfg.get("task", {}).get("model", {}).get("experiment_name",
                                                                         base_cfg.get("experiment_name", Path(config_path).stem))
        except Exception:
            exp_name = Path(config_path).stem
            print(
                f"Worker {task_index}: Warning - Could not read experiment_name from {config_path}, using filename: {exp_name}")

    print(
        f"Worker {task_index}: Config='{config_path}', Source Exp='{exp_name}', Source Rec='{recorder_id}'")
    print(f"Worker {task_index}: Overrides - Factors='{task_settings.get('factor_subset', 'all')}', Range='{task_settings.get('data_range', 'from_config')}'")

    #
    try:
        result = run_workflow(
            config_path=config_path,
            experiment_name=exp_name,
            recorder_id=recorder_id,
            factor_subset=task_settings.get("factor_subset", "all"),
            data_range=task_settings.get("data_range")
        )
        result["task_index"] = task_index
        print(
            f"Worker {task_index}: Task finished. Status: {result.get('status', 'unknown')}")
        return result

    except Exception as e:
        print(
            f"Worker {task_index}: Critical error running workflow for {config_path}: {e}")
        traceback.print_exc()
        return {
            "task_index": task_index,
            "config_path": config_path,
            "experiment_name": exp_name,
            "recorder_id": None,
            "source_recorder_id": recorder_id,
            "factor_subset_used": task_settings.get("factor_subset", "all"),
            "data_range_used": task_settings.get("data_range", "from_config"),
            "status": "critical_error",
            "message": str(e),
            "metrics": None
        }


def run_fast_benchmark(config_list_path: str, output_path: str, num_workers: int = None):
    """
    Run multiple Qlib prediction workflows defined in a YAML list in parallel
    and summarize results.

    Args:
        config_list_path (str): Path to the YAML file containing the list of benchmark tasks.
        output_path (str): Path to save the JSON report.
        num_workers (int, optional): Number of parallel worker processes.
                                     Defaults to the number of CPU cores.
    """

    # --- Load Benchmark Config ---
    try:
        with open(config_list_path, "r") as f:
            benchmark_tasks_raw = yaml.safe_load(f)["configs"]
        if not isinstance(benchmark_tasks_raw, list):
            raise ValueError(
                "'configs' section should contain a list of tasks.")
        benchmark_tasks = [{**task, "task_index": i+1}
                           for i, task in enumerate(benchmark_tasks_raw)]
    except Exception as e:
        print(f"Error loading benchmark config {config_list_path}: {e}")
        return

    # --- Initialize Qlib Once ---
    # Use a default or attempt to load from the *first* task's config for initialization params
    # TODO: Consider a dedicated qlib_init section in benchmark_config.yaml itself
    first_config_path = benchmark_tasks[0].get(
        'config_path') if benchmark_tasks else None
    qlib_init_params = {  # Defaults
        "provider_uri": "data/tw_data",
        "region": "tw"
    }
    if first_config_path and os.path.exists(first_config_path):
        try:
            with open(first_config_path, "r") as f:
                first_task_full_config = yaml.safe_load(f)
                if "qlib_init" in first_task_full_config:
                    qlib_init_params.update(
                        first_task_full_config["qlib_init"])
        except Exception as e:
            print(
                f"Warning: Could not load qlib_init from {first_config_path}. Using defaults. Error: {e}")

    try:
        provider_uri_expanded = os.path.expanduser(
            qlib_init_params["provider_uri"])
        qlib_init_params["provider_uri"] = provider_uri_expanded
        if not os.path.exists(os.path.join(provider_uri_expanded, "calendars")):
            print(
                f"Warning: Qlib data path '{provider_uri_expanded}' may be incorrect or does not exist.")
        # print(f"Initializing Qlib with parameters: {qlib_init_params}")
        # qlib.init(**qlib_init_params)
        print("Qlib init parameters prepared for workers.")
    except Exception as e:
        # print(f"Error initializing Qlib: {e}. Aborting benchmark.")
        print(
            f"Error preparing Qlib init params or checking path: {e}. Aborting.")
        traceback.print_exc()
        return

    # --- Run Benchmark Tasks in Parallel ---
    if num_workers is None:
        num_workers = cpu_count()
        print(
            f"Number of workers not specified, using CPU count: {num_workers}")
    else:
        print(f"Using specified number of workers: {num_workers}")

    starmap_args = [(task, qlib_init_params) for task in benchmark_tasks]
    print(
        f"\nStarting parallel benchmark execution with {num_workers} workers for {len(benchmark_tasks)} tasks...")

    results = []
    try:
        # Create a multiprocessing Pool with the specified number of workers
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(run_single_benchmark_task, starmap_args)
            print("\nParallel execution finished.")
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        traceback.print_exc()

    # --- Save Results ---
    if not results:
        print("Warning: No results were collected from the benchmark tasks.")
        return

    print(f"Collected {len(results)} results.")
    results.sort(key=lambda x: x.get("task_index", float('inf')))

    df = pd.DataFrame(results)
    df = df.drop(columns=['task_index'], errors='ignore')

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_json(output_path_obj, orient="records",
                   indent=4, force_ascii=False)
        print(f"\n[Done] Benchmark report saved to {output_path_obj}")
    except Exception as e:
        print(f"Error saving results to JSON {output_path_obj}: {e}")
