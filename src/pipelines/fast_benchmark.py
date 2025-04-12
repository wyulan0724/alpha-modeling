import yaml
from src.pipelines.run_workflow import run_workflow
import pandas as pd
import os
import traceback
from pathlib import Path
import qlib


def run_fast_benchmark(config_list_path: str, output_path: str):
    """Run multiple Qlib workflows defined in a YAML list and summarize results."""

    # --- Load Benchmark Config ---
    try:
        with open(config_list_path, "r") as f:
            benchmark_tasks = yaml.safe_load(f)["configs"]
        if not isinstance(benchmark_tasks, list):
            raise ValueError(
                "'configs' section should contain a list of tasks.")
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
        print(f"Initializing Qlib with parameters: {qlib_init_params}")
        qlib.init(**qlib_init_params)
    except Exception as e:
        print(f"Error initializing Qlib: {e}. Aborting benchmark.")
        return

    # --- Run Benchmark Tasks ---
    results = []
    for i, task_settings in enumerate(benchmark_tasks):
        print(f"\n--- Running Benchmark Task {i+1}/{len(benchmark_tasks)} ---")
        config_path = task_settings.get("config_path")
        mode = task_settings.get("mode")
        if not config_path or not mode:
            print(
                f"Skipping task {i+1}: Missing 'config_path' or 'mode'. Settings: {task_settings}")
            results.append({
                "config_path": config_path,
                "experiment_name": task_settings.get("experiment_name", "Unknown"),
                "recorder_id": None,
                "source_recorder_id": task_settings.get("recorder_id"),
                "mode": mode,
                "factor_subset_used": task_settings.get("factor_subset", "all"),
                "data_range_used": task_settings.get("data_range", "from_config"),
                "status": "skipped",
                "message": "Missing config_path or mode",
                "metrics": None
            })
            continue

        # Determine experiment name: Priority -> Task Setting > Base Config > Filename
        exp_name = task_settings.get("experiment_name")
        if not exp_name:
            try:
                with open(config_path, 'r') as f:
                    base_cfg = yaml.safe_load(f)
                    exp_name = base_cfg.get(
                        "experiment_name", Path(config_path).stem)
            except Exception:
                exp_name = Path(config_path).stem  # Fallback to filename
                print(
                    f"Warning: Could not read experiment_name from {config_path}, using filename: {exp_name}")

        print(f"Config: {config_path}, Mode: {mode}, Experiment: {exp_name}")
        print(f"Settings: {task_settings}")  # Print all settings for clarity

        try:
            result = run_workflow(
                config_path=config_path,
                mode=mode,
                experiment_name=exp_name,
                recorder_id=task_settings.get("recorder_id"),
                factor_subset=task_settings.get(
                    "factor_subset", "all"),  # Default to "all"
                data_range=task_settings.get("data_range")
            )
            results.append(result)
            print(
                f"Task {i+1} Result Status: {result.get('status', 'unknown')}")
        except Exception as e:
            print(
                f"Critical error running workflow for {config_path} (Task {i+1}): {e}")
            traceback.print_exc()
            results.append({
                "config_path": config_path,
                "experiment_name": exp_name,
                "recorder_id": None,
                "source_recorder_id": task_settings.get("recorder_id"),
                "mode": mode,
                "factor_subset_used": task_settings.get("factor_subset", "all"),
                "data_range_used": task_settings.get("data_range", "from_config"),
                "status": "critical_error",
                "message": str(e),
                "metrics": None
            })

    # --- Save Results ---
    # Convert results to DataFrame, then to JSON
    df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JSON
    try:
        df.to_json(output_path, orient="records", indent=4, force_ascii=False)
        print(f"\n[Done] Benchmark report saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON {output_path}: {e}")
