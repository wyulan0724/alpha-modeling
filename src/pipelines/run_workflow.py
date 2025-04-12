from qlib.workflow import R
from qlib.model.trainer import task_train
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config, flatten_dict
import yaml
from pathlib import Path
import pandas as pd
import traceback
import uuid


def _apply_benchmark_overrides(task_config: dict, factor_subset: list = None, data_range: dict = None):
    """
    Modifies the task_config dictionary in-place based on benchmark parameters.

    Args:
        task_config (dict): The loaded Qlib task configuration.
        factor_subset (list, optional): List of factors to use. If ["all"] or None, uses all factors. Defaults to None.
        data_range (dict, optional): Dictionary specifying data segments (e.g., {"train": [...], "test": [...]}). Defaults to None.

    Returns:
        dict: The modified task_config.
    """
    # --- Factor Subset Override ---
    # TODO: This simulation logic will be replaced by Agent input.
    if factor_subset and factor_subset != "all" and isinstance(factor_subset, list):
        print(f"Applying Factor Subset Override: {factor_subset}")
        try:
            handler_config = task_config["dataset"]["kwargs"]["handler"]
            # Ensure processors lists exist
            if "learn_processors" not in handler_config["kwargs"]:
                handler_config["kwargs"]["learn_processors"] = []
            if "infer_processors" not in handler_config["kwargs"]:
                handler_config["kwargs"]["infer_processors"] = []

            filter_col_processor = {
                "class": "FilterCol",
                "module_path": "qlib.data.processor",
                "kwargs": {
                    "col_list": factor_subset,
                    "fields_group": "feature"
                }
            }

            # Add FilterCol or update existing one
            # FIXME: A more robust approach would be to check if FilterCol exists and update its col_list
            handler_config["kwargs"]["learn_processors"].append(
                filter_col_processor)
            handler_config["kwargs"]["infer_processors"].append(
                filter_col_processor)

        except KeyError as e:
            print(
                f"Warning: Could not apply factor_subset override. Config structure error? Missing key: {e}")
        except Exception as e:
            print(f"Warning: Error applying factor_subset override: {e}")

    # --- Data Range Override ---
    # TODO: This simulation logic will be replaced by Agent input.
    if data_range and isinstance(data_range, dict):
        print(f"Applying Data Range Override: {data_range}")
        try:
            if "segments" not in task_config["dataset"]["kwargs"]:
                task_config["dataset"]["kwargs"]["segments"] = {}
            # Override existing segments or add new ones
            for key, value in data_range.items():
                task_config["dataset"]["kwargs"]["segments"][key] = value
            print(
                f"Updated segments: {task_config['dataset']['kwargs']['segments']}")
        except KeyError as e:
            print(
                f"Warning: Could not apply data_range override. Config structure error? Missing key: {e}")
        except Exception as e:
            print(f"Warning: Error applying data_range override: {e}")

    return task_config


def _run_prediction_and_analysis_with_records(
    task_config: dict,
    experiment_name: str,
    source_recorder_id: str,
    predict_segment: str = "predict",
    benchmark_experiment_name: str = "BenchmarkRuns"
) -> dict:
    """
    Runs prediction using a pre-trained model and performs analysis.
    """
    metrics = {}
    benchmark_recorder_id = None
    status = "success"
    message = ""

    try:
        # --- Load Source Recorder and Model ---
        # TODO: This Recorder loading will be replaced by Model Zoo/DB lookup.
        print(
            f"Loading source recorder: experiment='{experiment_name}', recorder_id='{source_recorder_id}'")
        source_recorder = R.get_recorder(
            experiment_name=experiment_name, recorder_id=source_recorder_id)
        print("Loading model object (params.pkl)...")
        # Assume model saved as 'params.pkl'
        model = source_recorder.load_object("params.pkl")

        # --- Prepare Dataset for Prediction (using potentially modified task_config) ---
        print(
            f"Initializing dataset for prediction segment: '{predict_segment}'")
        dataset_config = task_config["dataset"]
        dataset = init_instance_by_config(
            dataset_config, accept_types=DatasetH)

        # --- Generate Predictions ---
        print(f"Generating predictions for segment: '{predict_segment}'...")
        pred_df = model.predict(dataset, segment=predict_segment)
        if isinstance(pred_df, pd.Series):
            pred_df = pred_df.to_frame("score")  # Ensure DataFrame format
        print(f"Prediction generated, shape: {pred_df.shape}")

        # --- Start New Recorder for Benchmark Results ---
        # Generate a unique name to avoid collisions if run multiple times
        unique_suffix = uuid.uuid4().hex[:8]
        benchmark_recorder_name = f"predict_{source_recorder_id}_{unique_suffix}"
        print(
            f"Starting new recorder for benchmark run: experiment='{benchmark_experiment_name}', name='{benchmark_recorder_name}'")
        with R.start(experiment_name=benchmark_experiment_name, recorder_name=benchmark_recorder_name):
            current_run_recorder = R.get_recorder()

            if current_run_recorder is None:
                raise RuntimeError(
                    "Failed to get active recorder instance within R.start context.")

            benchmark_recorder_id = current_run_recorder.id
            print(
                f"Benchmark recorder created with ID: {benchmark_recorder_id}")

            # --- Save Prediction and Label to Benchmark Recorder ---
            print("Saving predictions (pred.pkl) to benchmark recorder...")
            current_run_recorder.save_objects(**{"pred.pkl": pred_df})

            # Prepare and save label if needed by analysis records
            try:
                label_df = dataset.prepare(
                    predict_segment, col_set="label", data_key=DataHandlerLP.DK_R)
                if label_df is not None and not label_df.empty:
                    print("Saving labels (label.pkl) to benchmark recorder...")
                    current_run_recorder.save_objects(
                        **{"label.pkl": label_df})
                else:
                    print("Label data is empty or None, skipping label.pkl save.")
            except Exception as e:
                print(
                    f"Warning: Could not prepare or save labels: {e}. Analysis requiring labels might fail.")

            # --- Run Analysis Records ---
            analysis_records_config = task_config.get("record", [])
            if isinstance(analysis_records_config, dict):
                analysis_records_config = [analysis_records_config]

            for record_cfg in analysis_records_config:
                record_class = record_cfg.get("class")
                # Skip SignalRecord, only run analysis records
                if record_class and record_class != "SignalRecord":
                    try:
                        print(
                            f"Initializing and generating record: {record_class}")
                        # Pass the current_run_recorder  to the analysis record instance
                        # init_instance_by_config will handle loading class and passing kwargs
                        analysis_record_instance = init_instance_by_config(
                            record_cfg,
                            recorder=current_run_recorder   # Pass the *new* recorder here
                        )
                        # generate() should load pred.pkl/label.pkl from current_run_recorder
                        analysis_record_instance.generate()
                        print(f"Successfully generated record: {record_class}")
                    except FileNotFoundError as e:
                        print(
                            f"Warning: Could not generate record {record_class}. Missing dependency (e.g., pred.pkl/label.pkl)? Error: {e}")
                    except Exception as e:
                        print(f"Error generating record {record_class}: {e}")
                        traceback.print_exc()
                        if status == "success":
                            status = "warning"
                        message += f" Error in {record_class}: {e};"

            # --- Retrieve Metrics from the Active Run Recorder ---
            print("Retrieving metrics logged during the benchmark run...")
            metrics = current_run_recorder.list_metrics()

    except FileNotFoundError as e:
        print(
            f"Error in prediction/analysis setup: Source model or recorder not found? {e}")
        traceback.print_exc()
        status = "error"
        message = f"Source model/recorder load error: {e}"
        # Ensure basic metric keys exist even on failure
        metrics.setdefault('IC', None)
        metrics.setdefault('Rank IC', None)

    except Exception as e:
        print(f"Error during prediction or analysis execution: {e}")
        traceback.print_exc()
        status = "error"
        message = f"Execution error: {e}"
        # Ensure basic metric keys exist even on failure
        metrics.setdefault('IC', None)
        metrics.setdefault('Rank IC', None)

    # Return collected metrics and the ID of the *new* benchmark recorder
    # Also return status and message
    return metrics, benchmark_recorder_id, status, message.strip()


# --- Main Workflow Function ---
def run_workflow(
    config_path: str,
    mode: str,
    experiment_name: str,  # For 'retrain' or *source* for 'predict'
    recorder_id: str = None,  # *Source* recorder ID for 'predict' mode
    factor_subset: list = None,
    data_range: dict = None
):
    """
    Run a Qlib workflow based on the specified mode ('retrain' or 'predict').
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        print(f"Error: Config file not found: {config_path}")
        return {"config_path": config_path, "status": "error", "message": "Config file not found"}

    # --- Load Base YAML Config ---
    try:
        with open(config_path_obj, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        if not isinstance(base_config, dict) or "task" not in base_config:
            raise ValueError(
                "Invalid config format or missing 'task' section.")
    except Exception as e:
        print(f"Error loading/parsing base config {config_path}: {e}")
        return {"config_path": config_path, "status": "error", "message": f"Config loading error: {e}"}

    # --- Prepare Task Config with Overrides ---
    task_config = base_config.get("task", {})
    task_config = _apply_benchmark_overrides(
        task_config, factor_subset, data_range)

    # --- Execute based on Mode ---
    metrics = {}
    final_recorder_id = None  # ID of the recorder containing the final results
    status = "success"
    message = ""

    try:
        if mode == "retrain":
            print(
                f"Starting Retrain Mode for {config_path}, Experiment: {experiment_name}")
            if "test" not in task_config.get("dataset", {}).get("kwargs", {}).get("segments", {}):
                print(
                    "Warning: 'test' segment not defined. Metrics might be incomplete.")

            recorder = task_train(task_config, experiment_name=experiment_name)
            final_recorder_id = recorder.id
            # Save potentially modified config
            recorder.save_objects(benchmark_run_config=task_config)
            metrics = recorder.list_metrics()
            print(
                f"[Success] Retrain completed. Recorder ID: {final_recorder_id}")

        elif mode == "predict":
            print(
                f"Starting Predict Mode for {config_path}, Source Experiment: {experiment_name}, Source Recorder: {recorder_id}")
            if not recorder_id:
                raise ValueError(
                    "Source Recorder ID is required for 'predict' mode.")

            # Determine prediction segment
            predict_segment_name = "predict"
            if data_range and predict_segment_name in data_range:
                pass
            elif "test" in task_config.get("dataset", {}).get("kwargs", {}).get("segments", {}):
                predict_segment_name = "test"
                print(
                    f"Using '{predict_segment_name}' segment from config for prediction.")
            else:
                raise ValueError("Cannot determine prediction segment.")

            # Define experiment name for the new benchmark recorder
            benchmark_experiment_name = f"Benchmark_{experiment_name}"

            # Call the helper function that uses Record Templates
            metrics, final_recorder_id, status, message = _run_prediction_and_analysis_with_records(
                task_config=task_config,  # Pass the potentially modified config
                experiment_name=experiment_name,  # Source experiment
                source_recorder_id=recorder_id,  # Source recorder
                predict_segment=predict_segment_name,
                benchmark_experiment_name=benchmark_experiment_name
            )
            # final_recorder_id now holds the ID of the *new* benchmark recorder
            print(
                f"[Finished] Prediction and Analysis completed. Status: {status}. Benchmark Recorder ID: {final_recorder_id}")

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'retrain' or 'predict'.")

    except Exception as e:
        print(
            f"Critical Error during workflow execution (mode: {mode}, config: {config_path}): {e}")
        traceback.print_exc()
        status = "error"
        message = f"Critical workflow error: {e}"
        # Ensure basic metric keys exist
        metrics.setdefault('IC', None)
        metrics.setdefault('Rank IC', None)

    # --- Consolidate Results ---
    # Extract common metrics (keys should now be consistent from recorder.list_metrics())
    metrics_dict = {
        "ic": metrics.get("IC", None),
        "icir": metrics.get("ICIR", None),
        "rank_ic": metrics.get("Rank IC", None),
        "rank_icir": metrics.get("Rank ICIR", None),
        "annualized_return": metrics.get("1day.excess_return_with_cost.annualized_return", None),
        "information_ratio": metrics.get("1day.excess_return_with_cost.information_ratio", None),
        "max_drawdown": metrics.get("1day.excess_return_with_cost.max_drawdown", None),
        "pred_score_mean": metrics.get("pred_score_mean", None),
        "pred_score_std": metrics.get("pred_score_std", None),
    }

    # Include any other metrics found in the final recorder
    other_metrics_keys = set(flatten_dict(
        metrics).keys()) - set(metrics_dict.keys())
    other_metrics_values = {k: metrics.get(
        k) for k in other_metrics_keys if pd.notna(metrics.get(k))}
    metrics_dict.update(other_metrics_values)

    result_dict = {
        "config_path": config_path,
        # Keep original experiment name for reference
        "experiment_name": experiment_name,
        # ID of recorder with results (new one for predict)
        "recorder_id": final_recorder_id,
        # Add source ID for predict mode
        "source_recorder_id": recorder_id if mode == 'predict' else None,
        "mode": mode,
        "factor_subset_used": factor_subset if factor_subset else "all",
        "data_range_used": data_range if data_range else "from_config",
        "status": status,
        "message": message,
        "metrics": metrics_dict
    }

    return result_dict
