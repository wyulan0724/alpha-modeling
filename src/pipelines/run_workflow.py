# src/pipelines/run_workflow.py
import qlib
from qlib.workflow import R
from qlib.model.trainer import task_train
import os
import yaml
from pathlib import Path


def run_workflow(config_path: str):
    """Run a single Qlib workflow from a YAML config path, and return the metrics."""
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load YAML config file
    try:
        with open(config_path_obj, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration file {config_path} format error, cannot parse as dictionary.")
        if "task" not in config:
            raise ValueError(
                f"Configuration file {config_path} is missing required 'task' section.")

    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error reading or parsing YAML config file {config_path}: {e}")
    except Exception as e:
        raise Exception(
            f"Unexpected error processing config file {config_path}: {e}")

    # Initialize Qlib (read parameters from config['qlib_init'] if it exists)
    # Set default values for Qlib initialization parameters
    qlib_init_params = {
        "provider_uri": "~/.qlib/qlib_data/tw_data",
        "region": "tw"
    }

    # If qlib_init section exists in config, use its parameters to override defaults
    if "qlib_init" in config and isinstance(config["qlib_init"], dict):
        qlib_init_params.update(config["qlib_init"])

    provider_uri_str = qlib_init_params["provider_uri"]
    provider_uri_expanded = os.path.expanduser(provider_uri_str)

    if not os.path.exists(os.path.join(provider_uri_expanded, "calendars")):
        print(
            f"Warning: Qlib data path '{provider_uri_expanded}' may be incorrect or does not exist.")

    # Update provider_uri with expanded path
    qlib_init_params["provider_uri"] = provider_uri_expanded

    # Initialize Qlib with parameters
    print(f"Initializing Qlib with parameters: {qlib_init_params}")
    qlib.init(**qlib_init_params)

    # Determine experiment name (use 'experiment_name' from config if available, otherwise infer from filename)
    experiment_name = config.get("experiment_name", config_path_obj.stem)

    recorder = None
    try:
        # Execute workflow using task_train
        print(
            f"Starting workflow execution, experiment name: {experiment_name}, config: {config_path}")
        recorder = task_train(config["task"], experiment_name=experiment_name)

        # Save config to recorder
        recorder.save_objects(config=config)

        print(
            f"[Success] Model training completed, record ID: {recorder.id}, experiment name: {experiment_name}")

        # Get metrics from recorder
        metrics = recorder.list_metrics()
        ic = metrics.get("IC", None)
        icir = metrics.get("ICIR", None)
        rank_ic = metrics.get("Rank IC", None)
        rank_icir = metrics.get("Rank ICIR", None)
        annualized_return = metrics.get(
            "1day.excess_return_with_cost.annualized_return", None)
        information_ratio = metrics.get(
            "1day.excess_return_with_cost.information_ratio", None)
        max_drawdown = metrics.get(
            "1day.excess_return_with_cost.max_drawdown", None)

    except Exception as e:
        print(
            f"Error executing workflow {config_path} (experiment {experiment_name}): {e}")
        ic, icir, rank_ic, rank_icir, annualized_return, information_ratio, max_drawdown = (
            None,) * 7

    return {
        "experiment_name": experiment_name,
        "recorder_id": recorder.id if recorder else None,
        "ic": ic,
        "icir": icir,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "annualized_return": annualized_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
    }
