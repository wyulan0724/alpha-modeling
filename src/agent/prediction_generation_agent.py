"""
Prediction Generation Worker Agent for Alpha Modeling

Responsible for generating predictions and calculating signal metrics.
"""

from langchain_core.tools import tool
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Qlib imports
from qlib.utils import init_instance_by_config
from qlib.contrib.eva.alpha import calc_ic
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R

logger = logging.getLogger("AlphaModeling.PredictionGenerationTool")


def _extract_series_from_output(data, col_name="score", default_col_idx=0):
    """
    Extracts a pandas Series from raw model output or dataset label output.
    Handles cases where output is a DataFrame or Series.
    Sorts the series by index.
    """
    if isinstance(data, pd.DataFrame):
        if col_name in data.columns:
            series = data[col_name].copy()
        elif not data.empty and len(data.columns) > default_col_idx:
            series = data.iloc[:, default_col_idx].copy()
        else:
            logger.warning(
                f"Output DataFrame is empty or column '{col_name}' (and default index {default_col_idx}) not found."
            )
            return pd.Series(dtype=np.float64)  # Return empty series
    elif isinstance(data, pd.Series):
        series = data.copy()
    else:
        logger.warning(f"Unsupported data type for extraction: {type(data)}")
        return pd.Series(dtype=np.float64)  # Return empty series
    return series.sort_index()


def _calculate_ic_metrics(pred_series: pd.Series, label_series: pd.Series):
    """
    Aligns prediction and label series, then calculates IC and Rank IC.

    Args:
        pred_series (pd.Series): Pandas Series of predictions, sorted by index.
        label_series (pd.Series): Pandas Series of labels, sorted by index.

    Returns:
        tuple: (ic_mean, ric_mean)
    """
    if pred_series.empty or label_series.empty:
        logger.warning("Predictions or labels series is empty before alignment.")
        return np.nan, np.nan

    common_index = pred_series.index.intersection(label_series.index)

    if common_index.empty:
        logger.warning("No common indices between predictions and labels.")
        return np.nan, np.nan

    aligned_preds = pred_series.loc[common_index]
    aligned_labels = label_series.loc[common_index]

    if aligned_preds.empty or aligned_labels.empty:
        logger.warning("Empty data after aligning predictions and labels.")
        return np.nan, np.nan

    # Calculate IC and Rank IC using qlib's calc_ic function
    try:
        ic_s, ric_s = calc_ic(aligned_preds, aligned_labels)

        ic_mean = (
            ic_s.mean() if ic_s is not None and not np.isnan(ic_s).all() else np.nan
        )
        ric_mean = (
            ric_s.mean() if ric_s is not None and not np.isnan(ric_s).all() else np.nan
        )
        ic_ir = (
            ic_mean / ic_s.std()
            if ic_s is not None and not np.isnan(ic_s).all()
            else np.nan
        )
        ric_ir = (
            ric_mean / ric_s.std()
            if ric_s is not None and not np.isnan(ric_s).all()
            else np.nan
        )

        return {"IC": ic_mean, "Rank IC": ric_mean, "ICIR": ic_ir, "Rank ICIR": ric_ir}
    except Exception as e:
        logger.error(f"Error calculating IC metrics: {e}")
        return {"IC": np.nan, "Rank IC": np.nan, "ICIR": np.nan, "Rank ICIR": np.nan}


@tool
async def generate_model_predictions_tool(
    experiment_name: str,
    recorder_id: str,
    dataset_config: Dict[str, Any],
    prediction_segment: str = "test",
) -> Dict[str, Any]:
    """
    Generates predictions using a trained model from Qlib recorder and saves pred.pkl and label.pkl.

    Args:
        experiment_name: The experiment name where the trained model is stored.
        recorder_id: The recorder ID of the trained model run.
        dataset_config: The dataset configuration dictionary for qlib DatasetH. This argument MUST be a Python dictionary object, NOT a string representation of a dictionary.
        prediction_segment: The segment to generate predictions for (default: "test").
    Returns:
        A dictionary containing prediction generation results and status.
    """
    try:
        logger.info(
            f"Loading trained model from experiment: {experiment_name}, recorder: {recorder_id}"
        )

        # Load the source recorder and model
        source_recorder = R.get_recorder(
            experiment_name=experiment_name, recorder_id=recorder_id
        )

        # Try different model object names that might be used
        model_object_names = ["trained_model.pkl", "params.pkl", "model.pkl"]
        model = None
        model_object_name = None

        for obj_name in model_object_names:
            try:
                model = source_recorder.load_object(obj_name)
                model_object_name = obj_name
                logger.info(f"Successfully loaded model from: {obj_name}")
                break
            except Exception as e:
                logger.debug(f"Failed to load model with {obj_name}: {e}")
                continue

        if model is None:
            raise ValueError(f"Could not load model from any of: {model_object_names}")

        # Initialize dataset from config
        logger.info("Initializing dataset from config")
        dataset = init_instance_by_config(dataset_config)

        # Determine segment to predict
        available_segments = dataset_config.get("kwargs", {}).get("segments", {})

        if prediction_segment not in available_segments:
            if "test" in available_segments:
                prediction_segment = "test"
            elif "valid" in available_segments:
                prediction_segment = "valid"
            else:
                prediction_segment = (
                    list(available_segments.keys())[-1]
                    if available_segments
                    else "test"
                )

        logger.info(f"Generating predictions for segment: {prediction_segment}")

        # Generate predictions
        predictions_raw = model.predict(dataset, segment=prediction_segment)

        # Ensure predictions are in DataFrame format
        if isinstance(predictions_raw, pd.Series):
            pred_df = predictions_raw.to_frame("score")
        elif isinstance(predictions_raw, pd.DataFrame):
            pred_df = predictions_raw
        else:
            pred_series = _extract_series_from_output(predictions_raw, col_name="score")
            pred_df = pd.DataFrame(pred_series, columns=["score"])

        if pred_df.empty:
            raise ValueError("Generated predictions are empty")

        logger.info(
            f"Generated {len(pred_df)} predictions for segment '{prediction_segment}'"
        )

        # Prepare labels for the same segment
        try:
            label_df = dataset.prepare(
                prediction_segment, col_set="label", data_key=DataHandlerLP.DK_R
            )

            if label_df is not None and not label_df.empty:
                logger.info(
                    f"Extracted {len(label_df)} labels for segment '{prediction_segment}'"
                )
            else:
                logger.warning("Label data is empty or None")
                label_df = None
        except Exception as e:
            logger.warning(f"Could not prepare labels: {e}")
            label_df = None

        # Start a new recorder for saving predictions
        prediction_experiment_name = f"Predictions_{experiment_name}"

        with R.start(experiment_name=prediction_experiment_name):
            current_recorder = R.get_recorder()

            logger.info(f"Saving predictions to new recorder: {current_recorder.id}")

            # Save pred.pkl
            current_recorder.save_objects(**{"pred.pkl": pred_df})

            # Save label.pkl if available
            if label_df is not None:
                current_recorder.save_objects(**{"label.pkl": label_df})

            # Log some metadata
            R.log_params(
                source_experiment=experiment_name,
                source_recorder_id=recorder_id,
                prediction_segment=prediction_segment,
                num_predictions=len(pred_df),
                model_object_used=model_object_name,
            )

            new_recorder_id = current_recorder.id

        return {
            "source_experiment": experiment_name,
            "source_recorder_id": recorder_id,
            "new_experiment": prediction_experiment_name,
            "new_recorder_id": new_recorder_id,
            "segment_predicted": prediction_segment,
            "num_predictions": len(pred_df),
            "num_labels": len(label_df) if label_df is not None else 0,
            "model_object_used": model_object_name,
            "status": "success",
            "message": f"Successfully generated {len(pred_df)} predictions and saved pred.pkl and label.pkl to recorder {new_recorder_id}",
        }

    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate predictions: {str(e)}",
        }


@tool
async def calculate_signal_metrics_tool(
    experiment_name: str, recorder_id: str
) -> Dict[str, Any]:
    """
    Calculates signal metrics like IC and Rank IC from predictions stored in Qlib recorder.

    Args:
        experiment_name: The experiment name where predictions are stored.
        recorder_id: The recorder ID containing pred.pkl and label.pkl.
    Returns:
        A dictionary containing calculated metrics (e.g., IC, Rank IC) and status.
    """
    try:
        logger.info(
            f"Calculating signal metrics from experiment: {experiment_name}, recorder: {recorder_id}"
        )

        # Load the recorder containing predictions and labels
        recorder = R.get_recorder(
            experiment_name=experiment_name, recorder_id=recorder_id
        )

        # Load predictions
        try:
            pred_df = recorder.load_object("pred.pkl")
            pred_series = _extract_series_from_output(pred_df, col_name="score")

            if pred_series.empty:
                raise ValueError("Loaded predictions are empty")

            logger.info(f"Loaded {len(pred_series)} predictions")

        except Exception as e:
            raise ValueError(f"Could not load pred.pkl: {e}")

        # Load labels
        try:
            label_df = recorder.load_object("label.pkl")

            # Try different label column names
            label_series = None
            possible_label_cols = ["LABEL0", "label", "target"]

            for col_name in possible_label_cols:
                label_series = _extract_series_from_output(label_df, col_name=col_name)
                if not label_series.empty:
                    logger.info(
                        f"Successfully extracted labels using column: {col_name}"
                    )
                    break

            if label_series is None or label_series.empty:
                # Try using the first column as fallback
                label_series = _extract_series_from_output(label_df, default_col_idx=0)
                if label_series.empty:
                    raise ValueError("Could not extract labels from label.pkl")

            logger.info(f"Loaded {len(label_series)} labels")

        except Exception as e:
            raise ValueError(f"Could not load or extract labels from label.pkl: {e}")

        # Calculate IC metrics
        metrics = _calculate_ic_metrics(pred_series, label_series)

        # Log metrics to the same recorder
        R.log_metrics(**metrics)

        metrics = {
            "IC": metrics["IC"],
            "Rank IC": metrics["Rank IC"],
            "ICIR": metrics["ICIR"],
            "Rank ICIR": metrics["Rank ICIR"],
            "num_predictions": len(pred_series),
            "num_labels": len(label_series),
            "experiment_name": experiment_name,
            "recorder_id": recorder_id,
            "status": "success",
        }

        logger.info(
            f"Calculated metrics - IC: {metrics['IC']:.6f}, Rank IC: {metrics['Rank IC']:.6f}"
        )

        metrics["message"] = (
            f"Successfully calculated signal metrics (IC: {metrics['IC']:.6f}, Rank IC: {metrics['Rank IC']:.6f}) from {len(pred_series)} predictions and {len(label_series)} labels"
        )

        return metrics

    except Exception as e:
        logger.error(f"Error calculating signal metrics: {e}")
        return {
            "ic": None,
            "rank_ic": None,
            "correlation": None,
            "status": "error",
            "message": f"Failed to calculate signal metrics: {str(e)}",
        }


# List of tools for this worker agent
prediction_generation_tools = [
    generate_model_predictions_tool,
    calculate_signal_metrics_tool,
]
