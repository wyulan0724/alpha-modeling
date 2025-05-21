"""
Configuration for Alpha Modeling Agents
"""
import os
from typing import Dict, Any

# API configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Model configuration
DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048


# Agent configuration
AGENT_CONFIG = {
    "orchestrator": {
        "name": "OrchestratorAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "data_preparation": {
        "name": "DataPreparationAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": 0.1,  # Lower temperature for deterministic configs
    },
    "model_training": {
        "name": "ModelTrainingAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
    },
    "prediction_generation": {
        "name": "PredictionGenerationAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
    },
    "backtesting": {
        "name": "BacktestingAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
    },
    "analysis_reporting": {
        "name": "AnalysisReportingAgent",
        "model_name": DEFAULT_MODEL_NAME,
        "temperature": DEFAULT_TEMPERATURE,
    }
}

# Output directories
OUTPUT_DIRS = {
    "models": "outputs/models",
    "predictions": "outputs/predictions",
    "backtests": "outputs/backtests",
    "reports": "outputs/reports"
}

# Create output directories
for directory in OUTPUT_DIRS.values():
    os.makedirs(directory, exist_ok=True)

# Data preparation configuration
DEFAULT_PARQUET_FILE_PATH = 'data/precomputed_features/tw50_alpha101_label_1d.parquet'
DEFAULT_TRAIN_START = "2014-12-31"
DEFAULT_TRAIN_END = "2021-12-31"
DEFAULT_VALID_START = "2022-01-01"
DEFAULT_VALID_END = "2022-12-31"
DEFAULT_TEST_START = "2023-01-01"
DEFAULT_TEST_END = "2024-12-27"
DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}]
DEFAULT_INFER_PROCESSORS = [
    {"class": "ZScoreNorm",
     "kwargs": {"fields_group": "feature",
                "fit_start_time": DEFAULT_TRAIN_START,
                "fit_end_time": DEFAULT_TRAIN_END}},
    {"class": "Fillna"},
]
