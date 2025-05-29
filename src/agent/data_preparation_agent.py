"""
Data Preparation Worker Agent for Alpha Modeling

Responsible for generating a standardized Dataset Configuration Dictionary
for Qlib DatasetH objects based on provided instructions.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .config import GOOGLE_API_KEY, AGENT_CONFIG


DATA_PREPARATION_SYSTEM_PROMPT = """
You are a Data Preparation Agent expert in Qlib library data handling.
Your task is to generate valid dataset configuration dictionaries for Qlib's DatasetH objects.

IMPORTANT RULES:
1. You must return a valid JSON dictionary following the Qlib schema.
2. Focus only on creating the dataset configuration based on the user's inputs.
3. The config must include segments for training, validation, and testing based on provided dates.
4. Use the provided parquet file path for data sources.
5. Include default data processors if none are specified.

Follow this exact schema for the DatasetH config:
{{
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {{
        "handler": {{
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {{
                "data_loader": {{
                    "class": "StaticDataLoader",
                    "module_path": "qlib.data.dataset.loader",
                    "kwargs": {{
                        "config": <data_path>                    
                    }}
                }},
                "learn_processors": [
                    {{"class": "DropnaLabel"}}
                ],
                "infer_processors": [
                    {{"class": "ZScoreNorm",
                     "kwargs": {{"fields_group": "feature",
                                "fit_start_time": "<train_start_time>",
                                "fit_end_time": "<train_end_time>"}}
                    }},
                    {{"class": "Fillna"}}
                ],
                "start_time": "<train_start_time>",
                "end_time": "<test_end_time>"
            }}
        }},
        "segments": {{
            "train": ("<train_start_date>", "<train_end_date>"),
            "valid": ("<valid_start_date>", "<valid_end_date>"),
            "test": ("<test_start_date>", "<test_end_date>")
        }}
    }}
}}

Your output must be a valid JSON object with the complete DatasetH configuration dictionary.
"""


# Tool function implementation - this replaces the _process_task method
@tool
def generate_qlib_dataset_config(
    parquet_file_path: Optional[str] = None,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    valid_start_date: Optional[str] = None,
    valid_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    learn_processors_config: Optional[List[Dict[str, Any]]] = None,
    infer_processors_config: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generates a Qlib DatasetH configuration dictionary based on the provided parameters.
    If 'parquet_file_path' is not provided, a system default path ('data/precomputed_features/tw50_alpha101_label_1d.parquet') will be used.
    Date parameters (train_start_date, train_end_date, etc.) will also use system defaults if not specified.
    Learn processors and infer processors will use default configurations if not provided.

    Args:
        parquet_file_path (Optional[str]): Path to the parquet file containing the data. Defaults to a system preconfigured path if None.
        train_start_date (Optional[str]): Start date for training data (YYYY-MM-DD). Uses system default if None.
        train_end_date (Optional[str]): End date for training data (YYYY-MM-DD). Uses system default if None.
        valid_start_date (Optional[str]): Start date for validation data (YYYY-MM-DD). Uses system default if None.
        valid_end_date (Optional[str]): End date for validation data (YYYY-MM-DD). Uses system default if None.
        test_start_date (Optional[str]): Start date for test data (YYYY-MM-DD). Uses system default if None.
        test_end_date (Optional[str]): End date for test data (YYYY-MM-DD). Uses system default if None.
        learn_processors_config (Optional[List[Dict[str, Any]]]): Optional list of learn processors configuration. Uses default if None.
        infer_processors_config (Optional[List[Dict[str, Any]]]): Optional list of infer processors configuration. Uses default if None.

    Returns:
        A dictionary containing the complete DatasetH configuration.
    """
    # Use default values if none provided
    from .config import (
        DEFAULT_PARQUET_FILE_PATH, DEFAULT_TRAIN_START, DEFAULT_TRAIN_END,
        DEFAULT_VALID_START, DEFAULT_VALID_END, DEFAULT_TEST_START,
        DEFAULT_TEST_END, DEFAULT_LEARN_PROCESSORS, DEFAULT_INFER_PROCESSORS
    )

    parquet_file_path = parquet_file_path or DEFAULT_PARQUET_FILE_PATH
    train_start_date = train_start_date or DEFAULT_TRAIN_START
    train_end_date = train_end_date or DEFAULT_TRAIN_END
    valid_start_date = valid_start_date or DEFAULT_VALID_START
    valid_end_date = valid_end_date or DEFAULT_VALID_END
    test_start_date = test_start_date or DEFAULT_TEST_START
    test_end_date = test_end_date or DEFAULT_TEST_END
    learn_processors = learn_processors_config or DEFAULT_LEARN_PROCESSORS
    infer_processors = infer_processors_config or DEFAULT_INFER_PROCESSORS

    # Create instruction for the LLM
    instruction = f"""
    Create a Qlib dataset configuration with the following parameters:
    - Parquet file path: {parquet_file_path}
    - Train dates: {train_start_date} to {train_end_date}
    - Validation dates: {valid_start_date} to {valid_end_date}
    - Test dates: {test_start_date} to {test_end_date}
    - Learn processors: {json.dumps(learn_processors)}
    - Infer processors: {json.dumps(infer_processors)}
    """

    # Initialize LLM for this function call
    config = AGENT_CONFIG["data_preparation"]
    llm = ChatGoogleGenerativeAI(
        model=config.get("model_name"),
        google_api_key=GOOGLE_API_KEY,
        temperature=config.get("temperature", 0.1),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", DATA_PREPARATION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    chain = prompt | llm

    # Get response from LLM (using synchronous invoke)
    response = chain.invoke({"input": instruction})
    llm_response = response.content

    # Extract JSON from the response
    try:
        # Sometimes the LLM includes markdown code blocks or explanations
        if "```json" in llm_response:
            json_str = llm_response.split(
                "```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            json_str = llm_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = llm_response.strip()

        dataset_config = json.loads(json_str)

        # Verify the configuration has the required structure
        if not validate_dataset_config(dataset_config):
            raise ValueError(
                "Generated dataset configuration is incomplete or invalid")

        return dataset_config

    except Exception as e:
        logger = logging.getLogger("AlphaModeling.DataPreparationTool")
        logger.error(f"Error parsing LLM response: {str(e)}")
        logger.debug(f"Raw LLM response: {llm_response}")
        raise ValueError(
            f"Failed to generate valid dataset configuration: {str(e)}")


def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate that the dataset configuration has the required structure"""
    # Check essential fields
    required_fields = ["class", "module_path", "kwargs"]
    if not all(field in config for field in required_fields):
        return False

    # Check for segments
    if "kwargs" not in config or "segments" not in config["kwargs"]:
        return False

    segments = config["kwargs"]["segments"]
    required_segments = ["train", "valid", "test"]
    if not all(segment in segments for segment in required_segments):
        return False

    return True


# List of tools for this worker agent
data_preparation_tools = [generate_qlib_dataset_config]
