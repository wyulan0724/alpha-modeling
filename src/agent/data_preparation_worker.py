"""
Data Preparation Worker Agent for Alpha Modeling

Responsible for generating a standardized Dataset Configuration Dictionary
for Qlib DatasetH objects based on provided instructions.
"""
from typing import Dict, Any
import json

from .base import BaseWorkerAgent, WorkerInput

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


class DataPreparationWorker(BaseWorkerAgent):
    """
    Data Preparation Worker that creates dataset configurations for Qlib.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="DataPreparationWorker",
            system_prompt=DATA_PREPARATION_SYSTEM_PROMPT,
            **kwargs
        )

    async def _process_task(self, worker_input: WorkerInput) -> Dict[str, Any]:
        """
        Process the data preparation task to generate a dataset configuration.

        Args:
            worker_input: Input containing task details

        Returns:
            Dictionary containing the dataset configuration
        """
        inputs = worker_input["inputs"]

        # Extract parameters from inputs
        parquet_file_path = inputs["parquet_file_path"]
        train_start_date = inputs["train_start_date"]
        train_end_date = inputs["train_end_date"]
        valid_start_date = inputs["valid_start_date"]
        valid_end_date = inputs["valid_end_date"]
        test_start_date = inputs["test_start_date"]
        test_end_date = inputs["test_end_date"]
        learn_processors = inputs["learn_processors_config"]
        infer_processors = inputs["infer_processors_config"]

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

        # Get response from LLM
        llm_response = await self._call_llm(instruction)

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
            if not self._validate_dataset_config(dataset_config):
                raise ValueError(
                    "Generated dataset configuration is incomplete or invalid")

            return {"dataset_config": dataset_config}

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            self.logger.debug(f"Raw LLM response: {llm_response}")
            raise ValueError(
                f"Failed to generate valid dataset configuration: {str(e)}")

    def _validate_dataset_config(self, config: Dict[str, Any]) -> bool:
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
