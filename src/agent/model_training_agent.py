"""
Model Training Worker Agent for Alpha Modeling

Responsible for managing the model training process, including model configuration generation,
initialization, training, logging, and saving.
"""

from typing import Dict, Any, Optional
import uuid
import logging
import datetime
import copy

from langchain_core.tools import tool

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

# QLib imports for training
from qlib.workflow import R
from qlib.utils import init_instance_by_config, flatten_dict

from .config import (
    MONGO_URI,
    MODEL_ZOO_DB_NAME,
    MODEL_TEMPLATES_COLLECTION_NAME,
)

# Initialize logger for this module
logger = logging.getLogger("AlphaModeling.ModelTrainingWorker")


def get_db_client_and_collection():
    """Helper function to get MongoDB client and collection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # Added timeout
        client.admin.command("ping")  # Verify connection
        db = client[MODEL_ZOO_DB_NAME]
        collection = db[MODEL_TEMPLATES_COLLECTION_NAME]
        return client, collection
    except PyMongoError as e:
        logger.error("Failed to connect to MongoDB: %s", e)
        return None, None


# Tool function implementation


@tool
async def access_model_zoo_from_db(
    model_name: Optional[str] = None,
    feature_set_name: Optional[str] = None,
    prediction_target: Optional[str] = None,
    template_name: Optional[str] = None,
    version: Optional[str] = None,
    status: str = "active",  # Default to searching for active templates
) -> Dict[str, Any]:
    """
    Accesses the Model Zoo database (MongoDB) to retrieve a specific model template
    based on the provided criteria. It prioritizes finding an exact match or the
    latest active version if multiple templates fit broader criteria.

    Args:
        model_name: The name of the model (e.g., 'XGBoost', 'LightGBM').
        feature_set_name: The name of the feature set the template is designed for (e.g., 'Alpha101').
        prediction_target: The prediction target of the template (e.g., '1d_return').
        template_name: The exact unique name of the template, if known.
        version: The specific version of the template to retrieve (e.g., '1.0.0').
        status: The status of the template to search for (default: 'active').

    Returns:
        A dictionary containing the model template details (including 'template_config')
        if found, otherwise an error message.
    """
    logger.info(
        "Accessing Model Zoo DB with criteria: template_name='%s', base_model_name='%s', feature_set_name='%s', "
        "prediction_target='%s', version='%s', status='%s'",
        template_name,
        model_name,
        feature_set_name,
        prediction_target,
        version,
        status,
    )

    client, collection = get_db_client_and_collection()
    if collection is None:
        return {
            "error": "Failed to connect to Model Zoo database.",
            "status": "db_connection_error",
        }

    query = {}
    if template_name:
        query["template_name"] = template_name
    if model_name:
        query["model"] = model_name
    if feature_set_name:
        query["feature_set_name"] = feature_set_name
    if prediction_target:
        query["prediction_target"] = prediction_target
    if version:
        query["version"] = version
    # Always include status unless explicitly set to None (which is not an option here)
    if status:
        query["status"] = status

    if not query:  # Prevent searching with an empty query
        return {
            "error": "No search criteria provided to access model zoo.",
            "status": "no_criteria",
        }

    logger.debug("Constructed MongoDB query: %s", query)

    try:
        # If version is not specified, we might want the latest (highest version or most recently updated)
        # For simplicity, if multiple match, we'll sort by version and updated_at.
        # Proper semantic version sorting is more complex.
        sort_criteria = []
        if not version:  # If no specific version, try to get the "best"
            # Sorting by version string descending might work for simple cases like "1.0.0", "1.1.0"
            # but not for "1.10.0" vs "1.2.0".
            # A more robust way would be to parse versions or have a separate "is_latest_active" flag.
            # For now, let's sort by 'updated_at' to get the most recent one if multiple active versions match.
            sort_criteria.append(("updated_at", -1))  # -1 for descending
            # Try to sort by version string as a secondary
            sort_criteria.append(("version", -1))

        query_cursor = collection.find(query)
        if sort_criteria:  # Only apply sort if there are criteria
            query_cursor = query_cursor.sort(sort_criteria)

        found_templates = list(query_cursor.limit(5))  # Limit to 5 if many match

        if not found_templates:
            logger.warning("No template found in Model Zoo DB for query: %s", query)
            # Try a broader search if specific criteria failed
            if (
                len(query) > 1 and "template_name" not in query
            ):  # if it wasn't a direct name lookup
                broader_query = (
                    {"model": model_name, "status": "active"}
                    if model_name
                    else {"status": "active"}
                )
                alternatives = list(
                    collection.find(
                        broader_query,
                        {
                            "template_name": 1,
                            "model": 1,
                            "version": 1,
                            "description": 1,
                        },
                    ).limit(5)
                )
                if alternatives:
                    alt_suggestions = [
                        {
                            "name": a.get("template_name"),
                            "model": a.get("model"),
                            "v": a.get("version"),
                            "desc": a.get("description"),
                        }
                        for a in alternatives
                    ]
                    return {
                        "message": f"No exact template found for your specific criteria. Found {len(alternatives)} other potentially relevant active templates.",
                        "suggestions": alt_suggestions,
                        "status": "not_found_suggestions_available",
                    }

            return {
                "message": "No template found matching the criteria.",
                "status": "not_found",
            }

        # If we found templates, select the first one (which should be the "best" due to sorting)
        template_doc = found_templates[0]

        # Convert ObjectId to string if they exist, for JSON serialization if needed by agent
        if "_id" in template_doc and isinstance(template_doc["_id"], ObjectId):
            template_doc["_id"] = str(template_doc["_id"])

        logger.info(
            "Found template: %s, version: %s",
            template_doc.get("template_name"),
            template_doc.get("version"),
        )

        result = {
            "message": f"Successfully retrieved template '{template_doc.get('template_name')}' v{template_doc.get('version')}.",
            "template_data": template_doc,  # Return the whole document
            "status": "success",
        }
        if len(found_templates) > 1:
            result["warning"] = (
                f"Found {len(found_templates)} templates matching criteria; returning the most recent/highest version one. Consider a more specific query if this is not the desired one."
            )
            result["other_matches_summary"] = [
                {"name": t.get("template_name"), "version": t.get("version")}
                for t in found_templates[1:]
            ]

        return result

    except PyMongoError as e:
        logger.error("Error querying Model Zoo DB: %s", e, exc_info=True)
        return {
            "error": f"An error occurred while querying the database: {str(e)}",
            "status": "db_query_error",
        }
    finally:
        if client:
            client.close()


@tool
async def generate_model_configuration(
    model_type: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    template_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates or refines a Qlib model configuration.
    This tool REQUIRES template_config from model zoo - it will not generate configurations from scratch.
    The workflow should first call access_model_zoo_from_db to get a template.

    Args:
        model_type: Base type of model, e.g., 'XGBModel'. Used for validation only.
        hyperparameters: Specific hyperparameters to override or add to the template config's kwargs. This MUST be a Python dictionary.
        template_config: A full Qlib model configuration (REQUIRED - from access_model_zoo_from_db). This argument MUST be a Python dictionary object, NOT a string representation of a dictionary.

    Returns:
        A dictionary containing the complete Qlib model configuration.
    """
    logger.info(
        "Generating model configuration. model_type=%s, provided_template_exists=%s",
        model_type,
        template_config is not None,
    )

    # Log incoming hyperparameters
    if hyperparameters:
        logger.info("Incoming hyperparameters: %s", hyperparameters)

    # Require template_config - don't allow LLM generation
    if not template_config:
        error_msg = (
            "ERROR: template_config is required. You must first call 'access_model_zoo_from_db' "
            "to retrieve a model template from the database before generating the configuration. "
            "This tool does not generate configurations from scratch using LLM."
        )
        logger.error(error_msg)
        return {
            "error": error_msg,
            "status": "template_required",
            "suggestion": f"Call access_model_zoo_from_db with model_name='{model_type}' first",
        }

    logger.info(
        "Using provided template_config for %s as base.",
        template_config.get("class", model_type),
    )

    # Replace JSON round-trip with proper deep copy
    try:
        final_config = copy.deepcopy(template_config)
    except Exception as e:
        error_msg = f"Invalid template_config structure: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "invalid_template_format"}

    # Override/add specific hyperparameters if provided
    if hyperparameters:
        if "kwargs" not in final_config:
            final_config["kwargs"] = {}

        final_config["kwargs"].update(hyperparameters)
        logger.info("Applied custom hyperparameters: %s", hyperparameters)

    # Validate the final configuration
    if not validate_model_config(final_config):
        logger.error("Template config is invalid after applying overrides.")
        return {
            "error": "Template config is invalid after applying overrides.",
            "status": "invalid_template_config",
        }

    # Validate model_type matches if provided
    if model_type and final_config.get("class") != model_type:
        logger.warning(
            f"Model type mismatch: requested {model_type}, template has {final_config.get('class')}"
        )

    logger.info(
        "Successfully generated model configuration from template for %s",
        final_config.get("class"),
    )
    return final_config


@tool
async def train_model(
    # Use 'm_config' instead of 'model_config' to avoid Pydantic V2 'model_config' directive name conflict during V1→V2 compatibility conversion
    #  (prevents TypeError: 'ellipsis' object is not iterable)
    m_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train a model using QLib framework with proper experiment tracking.

    Args:
        m_config: QLib model configuration with class, module_path, and kwargs. This argument MUST be a Python dictionary object, NOT a string representation of a dictionary.
        dataset_config: QLib dataset configuration. This argument MUST be a Python dictionary object, NOT a string representation of a dictionary.

    Returns:
        Dictionary containing training results, metrics, and file paths
    """
    logger.info(
        "Starting training for model: %s", m_config.get("class", "UnknownModel")
    )

    # Fix LangChain serialization issue again: parameters may have been converted back to float
    # during tool call transmission between generate_model_configuration and train_model
    if m_config.get("class") == "XGBModel" and "kwargs" in m_config:
        kwargs = m_config["kwargs"]
        # XGBoost integer parameters that get incorrectly converted to float by LangChain
        int_params = [
            "max_depth",
            "n_estimators",
            "min_child_weight",
            "max_delta_step",
            "num_parallel_tree",
            "subsample_freq",
        ]

        for param in int_params:
            if (
                param in kwargs
                and isinstance(kwargs[param], float)
                and kwargs[param].is_integer()
            ):
                old_value = kwargs[param]
                kwargs[param] = int(kwargs[param])
                logger.info(
                    f"Fixed LangChain serialization in train_model: {param} {old_value} -> {kwargs[param]} (float->int)"
                )

    try:
        # Generate unique identifiers and setup
        model_id = str(uuid.uuid4())
        model_class = m_config.get("class", "model")
        timestamp = datetime.datetime.now().isoformat()
        experiment_name = f"{model_class}_{model_id[:8]}_training"

        # Initialize model and dataset from configurations
        logger.info("Initializing model and dataset from configurations")
        model = init_instance_by_config(m_config)
        dataset = init_instance_by_config(dataset_config)

        # Training results to be populated
        training_results = {
            "model_id": model_id,
            "model_class": model_class,
            "experiment_name": experiment_name,
            "status": "training",
            "timestamp": timestamp,
        }

        # Start QLib recorder for experiment tracking
        logger.info("Starting QLib recorder with experiment name: %s", experiment_name)
        with R.start(experiment_name=experiment_name):
            # Log model parameters for traceability
            logger.info("Logging model parameters and configurations")
            R.log_params(**flatten_dict({"model": m_config, "dataset": dataset_config}))

            # Train the model
            logger.info("Training model with dataset")
            model.fit(dataset)

            # Save trained model
            R.save_objects(**{"trained_model.pkl": model})

            recorder_id = R.get_recorder().id

            # Update training results with success information
            training_results.update(
                {
                    "status": "completed",
                    "recorder_id": recorder_id,
                    "model_config_used": m_config,
                    "dataset_config_summary": {
                        "handler_class": dataset_config.get("kwargs", {})
                        .get("handler", {})
                        .get("class"),
                        "segments": dataset_config.get("kwargs", {}).get("segments"),
                    },
                    "completion_timestamp": datetime.datetime.now().isoformat(),
                }
            )

        logger.info("Model %s training completed successfully", model_class)

        return {
            "model_config": m_config,
            "training_log": training_results,
            "recorder_id": recorder_id,
            "experiment_name": experiment_name,
            "status": "success",
            "message": f"Model {model_class} trained successfully. Experiment: {experiment_name}, Recorder ID: {recorder_id}",
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            f"Training failed for model {m_config.get('class', 'Unknown')}: {str(e)}",
            exc_info=True,
        )

        # Return error information
        return {
            "model_config": m_config,
            "training_log": {
                "model_id": model_id if "model_id" in locals() else str(uuid.uuid4()),
                "model_class": m_config.get("class", "Unknown"),
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "status": "error",
            "message": f"Training failed for model {m_config.get('class', 'Unknown')}: {str(e)}",
        }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate that the model configuration has the required structure"""
    # Check essential fields
    required_fields = ["class", "module_path", "kwargs"]
    if not all(field in config for field in required_fields):
        return False

    return True


# List of tools for this worker agent
model_training_tools = [
    access_model_zoo_from_db,
    generate_model_configuration,
    train_model,
]
