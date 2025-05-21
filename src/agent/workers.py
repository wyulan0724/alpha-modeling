"""
Worker agents for Alpha Modeling
"""
from .data_preparation_worker import DataPreparationWorker
from .model_training_worker import ModelTrainingWorker

WORKER_REGISTRY = {
    "DataPreparationWorker": DataPreparationWorker,
    "ModelTrainingWorker": ModelTrainingWorker,
    # to be added
}


def get_worker(worker_name: str, **kwargs):
    """
    Get a worker agent by name

    Args:
        worker_name: Name of the worker agent
        **kwargs: Additional arguments to pass to the worker constructor

    Returns:
        Worker agent instance
    """
    if worker_name not in WORKER_REGISTRY:
        raise ValueError(f"Unknown worker: {worker_name}")

    worker_class = WORKER_REGISTRY[worker_name]
    return worker_class(**kwargs)
