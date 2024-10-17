import models.model_performance_metrics # !!! import that automatically registers all performance metrics
from models.model_performance_metric import AVAILABLE_METRICS


def get_model_performance_metric(metric: str):
    """
    Returns the performance metric with the given name.

    Args:
        metric (str): name of the metric

    Returns:
        PerformanceMetric: performance metric
    """
    if metric not in AVAILABLE_METRICS:
        raise ValueError(f"Performance metric {metric} not available.")
    return AVAILABLE_METRICS[metric]()