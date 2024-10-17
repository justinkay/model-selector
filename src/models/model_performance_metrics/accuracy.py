from models.interface_inferable import Inferable
from models.model_performance_metric import ModelPerformanceMetric
from models.model_results import ModelResults


class Accuracy(ModelPerformanceMetric):
    metric_name = "accuracy"

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, models: list[Inferable], model_performance_history: ModelResults):
        models_sorted = sorted(models, key=lambda model: model_performance_history.get_accuracy(model.id), reverse=True)
        return models_sorted