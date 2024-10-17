from abc import ABC, abstractmethod
from models.interface_inferable import Inferable
from models.model_results import ModelResults


AVAILABLE_METRICS = {}

class ModelPerformanceMetric(ABC):
    metric_name: str = None

    def __init__(self) -> None:
        super().__init__()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        AVAILABLE_METRICS[cls.metric_name] = cls

    @abstractmethod
    def __call__(self, models: list[Inferable], model_performance_history: ModelResults):
        raise NotImplementedError