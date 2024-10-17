from typing import Any
import numpy as np
from model_selection.method_config import MethodConfig
from data_provider.interface_data_provider import DataProvider
from abc import ABC, abstractmethod


AVAILABLE_STRATEGIES = {}


class ModelSelectionStrategy(ABC):
    strategy_type = None

    def __init__(self, method_config: MethodConfig) -> None:
        super().__init__()
        self.name = method_config.name
        self.method_config = method_config

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AVAILABLE_STRATEGIES[cls.strategy_type] = cls

    @abstractmethod
    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int) -> Any:
        raise NotImplementedError

