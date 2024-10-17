


from enum import Enum
import json
import os
import numpy as np
from benchmark.benchmark_config import ExperimentConfig
from data_provider.interface_data_provider import DataProvider
from models.interface_inferable import Inferable


class OptimalEpsilonMetric(Enum):
    ALL = "all"
    FASTEST = "fastest"
    SUCCESS_DIFFERENCE = "success_diff"
    SUCCESS_AVG = "success_avg"


class GridSearchConfig:

    def __init__(self, config: ExperimentConfig,
                        epsilons: list[int],
                        step_size: int,
                        epsilon_metric: OptimalEpsilonMetric,
                        epsilon_metric_threshold: float,
                        noisy_oracle: str,
                        noisy_oracle_enhancement: str,
                        noisy_oracle_enhancement_cnt: int,
                        recursive: bool,
                        raw: dict) -> None:
        self.experiment_config: ExperimentConfig = config
        self.epsilons = epsilons
        self.step_size = step_size
        self.epsilon_metric = epsilon_metric
        self.epsilon_metric_threshold = epsilon_metric_threshold
        self.noisy_oracle = noisy_oracle
        self.noisy_oracle_enhancement = noisy_oracle_enhancement
        self.noisy_oracle_enhancement_cnt = noisy_oracle_enhancement_cnt
        self.recursive = recursive
        self.raw = raw

    def to_json(self):
        return self.raw

    def from_json(data: dict):
        if not "epsilon_range" in data:
            epsilon_range = np.arange(0.35, 0.5, 0.01)
        else:
            epsilon_range = np.arange(data["epsilon_range"]["min"], data["epsilon_range"]["max"]+0.5*data["epsilon_range"]["step"], data["epsilon_range"]["step"])
            if not 'finegrained' in data['experiment']['name']:  # always check 0.49 for broad epsilon range
                 epsilon_range = np.append(epsilon_range, 0.49)
        
        return GridSearchConfig(ExperimentConfig.from_dict(data), 
                                epsilon_range, 
                                data["step_size"] if "step_size" in data else 10, 
                                OptimalEpsilonMetric(data["epsilon_metric"]) if "epsilon_metric" in data else OptimalEpsilonMetric.SUCCESS_DIFFERENCE,
                                data["epsilon_metric_threshold"] if "epsilon_metric_threshold" in data else 0.9,
                                data["noisy_oracle"] if "noisy_oracle" in data else None,
                                data["noisy_oracle_enhancement"] if "noisy_oracle_enhancement" in data else None,
                                data["noisy_oracle_enhancement_cnt"] if "noisy_oracle_enhancement_cnt" in data else 50,
                                data["recursive"] if "recursive" in data else False,
                                data)
    
    def __str__(self) -> str:
        base = str(self.experiment_config)
        return f"{base}\nGRIDSEARCH CONFIG:\nEpsilon range: {self.epsilons}\nStep size: {self.step_size}\nEpsilon metric: {self.epsilon_metric}\nEpsilon metric threshold: {self.epsilon_metric_threshold}\nRecursive: {self.recursive}\n====================================================================="