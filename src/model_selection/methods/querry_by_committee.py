from enum import Enum
from scipy.stats import entropy
import numpy as np
from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
from utils.math_utils import confidence_margin, distribution


class QBCSelectionMode(Enum):
    VOTE_MARGIN = 1
    ENTROPY = 2
    LEAST_CONFIDENCE = 3


class QBCConfig(MethodConfig):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.qbc_strategy = QBCSelectionMode[str(config["qbcStrategy"]).upper()] if "qbcStrategy" in config else QBCSelectionMode.ENTROPY


class QBC(ModelSelectionStrategy):

    strategy_type = "qbc"

    def __init__(self, method_config: dict) -> None:
        super().__init__(QBCConfig(method_config))

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Lt = []

        num_instances = len(data)
        num_classes = len(classes)
        num_models = predictions.shape[1]

        datapoint_values = np.zeros(num_instances)
        for i in range(num_instances):
            if self.method_config.qbc_strategy == QBCSelectionMode.VOTE_MARGIN:
                votes = np.bincount(predictions[i,:], minlength=num_classes) # get votes for each class (Cx1
                unique_votes = np.unique(votes)
                if len(unique_votes) == 1:
                    datapoint_values[i] = 0 # all models agree -> no uncertainty
                else:
                    datapoint_values[i] = confidence_margin(votes / num_models)
            elif self.method_config.qbc_strategy == QBCSelectionMode.ENTROPY:
                dist = distribution(predictions[i,:], n=num_classes) # get distribution of votes over data point i
                datapoint_values[i] = entropy(dist, base=2)
            elif self.method_config.qbc_strategy == QBCSelectionMode.LEAST_CONFIDENCE:
                dist = distribution(predictions[i,:], n=num_classes)
                datapoint_values[i] = 1 - np.max(dist)
            else:
                raise ValueError("Invalid QBC strategy")

        samples = np.argsort(-datapoint_values)[:budget] # argsort(-entropies) sorts descending (highest uncertainty first

        for i in samples:
            x = data[i]
            y = oracle[i]
            Lt.append((i, x, y))

        return Lt