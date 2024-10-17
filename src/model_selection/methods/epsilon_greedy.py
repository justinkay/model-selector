import numpy as np
from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
from utils.math_utils import distribution, entropy


class EpsilonGreedyMethodConfig(MethodConfig):
    def __init__(self, method_config: dict) -> None:
        super().__init__(method_config)
        self.epsilon = method_config["epsilon"]


class EpsilonGreedy(ModelSelectionStrategy):
    strategy_type = "epsilon_greedy"

    def __init__(self, method_config: dict) -> None:
        super().__init__(EpsilonGreedyMethodConfig(method_config))
        self.epsilon = self.method_config.epsilon


    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Lt = []
        Ut = data.copy().tolist()

        num_classes = len(classes)
        indices = np.arange(len(data)).tolist()
        
        entropy_vals = []
        for instance in predictions:
            dist = distribution(instance, num_classes)
            entropy_vals.append(entropy(dist))

        for t in range(budget):
            # do random action with probability epsilon, otherwise choose QBC entropy
            if np.random.rand() < self.method_config.epsilon:
                index = np.random.randint(0, len(entropy_vals))
            else:
                index = np.argmax(entropy_vals)
            
            x = Ut.pop[index]
            y = oracle[indices[index]]

            entropy_vals.pop(index)
            
            Lt.append((indices.pop(index), x, y))
        
        return Lt
