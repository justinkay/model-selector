from model_selection.method_config import MethodConfig
import numpy as np
from model_selection.model_selection_strategy import ModelSelectionStrategy


class RandomSelection(ModelSelectionStrategy):
    strategy_type = "random"

    def __init__(self, method_config: dict) -> None:
        super().__init__(MethodConfig(method_config))

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Ut = data.copy().tolist()
        Lt = []
        indices = list(range(len(Ut)))

        for _ in range(budget):
            i = np.random.randint(len(Ut)) # get a random index
            x = Ut.pop(i) # get a random datapoint from the unlabelled set
            y = oracle[indices[i]] # query the oracle for the label
            Lt.append((indices.pop(i),x,y))

        return Lt
