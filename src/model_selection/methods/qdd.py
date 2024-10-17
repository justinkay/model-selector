from typing import Any

import numpy as np
from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
from utils.fs_utils import load_embedding
from utils.math_utils import distribution, entropy


class QDDConfig(MethodConfig):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.gamma = config["gamma"] if "gamma" in config else 0.5
        self.beta = config["beta"] if "beta" in config else 0.5
        if "embedding_file" not in config:
            raise KeyError(f"QDD algorithm needs an embedding file to run but none was provided!")
        self.embedding_file = config["embedding_file"]
        self.k = config["k"] if "k" in config else None
    

class QDD(ModelSelectionStrategy):
    strategy_type = "qdd"

    def __init__(self, method_config: MethodConfig) -> None:
        super().__init__(QDDConfig(method_config))
        self.gamma = self.method_config.gamma
        self.beta = self.method_config.beta
        self.k = self.method_config.k
        self.embedding = load_embedding(self.method_config.embedding_file)

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int) -> Any:
        Lt = []
        Lt_indices = []
        Ut = data.copy().tolist()

        embedding_diversity = self.embedding.copy() 
        embedding_density = self.embedding.copy()
        np.fill_diagonal(embedding_diversity, np.inf)
        np.fill_diagonal(embedding_density, -np.inf) # set diagonal to 0 to ignore self comparison

        embedding_density = -np.sort(-embedding_density, axis=1)

        num_instances = len(data)
        num_classes = len(classes)
        indices = np.arange(len(data)).tolist()
        
        entropies = np.zeros(num_instances)
        for i in range(num_instances):
            predictions_i = predictions[i,:] # get predictions for each model (Mx1)
            dist = distribution(predictions_i, n=num_classes) # get the distribution of the predictions for each instance
            entropies[i] = entropy(dist)

        # UNSURE: If K-NN density is calculated over Lt and Ut then it is constant, otherwise move it into the loop
        k = np.min(self.embedding.shape[0] - 1, self.k) # set k of K-NN to number of all other instances (ignore self comparison -> np.inf -> last element)
        density = self._density(indices, embedding_density, k) # Paper equation (8) 

        for t in range(budget):
            diversity = self._diversity(indices, Lt_indices, embedding_diversity) # Paper equation (7)
            x_scores = (1 - self.gamma - self.beta) * entropies + self.gamma * density + self.beta * diversity # Paper equation (9)
            x_idx = np.argmax(x_scores)

            x = Ut.pop(x_idx)
            i = indices.pop(x_idx)
            y = oracle[i]

            entropies = np.delete(entropies, x_idx)
            density = np.delete(density, x_idx)
            Lt.append((i, x, y))
            Lt_indices.append(i)

        return Lt

    def _density(self, Ut_idxs, embedding, k=None):
        e_selected = embedding[Ut_idxs][:,:k]
        return np.sum(e_selected, axis=1) / e_selected.shape[1]
    
    def _diversity(self, Ut_idxs, Lt_idxs, embedding):
        if len(Lt_idxs) == 0:
            return np.zeros(len(Ut_idxs))
        else:
            return np.min(embedding[Ut_idxs][:,Lt_idxs], axis=1) # get all comparisons from x' in Ut to all x in Lt (shortest linkage selection)
