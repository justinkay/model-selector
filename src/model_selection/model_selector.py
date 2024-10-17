import numpy as np
from model_selection.model_selection_strategy import ModelSelectionStrategy


class ModelSelector:

    def __init__(self, strategy: ModelSelectionStrategy) -> None:
        """
        Args:
            models (Inferable): List of models to select from.
            dataSelector (DataSelector): Data selection algorithm to use.
        """
        super().__init__()  
        self.name = strategy.name
        self.strategy = strategy

    def run(self, data, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget=None):
        if budget is None:
            budget = len(data)
        else:
            budget = min(budget, len(data))

        # remove data points where all the predictions are the same
        p = np.expand_dims(predictions[:,0], axis=1)
        indices = np.where((predictions != p).any(axis=1))[0]
        # indices = np.arange(predictions.shape[0])  # do not remove points in agreement
        filtered_data = data[indices]
        filtered_predictions = predictions[indices]
        filtered_oracle = oracle[indices]

        Lt = self.strategy(filtered_data, filtered_predictions, filtered_oracle, classes, min(budget, len(filtered_data))) # run the model selection strategy

        accuracies = np.zeros(predictions.shape[1])
        model_ranking_t = []  # index of the selected model at timestep t
        for i in range(len(Lt)):
            index, x, y = Lt[i]
            new_i = indices[index]
            Lt[i] = (new_i, x ,y) # correct the index of the filtered samples to the original index
            accuracies += (predictions[new_i] == oracle[new_i]) * 1
            
            best_model_indices = np.argwhere(accuracies == np.max(accuracies)).flatten()
            selected_model_index = np.random.choice(best_model_indices)
            model_ranking_t.append(selected_model_index)

        # if we have not selected enough data points, we select any data points from the unimportant ones
        if len(Lt) < budget:
            rest_indices = np.where((predictions == p).all(axis=1))[0]
            x_i = np.random.choice(rest_indices, size=(budget-len(Lt),), replace=False) 
            x = data[x_i]
            y = oracle[x_i]
            model_ranking_t.extend([model_ranking_t[-1]] * (budget-len(Lt))) # fill up the model ranking with the last model (dosent change for the rest of the data points)
            Lt.extend(list(zip(x_i, x, y)))    


        return [model_ranking_t, Lt]

        