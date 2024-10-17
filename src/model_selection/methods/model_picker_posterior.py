from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
import numpy as np
from utils.math_utils import entropy
import scipy.stats as stats


class ModelPickerConfig(MethodConfig):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.epsilon = config["epsilon"] if "epsilon" in config.keys() else 0.46
        self.class_prior = config["class_priors"] if "class_priors" in config.keys() else None
        


class ModelPicker(ModelSelectionStrategy):
    strategy_type = "model_picker_posterior"

    def __init__(self, method_config: dict) -> None:
        super().__init__(ModelPickerConfig(method_config))
        self.epsilon = self.method_config.epsilon
        self.class_prior = self.method_config.class_prior
        self.gamma = (1-self.epsilon)/self.epsilon

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Ut = data.tolist()
        Lt = []
        num_models = predictions.shape[1]
        
        self.freq_count = np.ones(len(classes))

        indices = list(range(data.shape[0]))
        posterior = np.ones(num_models) / num_models # initialise posterior
        if self.class_prior is None:
            self.class_prior = np.ones(len(classes)) / len(classes)

        for _ in range(budget):
            entropies = self.calculate_hypothetical_uncertainties(posterior, predictions, classes)
        
            # i = np.argmin(entropies)
            i_all_indices = np.where(entropies == np.min(entropies))[0]
            i_index = int(np.random.choice(np.size(i_all_indices), 1))  # np.random.choice(i_all_indices)
            i = i_all_indices[i_index]

            x = Ut.pop(i)
            y = oracle[indices[i]]

            posterior = self.update_belief(posterior, predictions[i], y)
            
            predictions = np.delete(predictions, i, axis=0)
            
            Lt.append((indices.pop(i), x, y))
            
            self.freq_count[y] += 1
            self.update_class_prior()

        return Lt
    
    
    def update_class_prior(self):
        self.class_prior = self.class_prior * self.freq_count
        self.class_prior = self.class_prior / np.sum(self.class_prior)
        

    def calculate_hypothetical_uncertainties(self, belief, predictions, classes):
        """
        Calculates the hypothetical uncertainties of the unlabelled data points given a belief and model predictions.

        Args:
            belief (list | np.ndarray): Belief about the best model.
            predictions (np.ndarray): Predictions of the models on all unlabeled datapoints.

        Returns:
            np.ndarray: Hypothetical uncertainties of the unlabelled data points estimated over all classes.
        """
        num_instances = predictions.shape[0]

        entropies = 0
        for c in classes:
            p_c = ((predictions == c) * 1).astype(np.float32) # calculate number of correct predictions for each model = NxM
            post = np.multiply(belief, np.power(self.gamma, p_c), dtype=np.float32) # calculate posterior for each model = NxM
            norm_sum = np.sum(post, axis=1, dtype=np.float32).reshape(num_instances, 1) # sum of posteriors for each datapoint = Nx1
            post_norm = np.divide(post, norm_sum, dtype=np.float32) # normalise posterior = NxM
            cond_entropy = entropy(post_norm, axis=1) # calculate conditional entropy for each datapoint = Nx1
            entropies += cond_entropy * self.class_prior[c] # weighted sum of conditional entropies = Nx1

        return entropies
        
    def update_belief(self, belief, predictions_i, label):
        """
        Updates the belief given the prediction of the selected datapoint and its queried true label.

        Args:
            belief (list | np.ndarray): Belief about the best model.
            predictions_i (np.ndarray): Predictions of the models on the selected datapoint.
            label (int): The true label of the selected datapoint.

        Returns:
            np.ndarray: The updated belief.
        """
        correct = (predictions_i == label) * 1 # predictions on selected datapoint correct 
        belief = np.multiply(belief, np.power(self.gamma, correct))
        belief = belief / np.sum(belief)
        return np.squeeze(belief)