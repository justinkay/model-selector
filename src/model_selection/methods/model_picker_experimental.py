from model_selection.method_config import MethodConfig
from data_provider.interface_data_provider import DataProvider
from model_selection.model_selection_strategy import ModelSelectionStrategy
import numpy as np
from utils.math_utils import confidence_margin, entropy, least_confidence


class ModelPickerExperimentalConfig(MethodConfig):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.epsilon = config["epsilon"] if "epsilon" in config.keys() else 0.46
        self.class_prior = config["class_prior"] if "class_prior" in config.keys() else None
        self.selection = config["selection"] if "selection" in config.keys() else "argmin"
        self.k = config["k"] if "k" in config.keys() else None
        uncertainty_measure_type = config["uncertainty_measure"].lower() if "uncertainty_measure" in config.keys() else "entropy" # entropy, confidence_margin, least_confidence 
        if uncertainty_measure_type == "entropy":
            self.uncertainty_measure = entropy
        elif uncertainty_measure_type == "confidence_margin":
            self.uncertainty_measure = confidence_margin
        elif uncertainty_measure_type == "least_confidence":
            self.uncertainty_measure = least_confidence
        else:
            raise ValueError("Invalid uncertainty measure type. Must be one of: entropy, confidence_margin, least_confidence")


class ModelPickerExperimental(ModelSelectionStrategy):
    strategy_type = "model_picker_experimental"

    def __init__(self, method_config: dict, data_provider: DataProvider) -> None:
        super().__init__(ModelPickerExperimentalConfig(method_config), data_provider=data_provider)
        self.epsilon = self.method_config.epsilon
        self.uncertainty_measure = self.method_config.uncertainty_measure
        self.k = self.method_config.k
        self.selection = self.method_config.selection
        self.class_prior = self.method_config.class_prior if self.method_config.class_prior is not None else np.ones(len(data_provider.get_classes())) / len(data_provider.get_classes())

    def __call__(self, data: np.ndarray, predictions: np.ndarray, budget: int):
        Ut = data.tolist()
        Lt = []
        num_models = predictions.shape[1]

        indices = list(range(data.shape[0]))
        classes = self.data_provider.get_classes()
        class_prior = self.class_prior
        posterior = np.ones(num_models) / num_models # initialise posterior

        gamma = (1-self.epsilon)/self.epsilon

        for _ in range(budget):
            i, x, y, posterior = step(posterior, Ut, predictions, classes, gamma, self.data_provider, class_prior, self.uncertainty_measure, k=self.k, selection=self.selection)
            
            predictions = np.delete(predictions, i, axis=0)
            Ut.pop(i)
            Lt.append((indices.pop(i), x, y))

        return Lt
    

def step(belief, ut, predictions, classes, gamma, data_provider, class_prior: list=None, uncertainty_measure=entropy, k=None, selection="argmin"):
    entropies = calculate_rest_uncertainties(belief, predictions, classes, gamma, class_prior, uncertainty_measure, k=k)
    
    if selection == "argmin":
        i = np.argmin(entropies)
    elif selection == "argmax":
        b = entropy(belief)
        i = np.argmin(entropies)
        if b - entropies[i] <= 0:
            i = np.argmax(entropies)
    elif selection == "info_gain":
        b = entropy(belief)
        i = np.argmax(abs(b - entropies))
    else:
        raise ValueError("Invalid selection type. Must be one of: argmin, argmax, info_gain")
    x = ut[i] 
    y = data_provider.query_label(x)

    posterior = update_belief(belief, predictions[i], gamma, y)

    return (i, x, y, posterior)

def calculate_rest_uncertainties(belief, predictions, classes, gamma, class_prior: list=None, uncertainty_measure=entropy, k=None):
    num_classes = len(classes)
    num_instances = predictions.shape[0]
    if class_prior is None:
        class_prior = np.ones(num_classes) / num_classes

    entropies = 0
    for c in classes:
        p_c = (predictions == c) * 1 # calculate number of correct predictions for each model = NxM
        post = np.multiply(belief, np.power(gamma, p_c)) # calculate posterior for each model = NxM
        norm_sum = np.sum(post, axis=1).reshape(num_instances, 1) # sum of posteriors for each datapoint = Nx1
        post_norm = np.divide(post, norm_sum) # normalise posterior = NxM
        if k is not None:
            post_norm = np.partition(post_norm, -k, axis=1)[:,-k:]
            post_norm = post_norm / np.sum(post_norm, axis=1).reshape(num_instances, 1)
        cond_entropy = uncertainty_measure(post_norm, axis=1) # calculate conditional entropy for each datapoint = Nx1
        entropies += cond_entropy * class_prior[c] # weighted sum of conditional entropies = Nx1

    return entropies
    
def update_belief(posterior, predictions_i, gamma, label):
    correct = (predictions_i == label) * 1 # predictions on selected datapoint correct 
    posterior = np.multiply(posterior, np.power(gamma, correct))
    posterior = posterior / np.sum(posterior)
    return np.squeeze(posterior)
