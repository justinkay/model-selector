from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
import numpy as np
import numpy.matlib
from utils.math_utils import entropy
import scipy.stats as stats


class ModelPickerConfig(MethodConfig):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.epsilon = config["epsilon"] if "epsilon" in config.keys() else 0.46
        self.class_prior = config["class_priors"] if "class_priors" in config.keys() else None


class ModelPicker(ModelSelectionStrategy):
    strategy_type = "model_picker"

    def __init__(self, method_config: dict) -> None:
        super().__init__(ModelPickerConfig(method_config))
        self.epsilon = self.method_config.epsilon
        self.class_prior = self.method_config.class_prior
        self.gamma = (1-self.epsilon)/self.epsilon

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
    # Set params, vals
        num_instances =  predictions.shape[0]  # data._num_instances # number of unlabeled data points
        num_models = predictions.shape[1]  # data._num_models  # number of models
        num_classes = len(classes)  # data._num_classes
        # budget = data._budget
  

        # Initialization
        prior = np.ones(num_models, dtype = int) / num_models
        posterior = prior
        #
        unlabelled_indices = list(range(num_instances)) # list of unlabeled instances
        labelled_instances = [] # list of labeled instances
        #
        # predictions = data._predictions[pool_instances_real, :]
        # oracle = data._oracle[pool_instances_real]
        Lt = []
        for i in np.arange(budget): # for each budget, find the instance with minimum conditional entropy

            # Calculate entropies
            entropies = self.compute_entropies(predictions[unlabelled_indices, :], posterior, num_models, num_classes, self.gamma)

    #       Determine index with lowest entropy
            min_entropies = np.min(entropies)
            loc_i_stars = np.where(entropies.reshape(len(unlabelled_indices), 1) == min_entropies)[0]
            i_star = int(np.random.choice(np.size(loc_i_stars), 1)) # if multiple instances, output one randomly

            # Update labelled instances
            labelled_instances.append(unlabelled_indices[loc_i_stars[i_star]])

            # Update the posterior
            posterior = self.update_posterior(posterior.reshape(1, num_models), predictions[unlabelled_indices[loc_i_stars[i_star]], :].reshape(1, num_models), oracle[unlabelled_indices[loc_i_stars[i_star]]], self.gamma)

            ind = unlabelled_indices[loc_i_stars[i_star]]
            Lt.append((ind, data[ind], oracle[ind]))
            # Update the unlabelled indices
            del unlabelled_indices[loc_i_stars[i_star]]

        return Lt  # [(index, x, y), ...]
    

    def compute_entropies(self, predictions_unlabeled, posterior, num_models, num_classes, gamma):

        # Set params, vals
        num_unlabeled = np.size(predictions_unlabeled, 0) # size of currently unlabeled instances


        # preprocess
        posteriors_replicated = np.matlib.repmat(posterior, num_unlabeled, 1) # replicate the posterior (Mx1) to a MxN matrix

        entropies = 0

        # For each class, compute the entropy conditioned on the class
        for c in np.arange(num_classes):

            # Compute class posteriors
            agreements_c = (predictions_unlabeled == c) * 1 # find agreements with that label (NxM)
            new_posteriors_c = np.multiply(posteriors_replicated, np.power(gamma, agreements_c)) # update the posterior
            normalization_constant = np.sum(new_posteriors_c, axis=1) # compute normalization  constant Nx1
            normalization_constant_replicated = np.matlib.repmat(normalization_constant.reshape(num_unlabeled, 1), 1, num_models) # replicate to form a MxN matrix
            new_normalized_posteriors_c = np.divide(new_posteriors_c, normalization_constant_replicated) # normalize the posteriors
            # Above is a num_unalebeled x num_models matrix whose rows are normalized

            # Compute the entropies
            conditional_entropies = stats.entropy(np.matrix(new_normalized_posteriors_c).T, base=2) # compute the entropy over the rows: N sized array
            entropies = entropies + conditional_entropies / num_classes # add to the prev class value and normalize at the return


        return entropies

    """
    Update posterior
    """
    def update_posterior(self, posterior, predictions_i, oracle_i, gamma):

        agreements_i = (predictions_i == oracle_i)*1 # find agreements for the queried data instance --> 1xM
        next_posterior = np.multiply(posterior, np.power(gamma, agreements_i)) # update posterior --> 1xM
        next_posterior = next_posterior/np.sum(next_posterior) # renormalize
        next_posterior = np.squeeze(np.asarray(next_posterior))

        return next_posterior