from model_selection.method_config import MethodConfig
from data_provider.interface_data_provider import DataProvider
import numpy as np

from model_selection.model_selection_strategy import ModelSelectionStrategy
from models.interface_inferable import Inferable
from utils.math_utils import distribution


class AMC(ModelSelectionStrategy):
    strategy_type = "amc"

    def __init__(self, method_config: dict) -> None:
        super().__init__(MethodConfig(method_config))

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Ut = data.copy().tolist()
        Lt = []

        num_models = predictions.shape[1]
        num_classes = len(classes)
        num_instances = len(Ut)
        
        # Paper equation (13): Compute the p(y,x)
        p_yx = np.zeros((num_instances, num_classes), dtype=np.float32) # NxC matrix: class distribution for each instance by models 
        for i in np.arange(num_instances):
            p_yx[i, :] = distribution(predictions[i,:], n=num_classes) # compute the class distribution for each instance (NxC matrix)

        # Paper equation (3): Find risk per model and compute delta matrix
        predictions = predictions.T
        model_risks = np.zeros(num_models).reshape(num_models, 1) # an array of R[fm]'s
        for model_m in np.arange(num_models):
            for label_c in np.arange(num_classes):
                # how many times did model_m predict not label_c
                loss_m = ((predictions[model_m, :] != label_c)*1).reshape(1, num_instances)
                # the probability of label_c based on model opinions for each instance
                p_cx = p_yx[:, label_c].reshape(1, num_instances) / num_instances 
                # adding up the lossses where the model did not predict label_c times the probability of label_c based on model opinions
                model_risks[model_m] = model_risks[model_m] + np.inner(p_cx, loss_m)
        model_risks = model_risks/num_classes
        model_risks_repmat = np.repeat(model_risks, repeats=num_models, axis=1)
        delta = model_risks_repmat - model_risks_repmat.T # compute differences between the model losses (MxM matrix)

        # Paper theorem (1): Compute q* (Optimal sampling distribution)
        possible_classes = np.repeat(np.arange(0, num_classes).reshape([1, num_classes]), repeats=num_instances, axis=0) # NxC matrix: number instances times list of possible labels
        q_star = np.zeros(shape=[num_instances, num_models, num_models], dtype=np.float32) # NxMxM matrix: number instances times number models times number models
        for r in np.arange(num_models):
            for s in np.arange(num_models):
                if r == s:
                    continue
                # compute 0-1 loss for model r and s
                loss_r = np.array(np.expand_dims(predictions[r, :], axis=1) != possible_classes, dtype=np.int32)
                loss_s = np.array(np.expand_dims(predictions[s, :], axis=1) != possible_classes, dtype=np.int32)
                # for each class, square of difference between loss_s, loss_r, and delta_mat
                loss_diff_delta_squared = np.square(loss_r - loss_s - delta[r, s]) # NxC matrix: over all instances for each class compute loss difference squared
                # sqaure root, weight with p_theta, sum over classes
                q_star[:, r, s] = np.sqrt(np.sum(loss_diff_delta_squared * p_yx, axis=1)) # weight the loss difference with the probability of the class
        
        # Paper equation (15): Normalize q* to make a probability distribution
        q_star = np.sum(q_star, axis=(1, 2)) / (num_models * (num_models - 1)) # sum the model difference matrices at each instance, normalize with number of model comparisons
        q_star /= np.sum(q_star) # normalize to make a probability distribution over all instances

        # order the indices by q_star value
        samples = np.random.choice(np.arange(num_instances), size=budget, replace=False, p=q_star) # sample instances based on q_star
        # samples = np.argsort(-q_star)[:budget] # argsort returns indices of highest values

        predictions = predictions.T
        for sample_i in samples:
            x = Ut[sample_i]
            y = oracle[sample_i]
            Lt.append((sample_i ,x, y))
        
        return Lt