import numpy as np

def smooth_data(x, kernel_size=5):
    """
    Smooths the data using a moving average kernel.

    Args:
        x (np.ndarray): The data to smooth.
        kernel_size (int, optional): The size of the kernel. Defaults to 5.

    Returns:
        np.ndarray: The smoothed data.
    """
    kernel = np.ones(kernel_size) / kernel_size
    x_padded = np.pad(x, (len(kernel)//2, len(kernel)//2), 'constant', constant_values=(x[0], x[-1]))
    return np.convolve(x_padded, kernel, 'valid')


def calculate_model_ranking(predictions, oracle):
    """
    Returns the index of the models from best to worst.

    Args:
        predictions (np.ndarray): The predictions (size: NxM).
        oracle (np.ndarray): The oracle (size: N).

    Returns:
        np.ndarray: The ranking of the model in indices (size: M).
    """
    if len(oracle.shape) == 1:
        oracle = np.expand_dims(oracle, axis=1)
    accuracies = np.sum(predictions == oracle, axis=0) # according the predictions in the realisation, which model got the most right
    return np.argsort(-accuracies, kind='stable'), accuracies / len(oracle)


def calculate_model_accuracies(predictions, oracle):
    """
    Calculate the absolute model accuracy for the given predictions and oracle.

    Args:
        predictions (np.ndarray): The predictions (size: NxM).
        oracle (np.ndarray): The oracle (size: N).
    
    Returns:
        np.ndarray: The absolute model accuracy (size: M).
    """
    predictions = np.array(predictions)
    oracle = np.array(oracle)
    if len(oracle.shape) == 1:
        oracle = np.expand_dims(oracle, axis=1)
        
    correct = (predictions == oracle) * 1

    return np.mean(correct, axis=0)


def fleiss_kappa_score(predictions, oracle, classes):
    """
    Calculates the agreement rate for a given set of predictions and oracle.

    Args:
        predictions (np.array): The predictions.
        oracle (np.array): The oracle.

    Returns:
        float: The agreement rate.
    """
    num_classes = len(classes)
    n = predictions.shape[1]
    N = predictions.shape[0]

    votes = np.zeros((N, num_classes))
    for i,pred in enumerate(predictions):
        v,_ = np.histogram(pred, bins=num_classes)
        votes[i] = v

    # sum total votes per class normalized to number of votes
    p_j = np.sum(votes, axis=0) / (N*n)
    assert np.isclose(np.sum(p_j), 1.0), f"Sum of p_j is not 1.0: {np.sum(p_j)}"

    # sum of squared votes per class normalized to number of votes
    p_e = np.sum(p_j**2)

    # sum of squared votes per class
    n_ij_2 = (votes**2)
    p = (np.sum(np.sum(n_ij_2, axis=1))-N*n) / (N*n*(n-1))

    # Kappa score
    kappa_score = (p - p_e) / (1 - p_e)
    return kappa_score


def softmax(x):
    """
    Calculate the softmax of a given array.

    Args:
        x (np.ndarray): The array to calculate the softmax for.

    Returns:
        np.ndarray: The softmax of the array.
    """
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def entropy(dist, axis=0):
    """
    Calculates the entropy of a distribution. Clips values to 1e-12 to avoid log(0) errors.

    Args:
        dist (list): Distribution to calculate entropy for. Has to sum up to 1.
        axis (int, optional): Axis to calculate entropy on. Defaults to 0.

    Returns:
        float / list: Entropy of the distribution. Can be a list of distribution is multidimensional.
    """
    dist = np.clip(dist, a_min=1e-12, a_max=None)
    vals = -np.sum(np.multiply(dist, np.log2(dist)), axis=axis) / np.log2(dist.shape[axis])
    return vals


def confidence_margin(dist):
    """
    Calculates the confidence margin of a np.ndarray. Confidence margin is always calculated along the last axis.

    Args:
        dist (list): Distribution to calculate confidence margin for.

    Returns:
        float: Confidence margin of the distribution.
    """
    return 1 - (np.max(dist, axis=-1) - np.partition(dist, -2, axis=-1)[...,-2])
    


def least_confidence(dist, axis=0):
    """
    Calculates the least confidence of a distribution.

    Args:
        dist (list): Distribution to calculate least confidence for.

    Returns:
        float: Least confidence of the distribution.
    """
    return 1 - np.max(dist, axis=axis)


def kl_divergence(p, q):
    """
    Calculates the Kullback-Leibler divergence between two distributions.

    Args:
        p (np.ndarray): First distribution.
        q (np.ndarray): Second distribution.

    Returns:
        float: Kullback-Leibler divergence between the two distributions.
    """
    return np.sum(p * np.log2(p / q))


def distribution(data, n):
    """
    Estimates the probability distribution of the predictions.

    Args:
        predictions (np.ndarray): Predictions of the models.
        n (int): Number of entries in the distribution.

    Returns:
        np.ndarray: Probability distribution of the predictions.
    """
    dist = np.bincount(data, minlength=n)
    dist = dist / len(data)
    return dist

def calculate_class_distribution(oracle, classes):
    """
    Calculates the data distribution for a given oracle.

    Args:
        oracle (np.array): The oracle.

    Returns:
        dict: The data distribution.
    """
    return distribution(oracle.flatten(), len(classes))  


def create_noisy_oracle(predictions: np.ndarray, classes: np.ndarray, measure: str="vote", gt_oracle=None, gt_sampling="ordered", enhance_cnt=50):
    """
    Creates a noisy oracle from a list of predictions. 
    The measure can be either "vote", "vote_dist" or "snorkel". 

    Args:
        predictions (list): Predictions of the models.
        classes (list): Classes of the predictions.
        measure (str, optional): Method on how to estimate labels. Defaults to "vote".
        gt_oracle (list, optional): Ground truth oracle, if set the noisy oracle is enhanced. Defaults to None.
        gt_sampling (str, optional): Sampling method for enhancing the noisy oracle (ordered, weighted, random). Defaults to "ordered".
        enhance_cnt (int, optional): Number of enhanced samples. Defaults to 50.
    Returns:
        np.ndarray: Noisy oracle.
    """
    noisy_oracle = np.zeros(len(predictions))
    noisy_oracle_uncertainty = np.zeros(len(predictions))
    for i, x in enumerate(predictions):
        votes = np.bincount(x, minlength=len(classes))
        vote_dist = votes / np.sum(votes)
        noisy_oracle_uncertainty[i] = entropy(vote_dist)
        if measure == "vote":
            noisy_oracle[i] = classes[np.argmax(votes)]
        elif measure == "vote_dist":
            noisy_oracle[i] = np.random.choice(classes, p=vote_dist)
        elif measure == "snorkel":
            vote_dist = votes / np.sum(votes)
            # .... read snorkel paper

    if gt_oracle is not None and gt_sampling is not None:
        p = np.expand_dims(predictions[:,0], axis=1)
        indices = np.where((predictions != p).any(axis=1))[0]
        enhance_cnt = min(enhance_cnt, len(indices))
        if gt_sampling == "ordered":
            sorted_uncertainty = np.argsort(-noisy_oracle_uncertainty)
            noisy_oracle[sorted_uncertainty[:enhance_cnt]] = gt_oracle[sorted_uncertainty[:enhance_cnt]]
        elif gt_sampling == "random":
            random_indices = np.random.choice(indices, enhance_cnt, replace=False)
            noisy_oracle[random_indices] = gt_oracle[random_indices]
        elif gt_sampling == "weighted":
            uncertainty_weights = noisy_oracle_uncertainty[indices] / np.sum(noisy_oracle_uncertainty[indices])
            random_indices = np.random.choice(indices, enhance_cnt, p=uncertainty_weights, replace=False)
            noisy_oracle[random_indices] = gt_oracle[random_indices]
        else:
            raise ValueError(f"Sampling method {gt_sampling} is not supported.")
    
    return noisy_oracle


def calculate_precentile_return_accuracy(result, percentile=0.9, min_x=None, max_x=None):
    """
    Calculates the return accuracy of the model selection methods based on the x percentile worst return accuracies over all realisations. 

    Args:
        result (np.ndarray): The results.

    Returns:
        np.ndarray: The accuracy of the model predictions.
    """
    num_iter = len(result.iteration_results["Random"]["bestModelAccuracyT"])
    model_acc_t = {name: np.array(result.iteration_results[name]["bestModelAccuracyT"]) for name in result.iteration_results.keys()}

    best_x_acc = {}
    best_x_perc = int(percentile * num_iter)

    best_models_acc = np.zeros(num_iter)
    for realisation in range(num_iter): 
        best_models_acc[realisation] = np.max([model_acc_t[m][realisation] for m in model_acc_t.keys()])
    best_models_acc = np.expand_dims(best_models_acc, axis=1)

    for i, m in enumerate(model_acc_t.keys()):        
        model_acc_t[m] = best_models_acc - model_acc_t[m] # Calculate the difference to the best model at time t
        model_acc_t[m] = -np.sort(-model_acc_t[m], axis=0) # Order ascending over each iteration at time t (biggest gap first thats why - -)
        best_x_acc[m] = np.mean(model_acc_t[m][:best_x_perc], axis=0)[min_x:max_x] # Calculate the mean of the x percentile worst return accuracies

    return best_x_acc