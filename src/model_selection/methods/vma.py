from model_selection.method_config import MethodConfig
import numpy as np
from model_selection.model_selection_strategy import ModelSelectionStrategy
import numba



class VMA(ModelSelectionStrategy):
    """
    Active Model Selection: A Variance Minimization Approach: https://openreview.net/pdf?id=vBwfTUDTtz
    """
    strategy_type = "vma"

    def __init__(self, method_config: dict) -> None:
        super().__init__(MethodConfig(method_config))

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int):
        Ltrue = (predictions.T == oracle.flatten()).astype(np.float64)  
        Lte = self.generate_lte(predictions, len(classes))
        p_est = np.apply_along_axis(self.count_elements, 1, predictions, len(classes))  # (num_models, classes) estimated probability of each point belonigng to a class
        
        # data
        Nte = Lte.shape[1]
        Lt = [] # indices of queried points
        idx_u = list(range(Nte)) # indices of unqueried test points

        replace = False
        # initialization
        L, q, hat = [], [], []
        
        # for loop
        q_full = self.q_by_loss_diff(Lte, p_est, idx_u, normalize=False)
        for i in range(budget):
            
            # q
            q_now = q_full[idx_u]
            q_now = q_now / np.sum(q_now)
            
            # sampling from q_now
            k = self.sample_from_p(q_now)  #, seed=Nte*seed+i+1)  # seed set previously
            j = idx_u[k] # j: queried test index
            
            # observe the loss of j
            L.append([LL[j] for LL in Ltrue])
            q.append(q_now[k])
            
            # estimated average test loss
            # hat.append(est_fn(np.array(L), np.array(q), Nte))

            # update queried and unqueried points
            if replace is False:
                idx_u.pop(k)
                Lt.append((j, data[j], oracle[j]))

        return Lt
    
    def generate_lte(self, predictions, num_classes):
        # Get the number of examples and models
        num_examples, num_models = predictions.shape

        # Initialize the Lte array with ones (shape: (num_models, num_examples, num_classes))
        Lte = np.ones((num_models, num_examples, num_classes), dtype=np.float64)

        # Fill Lte: Lte[model][data_index][class] == 0 if predictions[data_index][model] == class, else 1
        for model in range(num_models):
            for data_index in range(num_examples):
                predicted_class = predictions[data_index][model]
                Lte[model][data_index][predicted_class] = 0
        return Lte
    
    def count_elements(self, row, classes):
        counts = np.array([np.sum(row == i) for i in range(classes)])
        return counts / counts.sum()

    
    # @numba.njit("int64(float64[:], int64)")
    def sample_from_p(self, p):
        # np.random.seed(seed)  # set set previously
        while True:
            j = np.random.choice(p.size, 10000)
            t = np.random.rand(j.size)
            k = np.where(t < p[j])[0]
            if k.size == 0:
                continue
            else:
                j = j[k[0]]
                break
        return j
    
    # @numba.njit("float64[:](float64[:, :, :], float64[:, :], int64[:])", parallel=True)
    def q_by_loss_diff_sub(self, loss_clfs, p_est, idx):
        K = len(loss_clfs)
        loss_diff = np.zeros(len(idx), dtype=np.float64)
        for k in numba.prange(len(idx)):
            v = idx[k]
            for i in range(K):
                for j in range(i+1, K):
                    for c in range(p_est.shape[1]):
                        loss_diff[k] = loss_diff[k] + np.abs(loss_clfs[i, v, c] - loss_clfs[j, v, c]) * p_est[v, c]
        return loss_diff

    def q_by_loss_diff(self, loss_clfs, p_est, idx, eps=1e-2, normalize=True):
        q = self.q_by_loss_diff_sub(loss_clfs, p_est, np.array(idx))
        if normalize:
            return (q + eps) / np.sum(q + eps)
        else:
            return q + eps
