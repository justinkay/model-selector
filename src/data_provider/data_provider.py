import math
from data_provider.interface_data_provider import  PartitionMethod
import numpy as np

def create_realisations(oracle, n, partition_method=PartitionMethod.RANDOM, num_samples=200, unique_indicies=True, fit_oracle_size=True):
    if oracle is None:
        raise Exception("Oracle not initialized")
    # no point in the code below
    # if n * num_samples > oracle.shape[0]:
    #     if fit_oracle_size:
    #         n_new = math.floor(oracle.shape[0] / num_samples)
    #         print(f"WARNING: Reduced number of realisations from {n} to {n_new} to fit the oracle size")
    #         n = n_new
    #     else:
    #         unique_indicies = False
    #         print(f"WARNING: Using non-unique indices due to insufficient oracle size for {n} realisations with {num_samples} samples")
    x = np.array([np.random.permutation(np.arange(oracle.shape[0], dtype=int))[:num_samples] for _ in range(n)])
    # if partition_method == PartitionMethod.RANDOM:
    #     # x = np.random.choice(range(0, oracle.shape[0]),size=(n,num_samples) ,replace=not unique_indicies)  THIS DOES NOT HAVE UNIQUE INDICES INSIDE EACH REALISATION
    #     x = np.array([np.random.choice(range(0, oracle.shape[0]), size=num_samples, replace=False) for _ in range(n)])
    # elif partition_method == PartitionMethod.FULL_ORDERED:
    #     x = np.arange(n*num_samples).reshape((n,num_samples))
    # else:
    #     raise Exception("Unknown partition method")
    print(f"Created {x.shape[0]} realisations with {x.shape[1]} elements each!")
    return x