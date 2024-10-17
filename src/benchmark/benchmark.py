import os
import time
import numpy as np
from tqdm import tqdm
from benchmark.benchmark_config import ExperimentConfig
from data_provider.data_provider import create_realisations
from model_selection.model_selector import ModelSelector
from benchmark.bechmark_results import ExperimentResult
from models.model_performance_metric import ModelPerformanceMetric
from model_selection.get_strategy import get_model_selection_strategy
from models.get_performance_metric import get_model_performance_metric
from utils.utils import get_available_cpus
from utils.math_utils import calculate_model_ranking
from typing import List, Dict, Any


import random
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _RunData:
    def __init__(self, methods: list[ModelSelector], classes: list[int], budget: int) -> None:
        self.methods = methods
        self.classes = classes
        self.budget = budget
        
        
def append_benchmark(predictions: np.array, oracle: np.array, classes: np.ndarray, config: ExperimentConfig, additional_methods: List[Dict[str, Any]], realisations: np.ndarray):
    """
    Run additional baselines for existing realisation pool.
    """
    methods: list[ModelSelector] = []
    for method_config in additional_methods:
        methods.append(ModelSelector(get_model_selection_strategy(method_config)))

    result: ExperimentResult = ExperimentResult(methods)
    
    realisations_results = np.zeros(config.iterations, dtype=object)
    for i, r in enumerate(realisations):
        # this calculates the model ranking for each realization pool, returns a tuple of model ranking and accuracies
        realisations_results[i] = calculate_model_ranking(predictions[r], oracle[r])  # predictions[r] takes all predictions for that realization pool

    # Start benchmark
    print(config)

    run_config = _RunData(methods, classes, config.budget)
    start_t = time.time()
    if config.parallel:
        import multiprocessing as mp
        processes = get_available_cpus()
        print(f"Running benchmark with {processes//2} cores...")
        with mp.Pool(processes//2) as pool:
            results = tqdm([pool.apply_async(run_realisation, args=(run_config, 
                                                                    realisations[i], 
                                                                    realisations_results[i], 
                                                                    predictions[realisations[i]], 
                                                                    oracle[realisations[i]], 
                                                                    classes, 
                                                                    i)) for i in range(config.iterations)])
            for res in results:
                result.add_iteration_result(res.get())
    else:
        for i in tqdm(range(config.iterations)):
            result.add_iteration_result(run_realisation(run_config, 
                                                        realisations[i], 
                                                        realisations_results[i], 
                                                        predictions[realisations[i]], 
                                                        oracle[realisations[i]], 
                                                        classes,
                                                        i))

    end_t = time.time()

    print(f"Finished benchmark!")
    print(f"Runtime: {end_t - start_t} seconds")
    result.realisations = realisations.tolist()
    result.runtime = end_t - start_t 

    return result


def run_noisy_benchmark(predictions: np.array, oracle: np.array, classes: np.ndarray, config: ExperimentConfig, realisations: np.ndarray):
    methods: list[ModelSelector] = []
    for method_config in config.methods:
        methods.append(ModelSelector(get_model_selection_strategy(method_config)))

    result: ExperimentResult = ExperimentResult(methods)    
    
    realisations_results = np.zeros(config.iterations, dtype=object)
    for i, r in enumerate(realisations):
        # this calculates the model ranking for each realization pool, returns a tuple of model ranking and accuracies
        realisations_results[i] = calculate_model_ranking(predictions[r], oracle[r])  # predictions[r] takes all predictions for that realization pool

    # Start benchmark
    print(config)

    run_config = _RunData(methods, classes, config.budget)
    start_t = time.time()
    if config.parallel:
        import multiprocessing as mp
        processes = get_available_cpus()
        print(f"Running benchmark with {processes//2} cores...")
        with mp.Pool(processes//2) as pool:
            results = tqdm([pool.apply_async(run_realisation, args=(run_config, 
                                                                    realisations[i], 
                                                                    realisations_results[i], 
                                                                    predictions[realisations[i]], 
                                                                    oracle[realisations[i]], 
                                                                    classes, 
                                                                    i)) for i in range(config.iterations)])
            for res in results:
                result.add_iteration_result(res.get())
    else:
        for i in tqdm(range(config.iterations)):
            result.add_iteration_result(run_realisation(run_config, 
                                                        realisations[i], 
                                                        realisations_results[i], 
                                                        predictions[realisations[i]], 
                                                        oracle[realisations[i]], 
                                                        classes,
                                                        i))

    end_t = time.time()

    print(f"Finished benchmark!")
    print(f"Runtime: {end_t - start_t} seconds")
    result.realisations = realisations.tolist()
    result.runtime = end_t - start_t 

    return result


def run_benchmark(predictions: np.array, oracle: np.array, classes: np.ndarray, config: ExperimentConfig):
    # Setup metric
    metric: ModelPerformanceMetric = get_model_performance_metric(config.model_performance_metric)
    
    # Setup methods
    methods: list[ModelSelector] = []
    for method_config in config.methods:
        methods.append(ModelSelector(get_model_selection_strategy(method_config)))

    result: ExperimentResult = ExperimentResult(methods)    

    # Create realisations
    realisations = create_realisations(oracle, config.iterations, config.parition_method, config.num_samples, config.unique_indices, fit_oracle_size=config.fit_oracle_size)
    
    # random winner
    # if 'cifar10_4070' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/cifar10_4070_poolsize1000_numreals100_Date-2024-09-15_Time-02-53-24/experiment_results.npz')['pool_instances_log'].T
    # elif 'cifar10_5592' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/cifar10_5592_poolsize1000_numreals100_Date-2024-09-15_Time-17-18-22/experiment_results.npz')['pool_instances_log'].T
    # elif 'emotion_detection' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/emotion_detection_poolsize1000_numreals100_Date-2024-09-15_Time-17-18-22/experiment_results.npz')['pool_instances_log'].T
    # elif 'domain_drift' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/domain_drift_poolsize750_numreals100_Date-2024-09-15_Time-17-18-23/experiment_results.npz')['pool_instances_log'].T
    # else:
    #     print("PROBLEM NO DATASET")
        
    # correct winner        
    # if 'cifar10_4070' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/cifar10_4070_poolsize1000_numreals100_Date-2024-09-15_Time-02-53-24/experiment_results.npz')['pool_instances_log'].T
    # elif 'cifar10_5592' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/cifar10_5592_poolsize1000_numreals100_Date-2024-09-15_Time-20-22-01/experiment_results.npz')['pool_instances_log'].T
    # elif 'emotion_detection' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/emotion_detection_poolsize1000_numreals100_Date-2024-09-15_Time-20-22-21/experiment_results.npz')['pool_instances_log'].T
    # elif 'domain_drift' in config.dataset_path:
    #     realisations = np.load('/cluster/home/pokanovic/pool-based-model-picker/resources/results/pool/domain_drift_poolsize750_numreals100_Date-2024-09-15_Time-20-22-21/experiment_results.npz')['pool_instances_log'].T
    # else:
    #     print("PROBLEM NO DATASET")
    
    
    realisations_results = np.zeros(config.iterations, dtype=object)
    for i, r in enumerate(realisations):
        # this calculates the model ranking for each realization pool, returns a tuple of model ranking and accuracies
        realisations_results[i] = calculate_model_ranking(predictions[r], oracle[r])  # predictions[r] takes all predictions for that realization pool

    # Start benchmark
    print(config)

    run_config = _RunData(methods, classes, config.budget)
    start_t = time.time()
    if config.parallel:
        import multiprocessing as mp
        processes = get_available_cpus()
        print(f"Running benchmark with {processes//2} cores...")
        with mp.Pool(processes//2) as pool:
            results = tqdm([pool.apply_async(run_realisation, args=(run_config, 
                                                                    realisations[i], 
                                                                    realisations_results[i], 
                                                                    predictions[realisations[i]], 
                                                                    oracle[realisations[i]], 
                                                                    classes, 
                                                                    i)) for i in range(config.iterations)])
            for res in results:
                result.add_iteration_result(res.get())
    else:
        for i in tqdm(range(config.iterations)):
            result.add_iteration_result(run_realisation(run_config, 
                                                        realisations[i], 
                                                        realisations_results[i], 
                                                        predictions[realisations[i]], 
                                                        oracle[realisations[i]], 
                                                        classes,
                                                        i))

    end_t = time.time()

    print(f"Finished benchmark!")
    print(f"Runtime: {end_t - start_t} seconds")
    result.realisations = realisations.tolist()
    result.runtime = end_t - start_t 

    return result

def run_realisation(config, realisation, results_r, predictions_r, oracle_r, classes, seed) -> ExperimentResult:
    """This will return a dictionary with method names (ModelPicker, Random,....) as keys + "gt"
    results_r_m["gt"][0] list of model rankings on the entire relizatiopn pool, e.g. ground_truth
    results_r_m["gt"][1] list of accuracies of each model on the entire realization pool
    
    results["Random"][0] will be index of the model that is best at timestep t with the selected budget for that method
    results["Random"][1] will be B (budget) elements (index, x, y_oracle)
    """
    # set methods for new realisation
    set_seed(seed)
    results_r_m = {}
    results_r_m["gt"] = results_r
    # results_r_m["realisations"] = realisation
    for method in config.methods:
        results_r_m[method.name] = method.run(realisation, predictions_r, oracle_r, classes, config.budget)
    return results_r_m
