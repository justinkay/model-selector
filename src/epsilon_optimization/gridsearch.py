import copy
import uuid
import numpy as np
from tqdm import tqdm
from benchmark.bechmark_results import ExperimentResult
from data_provider.data_provider import create_realisations
from epsilon_optimization.gridsearch_config import GridSearchConfig, OptimalEpsilonMetric
from epsilon_optimization.gridsearch_results import GridsearchResult, calculate_optimal_epsilon_all, calculate_optimal_epsilon_fastest, calculate_optimal_epsilon_success_avg, calculate_optimal_epsilon_success_diff
from model_selection.get_strategy import get_model_selection_strategy
from model_selection.methods.model_picker import ModelPicker
from model_selection.methods.model_picker_experimental import ModelPickerExperimental
from model_selection.model_selector import ModelSelector
from benchmark.benchmark import run_realisation, _RunData
from utils.math_utils import calculate_model_ranking, create_noisy_oracle
from utils.utils import get_available_cpus


def epsilon_grid_search(predictions: np.array, oracle: np.array, classes: np.ndarray, config: GridSearchConfig) -> np.ndarray:
    
    # if config.noisy_oracle is not None:
    #     new_oracle = create_noisy_oracle(predictions, classes, config.noisy_oracle)
    #     diff = np.mean((new_oracle == oracle.flatten())*1)
    #     print(f"Noisy oracle: {config.noisy_oracle} with {diff*100:.2f}% similarity")
    #     oracle = new_oracle
    
    # Setup methods
    methods: list[ModelSelector] = []
    for method_config in config.experiment_config.methods:
        methods.append(ModelSelector(get_model_selection_strategy(method_config)))

    # Create realisations
    realisations = create_realisations(oracle, config.experiment_config.iterations, config.experiment_config.parition_method, config.experiment_config.num_samples, config.experiment_config.unique_indices, fit_oracle_size=config.experiment_config.fit_oracle_size)
    realisation_results = np.zeros(config.experiment_config.iterations, dtype=object)
    
    oracle = oracle.flatten()
    oracles = np.zeros((config.experiment_config.iterations, config.experiment_config.num_samples))
    for i, r in enumerate(realisations):
        gt_oracle = oracle[r]
        if config.noisy_oracle is not None:
            oracles[i] = create_noisy_oracle(predictions[r], classes, measure=config.noisy_oracle, gt_oracle=gt_oracle, gt_sampling=config.noisy_oracle_enhancement, enhance_cnt=config.noisy_oracle_enhancement_cnt)
        else:
            oracles[i] = gt_oracle
        realisation_results[i] = calculate_model_ranking(predictions[r], oracles[i])

    gridsearch_methods = []
    model_pickers = {}
    other_methods = []
    for method in methods:
        if method.strategy.strategy_type == "model_picker_experimental" or method.strategy.strategy_type == "model_picker":
            m_id = uuid.uuid4()
            for epsilon in config.epsilons:
                strategy: ModelPicker | ModelPickerExperimental = copy.copy(method.strategy)
                model_pickers[f"{m_id}_{epsilon}"] = (method.strategy.name, epsilon)
                strategy.name = f"{m_id}_{epsilon}"
                strategy.epsilon = epsilon
                strategy.gamma = (1-epsilon)/epsilon
                gridsearch_methods.append(ModelSelector(strategy))
        else:
            gridsearch_methods.append(method)
            other_methods.append(method.strategy.name)

    data = _RunData(gridsearch_methods, classes, config.experiment_config.budget)
    result = ExperimentResult(gridsearch_methods)
    result.realisations = realisations.tolist()

    print(config)
    # run grid search
    if config.experiment_config.parallel:
        import multiprocessing as mp
        processes = get_available_cpus()
        print(f"Running gridsearch with {processes} cores...")
        with mp.Pool(processes) as pool:
            promises = tqdm([pool.apply_async(run_realisation, args=(data, 
                                                                     realisations[i], 
                                                                     realisation_results[i], 
                                                                     predictions[realisations[i]],
                                                                     oracles[i],
                                                                     classes,
                                                                     i)) for i in range(config.experiment_config.iterations)])
            for res in promises:
                result.add_iteration_result(res.get())
    else:
        for i in tqdm(range(config.experiment_config.iterations)):
            result.add_iteration_result(run_realisation(data, 
                                                        realisations[i], 
                                                        realisation_results[i], 
                                                        predictions[realisations[i]],
                                                        oracles[i],
                                                        classes,
                                                        i))

    # process results
    if config.epsilon_metric == OptimalEpsilonMetric.FASTEST:
        optimal_epsilons = calculate_optimal_epsilon_fastest(result, model_pickers, config.epsilons, config.epsilon_metric_threshold)
    elif config.epsilon_metric == OptimalEpsilonMetric.SUCCESS_DIFFERENCE:
        optimal_epsilons = calculate_optimal_epsilon_success_diff(result, model_pickers, other_methods, config.epsilons, config.epsilon_metric_threshold)
    elif config.epsilon_metric == OptimalEpsilonMetric.SUCCESS_AVG:
        optimal_epsilons = calculate_optimal_epsilon_success_avg(result, model_pickers, config.epsilons)
    elif config.epsilon_metric == OptimalEpsilonMetric.ALL:
        optimal_epsilons = calculate_optimal_epsilon_all(result, model_pickers, other_methods, config.epsilons, config.epsilon_metric_threshold)
    else:
        raise ValueError(f"Unknown epsilon metric: {config.epsilon_metric}")
    
    for method in list(result.iteration_results.keys()):
        if method in list(model_pickers.keys()):
            res = result.iteration_results.pop(method)
            original_name, epsilon = model_pickers[method]
            result.iteration_results[f"{original_name}_{epsilon:.3f}"] = res

    results = []
    for method in optimal_epsilons.keys():
        results.append(GridsearchResult(method, optimal_epsilons[method], list(config.epsilons)))

    return result, results
