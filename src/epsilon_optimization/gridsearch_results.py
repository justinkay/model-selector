import numpy as np
from benchmark.bechmark_results import ExperimentResult
from utils.math_utils import smooth_data


class GridsearchResult:

    def __init__(self, method: str, epsilons, epsilon_range):
        self.method = method
        self.epsilons = epsilons
        self.epsilon_range = epsilon_range

    def to_json(self):
        return {
            "name": self.method,
            "epsilons": self.epsilons,
            "epsilon_range": self.epsilon_range,
        }

    def from_json(data):
        return GridsearchResult(data["name"], data["epsilons"], data["epsilon_range"])
    

def calculate_optimal_epsilon_fastest(result: ExperimentResult, model_pickers: dict, epsilons: list[int], threshold = 0.9):
    success_mean, success_var, model_acc_mean, model_acc_var = result.get_variance_over_iterations()
    for strategy_name in result.iteration_results.keys():
        success_mean[strategy_name] = smooth_data(success_mean[strategy_name], kernel_size=5)

    model_picker_ids = list(model_pickers.keys())
    model_picker_res = {m[0]: [] for m in set(model_pickers.values())} # stores the success probabilities at t = success > threshold of each epsilon variation of each model picker version
    optimal_epsilons = {}

    for strategy_name in result.iteration_results.keys():
        if any(name in strategy_name for name in model_picker_ids):
            real_name: str = model_pickers[strategy_name][0]
            i = np.argmax(success_mean[strategy_name] > threshold) # find first t where success is above threshold
            if i == 0:
                model_picker_res[real_name].append(np.inf) # never above threshold
            else:
                model_picker_res[real_name].append(int(i)) # store the t where the success of the model picker is above threshold

    for m in model_picker_res.keys():
        optimal_epsilons[m] = {
            "optimal_epsilon": epsilons[np.argmin(model_picker_res[m])],
            "fastest_e": model_picker_res[m]
        }

    return optimal_epsilons


def calculate_optimal_epsilon_success_diff(result: ExperimentResult, model_pickers: dict, other_methods: list[str], epsilons: list[int], threshold = 0.9):
    success_mean, success_var, model_acc_mean, model_acc_var = result.get_variance_over_iterations()
    for strategy_name in result.iteration_results.keys():
        success_mean[strategy_name] = smooth_data(success_mean[strategy_name], kernel_size=5)

    model_picker_ids = list(model_pickers.keys())
    model_picker_res = {m[0]: [] for m in set(model_pickers.values())} # stores the success probabilities at t = success > threshold of each epsilon variation of each model picker version
    optimal_epsilons = {}

    for strategy_name in result.iteration_results.keys():
        if any(name in strategy_name for name in model_picker_ids):
            real_name: str = model_pickers[strategy_name][0]
            i = np.argmax(success_mean[strategy_name] >= threshold) # find first t where success is above threshold
            diff = success_mean[strategy_name][i] - np.max([success_mean[n][i] for n in other_methods])
            model_picker_res[real_name].append(diff) # store the difference of the success of the model picker and the other methods at t

    for m in model_picker_res.keys():
        optimal_epsilons[m] = {
            "optimal_epsilon": epsilons[np.argmax(model_picker_res[m])],
            "success_diff_e": model_picker_res[m]
        }

    return optimal_epsilons


def calculate_optimal_epsilon_success_avg(result: ExperimentResult, model_pickers: dict, epsilons: list[int]):
    success_mean, success_var, model_acc_mean, model_acc_var = result.get_variance_over_iterations()
    model_picker_ids = list(model_pickers.keys())
    model_picker_res = {m[0]: [] for m in set(model_pickers.values())} 
    optimal_epsilons = {}
    

    for strategy_name in result.iteration_results.keys():
        if any(name in strategy_name for name in model_picker_ids):
            real_name: str = model_pickers[strategy_name][0]
            model_picker_res[real_name].append(np.mean(success_mean[strategy_name])) # store the average success of the model picker and the other methods at t

    for m in model_picker_res.keys():
        optimal_epsilons[m] = {
            "optimal_epsilon": epsilons[np.argmax(model_picker_res[m])],
            "success_avg_e": model_picker_res[m]
        }
    return optimal_epsilons


def calculate_optimal_epsilon_all(result: ExperimentResult, model_pickers: dict, other_methods: list[str], epsilons: list[int], threshold = 0.9):
    # success_diff = calculate_optimal_epsilon_success_diff(result, model_pickers, other_methods, epsilons, threshold)
    fastest = calculate_optimal_epsilon_fastest(result, model_pickers, epsilons, threshold)
    avg = calculate_optimal_epsilon_success_avg(result, model_pickers, epsilons)
    res = {}
    res = {"fastest": fastest,"success_avg": avg}
    # for m in success_diff.keys():
    #     res[m] = {"success_diff": success_diff[m],"fastest": fastest[m],"success_avg": avg[m]}
    return res

