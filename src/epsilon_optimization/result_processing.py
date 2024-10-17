import numpy as np
from benchmark.bechmark_results import ExperimentResult


def process_results(results: ExperimentResult, epsilon_range: list[float]):
    r = results.get_avg_over_iterations()
    gamma_res = [0.5 * r.avg_best_model_accuracy_t[name] + 0.5 * r.avg_best_model_selected_t[name] for name in r.avg_best_model_accuracy_t.keys()]
    best_gamma = epsilon_range[np.argmax(gamma_res)]

    results = {epsilon_range[i]:(r.avg_best_model_accuracy_t[name], r.avg_best_model_selected_t[name], 0.5 * r.avg_best_model_accuracy_t[name] + 0.5 * r.avg_best_model_selected_t[name]) for i, name in enumerate(r.avg_best_model_accuracy_t.keys())}

    res_dict = {}
    res_dict["best_gamma"] = best_gamma
    res_dict["results"] = results

    print(f"Best gamma: {best_gamma} with value {np.max(gamma_res)}")
    return res_dict