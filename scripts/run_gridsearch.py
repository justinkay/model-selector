import os
import sys
import argparse
path = os.path.join(os.path.dirname(__file__),"../src")
sys.path.append(path)

from epsilon_optimization.gridsearch_config import GridSearchConfig
from epsilon_optimization.gridsearch import epsilon_grid_search
from utils.result_plotting import plot_benchmark_summary, plot_optimal_epsilons
from utils.fs_utils import load_dataset, load_gridsearch_config, save_results
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CONFIG = "resources/experiment_configs/gridsearch/emotion_detection.json"

# Parse arguments
parser = argparse.ArgumentParser(description='Run gamma grid search for information model selection method.')
parser.add_argument("-c", '--config', type=str, help="Path to gamma grid search config file.")
parser.add_argument("-o", "--output", type=str, default="results", help="Path to save results to. Default: \"results\"")
parser.add_argument("--silent", action="store_true", help="Flag to show plots. Default: True")
args = parser.parse_args()

set_seed(0)

config_path = args.config if args.config is not None else CONFIG

config: GridSearchConfig = load_gridsearch_config(config_path)

# Setup gamma grid search
oracle, predictions, classes = load_dataset(config.experiment_config.dataset_path)

# Run gamma grid search
experiment_res, gridsearch_res = epsilon_grid_search(predictions, oracle, classes, config)

# # Plot results
summary = experiment_res.get_benchmark_summary()
fig_experiment = plot_benchmark_summary(summary, show_var=False, threshold=config.epsilon_metric_threshold, show=not args.silent)
# fig_epsilons = plot_optimal_epsilons(gridsearch_res, show=not args.silent)

# # Save results
save_results(config.experiment_config.name, config, 
             experiment_result=experiment_res, 
             gridsearch_result=gridsearch_res,
             plots=[fig_experiment],  #, fig_epsilons],
             plot_names=["experiment_summary.png", "optimal_epsilons.png"],
             path=args.output)
