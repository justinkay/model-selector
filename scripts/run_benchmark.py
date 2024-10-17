import os
import sys

path = os.path.join(os.path.dirname(__file__),"../src")
sys.path.append(path)

from benchmark.benchmark import run_benchmark
from benchmark.benchmark_config import ExperimentConfig
from utils.result_plotting import plot_benchmark_summary
from utils.fs_utils import load_dataset, save_results
import argparse
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CONFIG_PATH = "resources/experiment_configs/method_comparison/emotion_detection.json"

# Parse arguments
parser = argparse.ArgumentParser(description='Run benchmark for model selection methods.')
parser.add_argument("-c", '--config', type=str, help="Path to benchmark config file.")
parser.add_argument("-o", "--output", type=str, default="results", help="Path to save results to. Default: \"results\"")
parser.add_argument("--silent", action="store_true", default=False, help="Flag to hide plots. Default: False")
args = parser.parse_args()

set_seed(0)

# Load config
p = CONFIG_PATH if args.config is None else args.config
config = ExperimentConfig.from_json_file(p)

# Setup benchmark
oracle, predictions, classes = load_dataset(config.dataset_path)
  
# Run benchmark
res = run_benchmark(predictions, oracle, classes, config)
summary = res.get_benchmark_summary()

# Plot results
fig_var = plot_benchmark_summary(summary, show=not args.silent, title=config.name)

# Save results
save_results(config.name, 
             config=config, 
             experiment_result=res, 
             plots=[fig_var], 
             plot_names=["graph.png"], 
             path=args.output)