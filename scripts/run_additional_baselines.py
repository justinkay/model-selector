import os
import sys

path = os.path.join(os.path.dirname(__file__),"../src")
sys.path.append(path)

from benchmark.benchmark import run_benchmark, append_benchmark
from benchmark.benchmark_config import ExperimentConfig
from benchmark.bechmark_results import ExperimentResult
from utils.result_plotting import plot_benchmark_summary
from utils.fs_utils import load_dataset, save_additional_results
import argparse
import random
import numpy as np
import torch
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def merge_results(res, new_res):
    for new_method in new_res.iteration_results.keys():
        res.iteration_results[new_method] = new_res.iteration_results[new_method]
    res.runtime += new_res.runtime


CONFIG_PATH = "resources/experiment_configs/method_comparison/emotion_detection.json"

# Parse arguments
parser = argparse.ArgumentParser(description='Run benchmark for model selection methods.')
parser.add_argument("-c", '--config', type=str, help="Path to benchmark config file.")  # later add possiblity for new config file
parser.add_argument("-o", "--output", type=str, default="results", help="Path to save results to. Default: \"results\"")  # later add possibility to change location
parser.add_argument("--folder", type=str, default="results", help="Path to previously saved results.")
args = parser.parse_args()

set_seed(0)

# Load config
# p = CONFIG_PATH if args.config is None else args.config
p = os.path.join(args.folder, "config.json")
config = ExperimentConfig.from_json_file(p)

# Setup benchmark
oracle, predictions, classes = load_dataset(config.dataset_path)

# Load saved results
with open(os.path.join(args.folder, 'experiment_result.json'), 'r') as f:
    data = json.load(f)

res = ExperimentResult.from_json(data)
  
realisations = np.array(res.realisations)
additional_methods = [
    {"name": "AMC", "type": "amc"},
    {"name": "VMA", "type": "vma"}
]
# Run benchmark
new_res = append_benchmark(predictions, oracle, classes, config, additional_methods, realisations)

# Merge the two result objects
merge_results(res, new_res)

summary = res.get_benchmark_summary()

# Plot results
fig_var = plot_benchmark_summary(summary, title=config.name)

# Save results
config.methods.extend(additional_methods)
save_additional_results(config.name, 
             config=config, 
             experiment_result=res, 
             plots=[fig_var], 
             plot_names=["graph_additional.png"], 
             path=args.folder)