import numpy as np
from benchmark.bechmark_results import BenchmarkSummary, ExperimentResult
import json
import os
from datetime import datetime
from benchmark.benchmark_config import ExperimentConfig
from epsilon_optimization.gridsearch_config import GridSearchConfig
from epsilon_optimization.gridsearch_results import GridsearchResult
    
def save_embedding(embedding: np.ndarray, out_path: str):
    dir = os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    import pickle
    with open(out_path, "bw") as f:
        pickle.dump(embedding, f)
        print(f"Saved calculated embedding to {out_path}!")


def load_embedding(path: str, decoding=np.float32):
    if not os.path.isfile(path):
        raise ValueError(f"Input path {path} invalid!")
    
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
        data = data.astype(decoding)

    print(f"Loaded embedding data from {path}.")
    print(f"Data shape: {data.shape}")
    return data


def load_gridsearch_config(filename: str):
    """
    Loads the configuration of a gridsearch from a json file.

    Args:
        filename (str): The name of the file to load the configuration from.

    Returns:
        GridSearchConfig: The configuration of the gridsearch.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Config file {filename} does not exist.")
    
    with open(filename, "r") as f:
        data = json.load(f)
        result = GridSearchConfig.from_json(data)
    return result


def load_experiment_result(filename: str):
    """
    Loads the results of an experiment from a json file.

    Args:
        filename (str): The name of the file to load the results from.

    Returns:
        ExperimentResult: The results of the experiment.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Result file {filename} does not exist.")
    
    with open(filename, "r") as f:
        data = json.load(f)
        result = ExperimentResult.from_json(data)
    return result


def load_experiment_config(filename: str):
    """
    Loads the configuration of an experiment from a json file.

    Args:
        filename (str): The name of the file to load the configuration from.

    Returns:
        ExperimentConfig: The configuration of the experiment.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Config file {filename} does not exist.")
    
    with open(filename, "r") as f:
        data = json.load(f)
        result = ExperimentConfig.from_dict(data)
    return result


def load_benchmark_summary(filename: str):
    """
    Loads the results of a benchmark from a json file.

    Args:
        filename (str): The name of the file to load the results from.

    Returns:
        Tuple: (BenchmarkConfig, BenchmarkSummary)
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Result file {filename} does not exist.")
    
    with open(filename, "r") as f:
        data = json.load(f)
        result = BenchmarkSummary.from_dict(data["result"])
    return result


def check_path(name: str, path: str):
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, name)
    if os.path.exists(path):
        d = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path += f"_{d}"
    os.makedirs(path)
    return path


def save_results(name: str, 
                 config=None, 
                 experiment_result=None, 
                 gridsearch_result=None, 
                 plots=None, 
                 plot_names=None, 
                 path="results"):
    """
    Saves the results of a gridsearch to a json file.

    Args:
        filename (str): The name of the file to save the results to.
        config (GridSearchConfig): The configuration of the gridsearch.
        result (GridsearchResult): The results of the gridsearch.
    """
    path = check_path(name, path)

    if gridsearch_result is not None:
        if isinstance(gridsearch_result, list):
            res = {}
            for r in gridsearch_result:
                res[r.method] = r.to_json()
            gridsearch_result = res
        else:
            gridsearch_result = gridsearch_result.to_json()

        with open(os.path.join(path, "gridsearch_result.json"), "w") as f:
            json.dump(gridsearch_result, f, indent=4)

    if experiment_result is not None:
        with open(os.path.join(path, "experiment_result.json"), "w") as f:
            json.dump(experiment_result.to_json(), f, indent=4)

    if config is not None:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config.to_json(), f, indent=4)

    if isinstance(plots, list):
        for name, fig in zip(plot_names,plots):
            fig.savefig(os.path.join(path, name))
            print("Saved plot to " + os.path.join(path, name))
    else:
        plots.savefig(os.path.join(path, plot_names))
        print("Saved plot to " + os.path.join(path, plot_names))
        
        
def save_additional_results(name: str, 
                 config=None, 
                 experiment_result=None, 
                 gridsearch_result=None, 
                 plots=None, 
                 plot_names=None, 
                 path="results"):
    """
    Saves the results of a gridsearch to a json file.

    Args:
        filename (str): The name of the file to save the results to.
        config (GridSearchConfig): The configuration of the gridsearch.
        result (GridsearchResult): The results of the gridsearch.
    """
    # path = check_path(name, path)  # it is secured that the path exists, do not create new folder

    if gridsearch_result is not None:
        if isinstance(gridsearch_result, list):
            res = {}
            for r in gridsearch_result:
                res[r.method] = r.to_json()
            gridsearch_result = res
        else:
            gridsearch_result = gridsearch_result.to_json()

        with open(os.path.join(path, "gridsearch_result_additional.json"), "w") as f:
            json.dump(gridsearch_result, f, indent=4)

    if experiment_result is not None:
        with open(os.path.join(path, "experiment_result_additional.json"), "w") as f:
            json.dump(experiment_result.to_json(), f, indent=4)

    if config is not None:
        with open(os.path.join(path, "config_additional.json"), "w") as f:
            json.dump(config.to_json(), f, indent=4)

    if isinstance(plots, list):
        for name, fig in zip(plot_names,plots):
            fig.savefig(os.path.join(path, name))
            print("Saved plot to " + os.path.join(path, name))
    else:
        plots.savefig(os.path.join(path, plot_names))
        print("Saved plot to " + os.path.join(path, plot_names))
        
        
        
def save_noisy_results(name: str, 
                 config=None, 
                 experiment_result=None, 
                 gridsearch_result=None, 
                 plots=None, 
                 plot_names=None, 
                 path="results"):
    """
    Saves the results of a gridsearch to a json file.

    Args:
        filename (str): The name of the file to save the results to.
        config (GridSearchConfig): The configuration of the gridsearch.
        result (GridsearchResult): The results of the gridsearch.
    """
    # path = check_path(name, path)  # it is secured that the path exists, do not create new folder

    if gridsearch_result is not None:
        if isinstance(gridsearch_result, list):
            res = {}
            for r in gridsearch_result:
                res[r.method] = r.to_json()
            gridsearch_result = res
        else:
            gridsearch_result = gridsearch_result.to_json()

        with open(os.path.join(path, "gridsearch_result_noisy.json"), "w") as f:
            json.dump(gridsearch_result, f, indent=4)

    if experiment_result is not None:
        with open(os.path.join(path, "experiment_result_noisy.json"), "w") as f:
            json.dump(experiment_result.to_json(), f, indent=4)

    if config is not None:
        with open(os.path.join(path, "config_noisy.json"), "w") as f:
            json.dump(config.to_json(), f, indent=4)

    if isinstance(plots, list):
        for name, fig in zip(plot_names,plots):
            fig.savefig(os.path.join(path, name))
            print("Saved plot to " + os.path.join(path, name))
    else:
        plots.savefig(os.path.join(path, plot_names))
        print("Saved plot to " + os.path.join(path, plot_names))


def load_gridsearch_result(filename: str):
    """
    Loads the results of a gridsearch from a json file.

    Args:
        filename (str): The name of the file to load the results from.

    Returns:
        GridsearchResult: The results of the gridsearch.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Result file {filename} does not exist.")
    
    with open(filename, "r") as f:
        data = json.load(f)
        result = GridsearchResult.from_json(data)
    return result



def save_dataset(dataset_path: str, oracle: np.array, predictions: np.array):
    """
    Saves the dataset to a given path.

    Args:
        datasetPath (str): Path to the dataset folder.
        oracle (np.array): The oracle (labels) of the dataset.
        predictions (np.array): The predictions of the dataset.
    """
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    else:
        raise Exception(f"Output path {dataset_path} already exists!")
    with open(os.path.join(dataset_path, "oracle.npy"), "wb") as f:
        np.save(f, oracle)
    with open(os.path.join(dataset_path, "predictions.npy"), "wb") as f:
        np.save(f, predictions)
    print(f"Saved dataset to {dataset_path}")

def load_npy_file(path: str):
    if not os.path.isfile(path):
        raise Exception(f"Did not find {path}!")
    return np.load(path)

def load_dataset(dataset_path: str):
    """
    Loads dataset from a given path. The dataset must contain a file oracle.npy which contains the labels for the dataset.

    Args:
        datasetPath (str): _description_

    Raises:
        Exception: path to folder does not contain oracle.npy

    Returns:
        Tuple: (Oracle, Predictions, Classes)
    """
    if not os.path.isfile(os.path.join(dataset_path, 'oracle.npy')) or not os.path.isfile(os.path.join(dataset_path, 'predictions.npy')):
        raise Exception(f"Did not find {dataset_path}/oracle.npy or {dataset_path}/predictions.npy!")
    oracle = np.load(os.path.join(dataset_path, 'oracle.npy'))
    predictions = np.load(os.path.join(dataset_path, 'predictions.npy'))
    classes = np.unique(oracle)
    # print(f"Loaded oracle from {dataset_path} with {len(classes)} classes!")
    # print(f"Loaded predictions from {dataset_path}!")
    dtype = predictions.dtype

    if dtype != np.int64 and \
        dtype != np.int32 and \
        dtype != np.int16 and \
        dtype != np.int8:
        print(f"INFO: Predictions are of type {dtype}. Convert to int32.")
        predictions = predictions.astype(np.int32)
        oracle = oracle.astype(np.int32)
        classes = classes.astype(np.int32)
    min_class = np.min(classes)
    if min_class > 0:
        print(f"INFO: Classes start at {min_class} instead of 0. Adjusting dataset.")
        classes -= min_class
        oracle -= min_class
        predictions -= min_class
    return oracle, predictions, classes