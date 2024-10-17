import json
import os
from model_selection.method_config import MethodConfig
from data_provider.interface_data_provider import PartitionMethod


class ExperimentConfig:

    def __init__(self, 
                 dataset_path="",
                 config_path="",
                 name=None,
                 parition_method=PartitionMethod.RANDOM, 
                 num_samples=200, 
                 unique_indices=True, 
                 fit_oracle_size=False,
                 iterations=5,
                 budget=None,
                 parallel=False,
                 methods: list[MethodConfig]=[],
                 model_performance_metric=None,
                 ) -> None:
        self.parition_method = parition_method
        self.num_samples = num_samples
        self.unique_indices = unique_indices
        self.fit_oracle_size = fit_oracle_size
        self.iterations = iterations      
        self.methods = methods
        self.parallel = parallel
        self.budget = budget
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.name = name
        self.model_performance_metric = model_performance_metric
        
    def to_json(self):
        conf_dict = {
            "experiment": {
                "datasetPath": self.dataset_path,
                "configPath": self.config_path,
                "partitionMethod": self.parition_method.name,
                "numSamples": self.num_samples,
                "uniqueIndices": self.unique_indices,
                "fitOracleSize": self.fit_oracle_size,
                "iterations": self.iterations,
                "parallel": self.parallel
            },
            "methods": self.methods
        }

        if self.budget:
            conf_dict["experiment"]["budget"] = self.budget
        if self.name:
            conf_dict["experiment"]["name"] = self.name
        if self.model_performance_metric:
            conf_dict["experiment"]["modelPerformanceMetric"] = self.model_performance_metric

        return conf_dict  

    def from_dict(data: dict):
        return ExperimentConfig._parse_config(data)

    def from_json_file(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file {path} not found!")
        with open(path, "r") as f:
            data = json.load(f)
        return ExperimentConfig._parse_config(data, path)

    def _parse_config(config: dict, path: str=""):
        if "methods" in config:
            methods = config["methods"]
        else:
            methods = []
        config = config["experiment"]
        return ExperimentConfig(
                config["datasetPath"],
                path,
                num_samples=config["numSamples"], 
                iterations=config["iterations"],
                name=config["name"] if "name" in config else None,
                parition_method=PartitionMethod[str(config["partitionMethod"]).upper()], 
                unique_indices=config["uniqueIndices"] if "uniqueIndices" in config else True, 
                fit_oracle_size=config["fitOracleSize"] if "fitOracleSize" in config else False,
                budget=config["budget"] if "budget" in config else None,
                parallel=config["parallel"] if "parallel" in config else False,
                model_performance_metric=config["modelPerformanceMetric"] if "modelPerformanceMetric" in config else "accuracy",
                methods=methods
            )

    def __str__(self) -> str:
        divider = "====================================================================="
        config_str =  (f"{divider}\nEXPERIMENT CONFIG:\n"
                      f"Name: {self.name}\n"
                      f"Dataset Path: {self.dataset_path}\n"
                      f"Config Path: {self.config_path}\n"
                      f"Partition Method: {self.parition_method}\n"
                      f"Iterations: {self.iterations}\n"
                      f"Number of Samples: {self.num_samples}\n"
                      f"Unique Indices: {self.unique_indices}\n"
                      f"Fit Oracle Size: {self.fit_oracle_size}\n"
                      f"Parallel: {self.parallel}")
        if self.budget is not None:
            config_str += f"\nBudget: {self.budget}"
        if self.model_performance_metric is not None:
            config_str += f"\nModel Metric: {self.model_performance_metric}"

        config_str += f"\n{divider}"
        return config_str