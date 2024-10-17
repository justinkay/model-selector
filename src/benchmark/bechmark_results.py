import numpy as np
from model_selection.model_selector import ModelSelector
    

class BenchmarkSummary:

    def __init__(self,
                 success_prob_mean,
                 success_prob_var, 
                 selected_acc_mean,
                 selected_acc_var,
                 global_best_accuracy,
                 runtime=None) -> None:
        self.success_prob_mean = success_prob_mean
        self.success_prob_var = success_prob_var
        self.selected_acc_mean = selected_acc_mean
        self.selected_acc_var = selected_acc_var
        self.global_best_accuracy = global_best_accuracy
        self.runtime = runtime
        self.realizations = None

    def to_json(self):
        d = {
            "successProbMean": self.success_prob_mean,
            "successProbVar": self.success_prob_var,
            "selectedAccMean": self.selected_acc_mean,
            "selectedAccVar": self.selected_acc_var,
            "globalBestAccuracy": self.global_best_accuracy
        }
        if self.runtime is not None:
            d["runtime"] = self.runtime
        return d
    
    def from_dict(data):
        return BenchmarkSummary(data["successProbMean"],
                                data["successProbVar"], 
                                data["selectedAccMean"],
                                data["selectedAccVar"],
                                data["globalBestAccuracy"],
                                data["runtime"] if "runtime" in data.keys() else 0)
    

class ExperimentResult:

    """
    Class that represents the performance of a model selector method. It stores the model ranking, the best model selected, the accuracy of the best model and the number of correct selections over time.
    """
    def __init__(self, methods: list[ModelSelector]) -> None:
        """
        Args:
            method (ModelSelector): Reference to the model selector method.
        """
        self.iteration_results = {strategy.name: {
            "modelRankingT": [],
            "bestModelSelectedT": [],
            "bestModelAccuracyT": [],
            "selectedDatapointT": [],
            
        } for strategy in methods}
        self.global_best_accuracy = []
        self.runtime = 0

    def add_iteration_result(self, result: dict):
        gt = result.pop("gt")
        # best_model = gt[0][0]  # is this correct? is best model always the first one? should it be gt[1].argmax() or gt[0].argmin() ? this has to be an array if there are multiple best models
        
        best_accuracy = np.max(gt[1])
        best_models =  np.argwhere(gt[1] == best_accuracy).tolist()
        # best_model = np.argmax(gt[1])
        
        model_ranking_t = {name: [] for name in result.keys()}
        best_model_selected_t = {name: [] for name in result.keys()}
        best_model_accuracy_t = {name: [] for name in result.keys()}

        for name, (ranking, Lt) in result.items():
            model_ranking_t[name] = ranking
            for t in range(len(ranking)): # number of samples in realisation, at t what is the index of the returned model
                best_model_selected_t[name].append(bool(ranking[t] in best_models)) # compare the best model at t with best model from gt at t
                best_model_accuracy_t[name].append(gt[1][ranking[t]]) # get the accuracy of the model selected at t
            
            self.iteration_results[name]["modelRankingT"].append(list(map(int, ranking)))  # this actually stores the index of the model selected at timestep t, ranking was np.int64
            self.iteration_results[name]["bestModelSelectedT"].append(best_model_selected_t[name])
            self.iteration_results[name]["bestModelAccuracyT"].append(best_model_accuracy_t[name])
            self.iteration_results[name]["selectedDatapointT"].append([int(elem[1]) for elem in Lt])  # Lt (index, x, y_oracle)
        self.global_best_accuracy.append(best_accuracy)


    def to_json(self):
        d = {
            "iterationResults": {name:{
                "bestModelSelectedT": self.iteration_results[name]["bestModelSelectedT"],
                "bestModelAccuracyT": self.iteration_results[name]["bestModelAccuracyT"],
                "selectedDatapointT": self.iteration_results[name]["selectedDatapointT"],
                "modelRankingT": self.iteration_results[name]["modelRankingT"]
            } for name in self.iteration_results.keys()},
            "globalBestAccuracy": self.global_best_accuracy,
            "runtime": self.runtime,
            "realisations": self.realisations
        }
        return d
    
    def from_json(data):
        res = ExperimentResult([])
        res.iteration_results = {name: {
            "bestModelSelectedT": data["iterationResults"][name]["bestModelSelectedT"],
            "bestModelAccuracyT": data["iterationResults"][name]["bestModelAccuracyT"],
            "selectedDatapointT": data["iterationResults"][name]["selectedDatapointT"],
            "modelRankingT": data["iterationResults"][name]["modelRankingT"]
        } for name in data["iterationResults"].keys()}
        res.global_best_accuracy = data["globalBestAccuracy"]
        res.runtime = data["runtime"]
        res.realisations = data["realisations"]
        return res
    

    def get_benchmark_summary(self):
        """
        Calculates the mean and variance of the best model selected and the accuracy of the best model over time.

        Returns:
            BenchmarkSummary: The mean and variance of the best model selected and the accuracy of the best model over time.
        """
        success_mean, success_var, selected_accuracy_mean, selected_accuracy_var = self.get_variance_over_iterations()
        avg_global_best_accuracy = np.mean(self.global_best_accuracy)

        return BenchmarkSummary(success_prob_mean=success_mean,
                                success_prob_var=success_var, 
                                selected_acc_mean=selected_accuracy_mean,
                                selected_acc_var=selected_accuracy_var,
                                global_best_accuracy=avg_global_best_accuracy,
                                runtime=self.runtime)

    def get_variance_over_iterations(self):
        """
        Calculates the variance of the best model selected and the accuracy of the best model over time.

        Returns:
            tuple[dict, dict, dict, dict]: The variance of the best model selected and the accuracy of the best model over time with method name as.
        """
        model_selected_var = {}
        model_accuracy_var = {}
        model_selected_mean = {}
        model_accuracy_mean = {}

        for name in self.iteration_results.keys():
            model_selected_var[name] = np.var(self.iteration_results[name]["bestModelSelectedT"], axis=0).tolist()
            model_accuracy_var[name] = np.var(self.iteration_results[name]["bestModelAccuracyT"], axis=0).tolist()
            model_selected_mean[name] = np.mean(self.iteration_results[name]["bestModelSelectedT"], axis=0).tolist()
            model_accuracy_mean[name] = np.mean(self.iteration_results[name]["bestModelAccuracyT"], axis=0).tolist()
        return model_selected_mean, model_selected_var, model_accuracy_mean, model_accuracy_var
