import numpy as np


class ModelResults():
    """
    Class that represents the performance of a model. It stores the predictions, number of correct predictions and the accuracy over time.
    """
    def __init__(self, models) -> None:
        """
        Args:
            model (Inferable): Reference to the model.
        """
        self.gt = []
        self.correct = {model.id: [] for model in models}
        self.predictions = {model.id: [] for model in models}
        self.accuracies = {model.id: [] for model in models}

    def add_prediction(self, predictions: np.ndarray, y):
        """
        Adds a prediction to the model performance. The prediction is compared to the true label and the accuracy is updated.

        Args:
            prediction (Label): The prediction of the model. 
            y (Label): The true label. 
        """
        self.gt.append(y)
        for modelId, prediction in zip(self.predictions.keys(), predictions):
            self.predictions[modelId].append(prediction)
            if len(self.correct[modelId]) == 0:
                self.correct[modelId].append(int(prediction == y))
            else:
                self.correct[modelId].append(self.correct[modelId][-1] + int(prediction == y)) 
            self.accuracies[modelId].append(self.correct[modelId][-1] / len(self.predictions[modelId]))


    def get_accuracy(self, model_id, t=None):
        """
        Returns the latest accuracy of the model.

        Returns:
            float: The latest accuracy of the model.
        """
        if not model_id in self.accuracies:
            raise Exception(f"Model {model_id} not found in {self.accuracies}")
        if len(self.accuracies[model_id]) == 0:
            return 0
        if t and t < len(self.accuracies[model_id]):
            return self.accuracies[model_id][t]
        return self.accuracies[model_id][-1]
    
    
