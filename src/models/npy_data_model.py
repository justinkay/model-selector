from models.interface_inferable import Inferable
import numpy as np


class NpyDataModel(Inferable):
    
    def __init__(self, name, predictions, id=None, accuracy=None) -> None:
        super().__init__(name, id=id)
        self.predictions = predictions
        self.accuracy = accuracy

    def make_prediction(self, x):
        if x < 0 or x >= len(self.predictions):
            raise Exception(f"Model {self.name}: Requested prediction {x} out of bounds")
        return self.predictions[x]


def setup_models(predictions: np.ndarray, accuracies= None):
    models = []
    if accuracies is not None and len(accuracies) == predictions.shape[1]:
        for i, acc in enumerate(accuracies):
            models.append(NpyDataModel(f"NpyModel_{acc}", predictions[:,i], id=i, accuracy=acc))
    else:
        for i in range(predictions.shape[1]):
            models.append(NpyDataModel(f"NpyModel_{i}", predictions[:,i], id=i))
    return models
    