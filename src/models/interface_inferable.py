from abc import ABC, abstractmethod
from uuid import uuid4

class Inferable(ABC):
    """
    Abstract class that defines the interface for a model that can be used by the model selection algorithm. The only thing a model needs to provide is the ability to make a prediction for a given datapoint.
    """
    def __init__(self, name, id=None) -> None:
        """
        Args:
            name (str): The name of the model.
        """
        super().__init__()
        self.name = name
        if id is not None:
            self.id = id
        else:
            self.id = str(uuid4())

    @abstractmethod
    def make_prediction(self, x):
        """
        Makes a prediction for the given datapoint.

        Args:
            x (Datapoint): The datapoint for which the prediction should be made.
        Returns:
            Label: The prediction for the given datapoint.
        """
        pass