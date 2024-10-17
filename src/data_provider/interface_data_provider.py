from abc import ABC, abstractmethod
from enum import Enum

class PartitionMethod(Enum):
    RANDOM = 0 
    FULL_ORDERED = 1

class DataProvider(ABC):
    """ 
    Abstract class that defines the interface for a data provider that can be used by the data selection algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        self.oracle = None
        self.classes = None

    def get_classes(self):
        """
        Returns the classes of the dataset.

        Returns:
            list: The list of classes.
        """
        return self.classes

    @abstractmethod
    def query_label(self, x):
        """ 
        Returns the label for the given index.

        Args:
            x (depends on implementation (datapoint)): The index or datapoint for which the label should be returned.
        Returns:
            Label: The label for the requested datapoint.
        """
        pass

    @abstractmethod
    def size(self):
        """
        Returns the size of the dataset.

        Returns:
            Int: The size of the dataset.
        """
        pass

    @abstractmethod
    def create_realisation(self):
        """
        Creates a new realisation of the dataset. The realisation is a list of indices or datapoints are used for one evaluation round.

        Returns:
            list: A list of datapoints.
        """
        pass

    @abstractmethod
    def create_realisations(self, n, partition_method=PartitionMethod.RANDOM, num_samples=200, unique_indices=True, fit_oracle_size=True):
        """
        Creates a new realisation of the dataset. The realisation is a list of indices or datapoints are used for one evaluation round.

        Args:
            n (int): The number of realisations to create.
            partitionMethod (PartitionMethod, optional): The method used to create the realisations. Defaults to PartitionMethod.RANDOM.
            numSamples (float, optional): The number of samples in each realisation. Defaults to 200.
            uniqueIndices (bool, optional): Whether the indices should be unique. Defaults to True.

        Returns:
            list: A list of lists of datapoints.
        """
        pass
        

    @abstractmethod
    def get_test_set(self):
        """
        Returns the datapoints of the whole test set.

        Returns:
            list: The datapoints of the test set.
        """
        pass
