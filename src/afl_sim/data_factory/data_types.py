from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset, Subset

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

type IndexArray = NDArray[np.int64]
type DataSplit = list[IndexArray]


class DatasetWrapperABC(ABC):
    """
    Abstract base class defining the contract for dataset wrappers.

    Attributes:
        train_size (int): Number of samples in the training set.
        train_dataset (Dataset[Any]): The raw training dataset instance.
        test_dataset (Dataset[Any]): The raw evaluation dataset instance.
        targets (Any): The sequence of class labels for the training dataset.
        train_transform (Callable[..., Any] | None): The transformation pipeline for training data.
        eval_transform (Callable[..., Any] | None): The transformation pipeline for evaluation data.
    """

    @property
    @abstractmethod
    def train_size(self) -> int:
        """int: Number of samples in the training set."""
        pass

    @property
    @abstractmethod
    def train_dataset(self) -> Dataset[Any]:
        """Dataset[Any]: Retrieves the raw training dataset."""
        pass

    @property
    @abstractmethod
    def test_dataset(self) -> Dataset[Any]:
        """Dataset[Any]: Retrieves the raw evaluation dataset."""
        pass

    @property
    @abstractmethod
    def targets(self) -> Any:
        """Any: Retrieves the class labels for the training dataset."""
        pass

    @property
    @abstractmethod
    def train_transform(self) -> Callable[..., Any] | None:
        """Callable[..., Any] | None: Retrieves the transform applied to training data during training."""
        pass

    @property
    @abstractmethod
    def eval_transform(self) -> Callable[..., Any] | None:
        """Callable[..., Any] | None: Retrieves the transform applied to evaluation data during evaluation."""
        pass

    @abstractmethod
    def get_subset(self, indices: IndexArray) -> Subset[Any]:
        """
        Creates a dataset subset based on the provided indices.

        Args:
            indices (IndexArray): A 1D array of dataset indices.

        Returns:
            Subset[Any]: A PyTorch Subset object mapped to the specified indices.
        """
        pass
