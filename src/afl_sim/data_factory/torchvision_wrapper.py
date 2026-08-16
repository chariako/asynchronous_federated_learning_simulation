from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from loguru import logger
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import v2

from afl_sim.enums import DatasetType

from .data_types import DatasetWrapperABC, IndexArray

type TorchvisionDataset = MNIST | FashionMNIST | CIFAR100 | CIFAR10

if TYPE_CHECKING:
    from collections.abc import Sequence


class TorchvisionDatasetWrapper(DatasetWrapperABC):
    """
    A wrapper class for provisioning and handling Torchvision datasets.

    Attributes:
        dataset_type (DatasetType): The enumeration specifying which dataset to load.
        data_root (Path): The root directory where dataset files are downloaded and stored.
        _train_transform_list (list[v2.Transform]): The underlying list of transformations for training.
        _eval_transform_list (list[v2.Transform]): The underlying list of transformations for evaluation.
        _train_data (TorchvisionDataset): The loaded Torchvision training dataset instance.
        _test_data (TorchvisionDataset): The loaded Torchvision evaluation dataset instance.
    """

    def __init__(self, dataset_type: DatasetType, data_root: Path) -> None:
        """
        Initializes the wrapper, builds transforms, and loads the data into memory.

        Args:
            dataset_type (DatasetType): The target dataset to load (e.g., MNIST, CIFAR10).
            data_root (Path): The directory path for storing and loading dataset files.
        """
        self.dataset_type = dataset_type
        self.data_root = data_root

        self._train_transform_list = self._build_train_transform_list()
        self._eval_transform_list = self._build_eval_transform_list()

        self._load_data()

    def _get_dataset_class(self) -> TorchvisionDataset:
        """Matches the requested dataset type to a Torchvision dataset.

        Returns:
            TorchvisionDataset: The corresponding Torchvision dataset class.
        """
        match self.dataset_type:
            case DatasetType.MNIST:
                return MNIST
            case DatasetType.FASHION_MNIST:
                return FashionMNIST
            case DatasetType.CIFAR10:
                return CIFAR10
            case DatasetType.CIFAR100:  # pragma: no branch
                return CIFAR100

    def _load_data(self) -> None:
        """
        Downloads and instantiates the underlying Torchvision datasets.

        Applies a base transform to convert images to standard tensor representations
        before caching them in memory.
        """

        base_transform = v2.ToImage()
        dataset_class = self._get_dataset_class()

        logger.info(f"Fetching dataset: {self.dataset_type}")

        self._train_data = dataset_class(
            root=self.data_root, train=True, download=True, transform=base_transform
        )
        self._test_data = dataset_class(
            root=self.data_root, train=False, download=True, transform=base_transform
        )

    @property
    def train_size(self) -> int:
        """int: Number of samples in the Torchvision training dataset."""
        return len(self._train_data)

    @property
    def train_dataset(self) -> TorchvisionDataset:
        """TorchvisionDataset: Retrieves the Torchvision training dataset."""
        return self._train_data

    @property
    def test_dataset(self) -> TorchvisionDataset:
        """TorchvisionDataset: Retrieves the Torchvision evaluation dataset."""
        return self._test_data

    @property
    def targets(self) -> Any:
        """Any: Retrieves the target labels from the training dataset."""
        return self._train_data.targets

    @property
    def train_transform(self) -> v2.Compose:
        """v2.Compose: Retrieves the composition of training transforms."""
        return v2.Compose(self._train_transform_list)

    @property
    def eval_transform(self) -> v2.Compose:
        """v2.Compose: Retrieves the composition of evaluation transforms."""
        return v2.Compose(self._eval_transform_list)

    def _build_train_transform_list(self) -> list[v2.Transform]:
        """
        Constructs the data augmentation and normalization pipeline for training.

        Returns:
            list[v2.Transform]: A list of initialized Torchvision v2 transforms.
        """
        transform_list = []
        if self.dataset_type.apply_horizontal_flip_transform:
            transform_list.append(v2.RandomHorizontalFlip())
        if self.dataset_type.apply_crop_transform:
            transform_list.append(
                v2.RandomCrop(
                    self.dataset_type.image_size,
                    padding=int(self.dataset_type.image_size / 8),
                )
            )

        transform_list.extend(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.dataset_type.mean, self.dataset_type.std),
            ]
        )
        return transform_list

    def _build_eval_transform_list(self) -> list[v2.Transform]:
        """
        Constructs the standardization and normalization pipeline for evaluation.

        Returns:
            list[v2.Transform]: A list of initialized Torchvision v2 transforms.
        """
        return [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(self.dataset_type.mean, self.dataset_type.std),
        ]

    def get_subset(self, indices: IndexArray) -> Subset[Any]:
        """
        Creates a PyTorch Subset of the training data mapped to specific indices.

        Args:
            indices (IndexArray): A 1D array of integer indices.

        Returns:
            Subset[Any]: The subset of the training dataset.
        """
        return Subset(self._train_data, cast("Sequence[int]", indices))
