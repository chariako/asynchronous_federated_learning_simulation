from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision import datasets, transforms

from afl_sim.config import AppConfig
from afl_sim.enums import DatasetType

from .data_partitioner import get_partition, id_to_client_indices


class LabeledDataset(Dataset[tuple[Any, Any]]):
    targets: Any

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        raise NotImplementedError


TransformFn = Callable[[Any], Any]


@dataclass
class DatasetSpec:
    """Defines how to load and process a specific dataset."""

    dataset_cls: type[Any]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_transforms: list[Any] | None = None


DATASET_REGISTRY = {
    DatasetType.MNIST: DatasetSpec(
        dataset_cls=datasets.MNIST, mean=(0.1307,), std=(0.3081,)
    ),
    DatasetType.FASHION_MNIST: DatasetSpec(
        dataset_cls=datasets.FashionMNIST, mean=(0.2860,), std=(0.3530,)
    ),
    DatasetType.CIFAR10: DatasetSpec(
        dataset_cls=datasets.CIFAR10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        train_transforms=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ],
    ),
    DatasetType.CIFAR100: DatasetSpec(
        dataset_cls=datasets.CIFAR100,
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762),
        train_transforms=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ],
    ),
}


class DataManager:
    def __init__(
        self, config: AppConfig, data_dir: Path, visualize: bool, base_seed: int
    ):
        torch.manual_seed(seed=base_seed)
        self.data_root = data_dir
        self.dataset = config.data.dataset
        self.num_clients = config.simulation.num_clients
        self.alpha = config.data.dirichlet_alpha
        self.split_seed = config.data.split_seed
        self.eval_config = config.evaluation
        self.optim_config = config.optimization

        self.train_dataset: LabeledDataset
        self.test_dataset: LabeledDataset
        self.train_dataset, self.test_dataset = self._load_raw_datasets()

        targets_np = np.array(self.train_dataset.targets, dtype=np.int64)
        self.client_indices = get_partition(
            data_root=self.data_root,
            num_clients=self.num_clients,
            dataset=self.dataset,
            alpha=self.alpha,
            batch_size=self.optim_config.batch_size,
            seed=self.split_seed,
            targets=targets_np,
            visualize=visualize,
        )

    def _get_transforms(self, train: bool = False) -> TransformFn:
        """Define transforms based on dataset type."""
        spec = DATASET_REGISTRY[self.dataset]
        transform_list = []

        if train and spec.train_transforms:
            transform_list.extend(spec.train_transforms)

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(spec.mean, spec.std))

        return cast(TransformFn, transforms.Compose(transform_list))

    def _load_raw_datasets(
        self,
    ) -> tuple[
        LabeledDataset,
        LabeledDataset,
    ]:
        """Downloads and loads the specific dataset based on Enum."""
        if self.dataset not in DATASET_REGISTRY:
            raise ValueError(f"Dataset {self.dataset} is not defined in the Registry.")

        spec = DATASET_REGISTRY[self.dataset]

        train_transform = self._get_transforms(train=True)
        test_transform = self._get_transforms(train=False)

        train_ds = spec.dataset_cls(
            root=self.data_root, train=True, download=True, transform=train_transform
        )
        test_ds = spec.dataset_cls(
            root=self.data_root, train=False, download=True, transform=test_transform
        )

        if not hasattr(train_ds, "targets"):
            raise TypeError(f"Dataset {self.dataset} missing 'targets'.")

        return cast(LabeledDataset, train_ds), cast(LabeledDataset, test_ds)

    def get_client_dataloader(
        self, client_id: int
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a dedicated DataLoader for a specific client."""
        indices = id_to_client_indices(self.client_indices, client_id)

        subset = Subset(self.train_dataset, indices.tolist())

        sampler = RandomSampler(
            subset,
            replacement=True,
            num_samples=self.optim_config.num_local_steps
            * self.optim_config.batch_size,
        )

        return DataLoader(
            subset,
            batch_size=self.optim_config.batch_size,
            sampler=sampler,
            num_workers=self.optim_config.num_workers,
        )

    def get_client_weight(self, client_id: int) -> float:
        "Returns importance weight for a specific client."
        num_samples_global = len(self.train_dataset)
        num_samples_local = len(id_to_client_indices(self.client_indices, client_id))

        return num_samples_local / num_samples_global

    def get_evaluation_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Returns the global test set DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_config.batch_size,
            shuffle=False,
            num_workers=self.eval_config.num_workers,
        )
