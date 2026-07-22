from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from torch.utils.data import DataLoader, RandomSampler

from afl_sim.config import DataConfig, EvaluationConfig, OptimizationConfig
from afl_sim.enums import DatasetType

from .data_partitioner import get_client_partition, get_data_split
from .data_types import DatasetWrapperABC
from .torchvision_wrapper import TorchvisionDatasetWrapper


class DataManager:
    """
    Manages data provisioning, partitioning, and loader generation for federated learning.

    Attributes:
        eval_config (EvaluationConfig): Configuration for evaluation data handling.
        optim_config (OptimizationConfig): Configuration for training optimization and batch sizing.
        device_type (str): The device type used for pinning memory during dataloader generation.
        dataset_type (DatasetType): The enumeration specifying the target dataset.
        wrapper (DatasetWrapperABC): The underlying dataset wrapper providing access to raw data.
        client_indices (DataSplit): The partitioned dataset indices assigned to each client.
    """

    def __init__(
        self,
        num_clients: int,
        data_config: DataConfig,
        eval_config: EvaluationConfig,
        optim_config: OptimizationConfig,
        data_dir: Path,
        device_type: str,
        visualize: bool,
    ):
        """
        Initializes the DataManager and orchestrates data partitioning.

        Args:
            num_clients (int): Total number of clients in the simulation.
            data_config (DataConfig): Configuration parameters for the dataset and partitioning.
            eval_config (EvaluationConfig): Configuration for evaluation dataloaders.
            optim_config (OptimizationConfig): Configuration for optimization parameters like batch size.
            data_dir (Path): Root directory where dataset files are stored.
            device_type (str): Hardware device type (e.g., "cpu", "cuda").
            visualize (bool): Whether to generate and save a visualization of the data split.
        """
        self.eval_config = eval_config
        self.optim_config = optim_config
        self.device_type = device_type
        self.dataset_type = data_config.dataset

        self.wrapper = self._init_wrapper(
            dataset_type=data_config.dataset, data_dir=data_dir
        )

        self.client_indices = get_data_split(
            data_root=data_dir,
            num_clients=num_clients,
            dataset=data_config.dataset,
            alpha=data_config.dirichlet_alpha,
            batch_size=self.optim_config.batch_size,
            seed=data_config.split_seed,
            targets=self.wrapper.targets,
            visualize=visualize,
        )

    def _init_wrapper(
        self, dataset_type: DatasetType, data_dir: Path
    ) -> DatasetWrapperABC:
        """
        Initializes the dataset wrapper for the target dataset.

        Args:
            dataset_type (DatasetType): The enumeration specifying the target dataset.
            data_dir (Path): Root directory where dataset files are stored.

        Returns:
            DatasetWrapperABC: The PyTorch DataLoader configured for the client's data partition.
        """
        return TorchvisionDatasetWrapper(dataset_type=dataset_type, data_root=data_dir)

    def get_client_dataloader(self, client_id: int) -> DataLoader[Any]:
        """
        Generates a DataLoader for a specific client's local training data.

        Args:
            client_id (int): The unique identifier of the client.

        Returns:
            DataLoader[Any]: The PyTorch DataLoader configured for the client's data partition.
        """
        indices = get_client_partition(self.client_indices, client_id)

        subset = self.wrapper.get_subset(indices)

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
            num_workers=0,
            pin_memory=self.device_type == "cuda",
            persistent_workers=False,
        )

    def get_client_weight(self, client_id: int) -> float:
        """
        Calculates the statistical weight of a client based on its partition size.

        Args:
            client_id (int): The unique identifier of the client.

        Returns:
            float: The client's weight, calculated as local samples divided by global samples.
        """
        num_samples_global = self.wrapper.train_size
        num_samples_local = cast(
            "int", get_client_partition(self.client_indices, client_id).shape[0]
        )
        return num_samples_local / num_samples_global

    def get_evaluation_dataloader(self) -> DataLoader[Any]:
        """
        Generates a DataLoader for the global evaluation dataset.

        Returns:
            DataLoader[Any]: The PyTorch DataLoader configured for the test dataset.
        """
        return DataLoader(
            self.wrapper.test_dataset,
            batch_size=self.eval_config.batch_size,
            shuffle=False,
            num_workers=self.eval_config.num_workers,
            pin_memory=self.device_type == "cuda",
            persistent_workers=self.eval_config.num_workers > 0,
        )

    def get_train_transform(self) -> Callable[..., Any] | None:
        """
        Retrieves the transformation pipeline applied to training data during training.

        Returns:
            Callable[..., Any] | None: The callable transform, or None if no transform is applied.
        """
        return self.wrapper.train_transform

    def get_eval_transform(self) -> Callable[..., Any] | None:
        """
        Retrieves the transformation pipeline applied to evaluation data during evaluation.

        Returns:
            Callable[..., Any] | None: The callable transform, or None if no transform is applied.
        """
        return self.wrapper.eval_transform
