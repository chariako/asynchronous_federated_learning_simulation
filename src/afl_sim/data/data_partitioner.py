import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from afl_sim.enums import DatasetType
from afl_sim.paths import PartitionPathCollection
from afl_sim.utils import compute_hash_from_dict, save_partition_plot

_MAX_RETRIES = 5000

type DataSplit = list[NDArray[np.int64]]


def _create_partition_dict(
    num_clients: int,
    dataset: DatasetType,
    alpha: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """
    Generates a dictionary of parameters that uniquely identifies a data split configuration.

    This dictionary is typically used to compute a deterministic hash, ensuring that
    identical configuration parameters always map to the same cached data split.

    Args:
        num_clients (int): The total number of clients participating in the federated pool.
        dataset (DatasetType): The dataset enumeration indicating the target data.
        alpha (float): The concentration parameter for the Dirichlet distribution.
        batch_size (int): The minimum local batch size required for each client.
        seed (int): The random seed used for reproducibility.

    Returns:
        dict[str, Any]: A dictionary containing the partition configuration.
    """
    return {
        "num_clients": num_clients,
        "dataset": dataset,
        "alpha": alpha,
        "batch_size": batch_size,
        "seed": seed,
    }


def id_to_client_indices(partition: DataSplit, client_id: int) -> NDArray[np.int64]:
    """
    Extracts the dataset indices assigned to a specific client from the global partition.

    Args:
        partition (DataSplit): The complete data split containing index arrays for all clients.
        client_id (int): The unique identifier of the requested client.

    Returns:
        NDArray[np.int64]: A 1D NumPy array containing the assigned dataset indices.
    """
    return partition[client_id]


def get_partition(
    data_root: Path,
    num_clients: int,
    dataset: DatasetType,
    alpha: float,
    batch_size: int,
    seed: int,
    targets: np.ndarray,
    visualize: bool,
) -> DataSplit:
    """
    Orchestrates the retrieval or generation of a non-IID dataset partition.

    Checks the disk for an existing data split matching the provided parameters (via hashing).
    If a cached split exists, it loads it into memory. If not, it generates a new Dirichlet-based
    split, saves the metadata, arrays, and an optional visualization plot to disk, and then returns it.

    Args:
        data_root (Path): The base directory where data partitions are cached.
        num_clients (int): The total number of participating clients.
        dataset (DatasetType): The target dataset enumeration.
        alpha (float): The concentration parameter for the Dirichlet distribution (lower = more heterogeneous).
        batch_size (int): The minimum number of samples every client must receive.
        seed (int): The seed for the random number generator.
        targets (np.ndarray): The 1D array of class labels for the entire dataset.
        visualize (bool): Flag indicating whether to generate and save a bar chart of the split.

    Returns:
        DataSplit: A list where each element is a 1D NumPy array of dataset indices assigned to a specific client.
    """

    partition_dict = _create_partition_dict(
        num_clients=num_clients,
        dataset=dataset,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )
    split_hash = compute_hash_from_dict(partition_dict)
    partitions_dir = data_root / "partitions" / split_hash
    partitions_dir.mkdir(parents=True, exist_ok=True)

    paths = PartitionPathCollection.from_hash(partitions_dir, split_hash)

    if paths.data_path.exists():
        logger.info(f"Loading existing partition: {paths.data_path.name}")
        return _load_partition(paths.data_path)

    logger.info(f"Generating new partition (Alpha={alpha})...")
    client_indices = _generate_dirichlet_split(
        targets=targets,
        alpha=alpha,
        num_clients=num_clients,
        num_classes=dataset.num_classes,
        seed=seed,
        batch_size=batch_size,
    )

    logger.info(
        f"Saving partition to: {paths.data_path.name} (visualization={visualize})"
    )
    meta_data = {
        "split_hash": split_hash,
        "config_dump": partition_dict,
    }

    _save_split_packet(
        client_indices=client_indices,
        paths=paths,
        num_clients=num_clients,
        num_classes=dataset.num_classes,
        meta_data=meta_data,
        targets=targets,
        visualize=visualize,
    )

    return client_indices


def _save_split_packet(
    client_indices: DataSplit,
    paths: PartitionPathCollection,
    num_clients: int,
    num_classes: int,
    meta_data: dict[str, Any],
    targets: np.ndarray,
    visualize: bool,
) -> None:
    """
    Serializes and saves a newly generated data partition and its associated artifacts to disk.

    Saves the numerical split as a compressed `.npz` file, the configuration state as a `.json`
    metadata file, and optionally renders a `.png` bar chart of the class distributions.

    Args:
        client_indices (DataSplit): The generated list of client index arrays.
        paths (PartitionPathCollection): The collection of resolved file paths for saving artifacts.
        num_clients (int): The total number of clients.
        num_classes (int): The total number of unique classes in the dataset.
        meta_data (dict[str, Any]): The configuration dictionary used to generate the split.
        targets (np.ndarray): The array of class labels for the dataset.
        visualize (bool): Flag indicating whether to execute the visualization saving routine.
    """
    index_dict: dict[str, Any] = {
        f"client_{i}": client_indices[i] for i in range(num_clients)
    }

    # Save the data split
    np.savez_compressed(
        paths.data_path,
        **index_dict,
    )

    # Save metadata
    with paths.meta_path.open("w") as f:
        json.dump(meta_data, f, indent=4)

    # Save visualization
    if visualize:
        try:
            save_partition_plot(
                targets=targets,
                client_indices=client_indices,
                num_clients=num_clients,
                num_classes=num_classes,
                filepath=paths.plot_path,
            )
        except Exception as e:
            logger.warning(f"Skipping data split visualization due to error: {e}")


def _generate_dirichlet_split(
    targets: np.ndarray,
    alpha: float,
    num_clients: int,
    num_classes: int,
    seed: int,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Generates a non-IID data partition using a Dirichlet distribution over target classes.

    The algorithm attempts to distribute indices such that the proportion of classes assigned
    to each client follows a Dirichlet distribution defined by `alpha`. It validates that every
    client receives at least `batch_size` samples, retrying up to a maximum number of attempts
    if the condition is not met.

    Args:
        targets (np.ndarray): The 1D array of class labels for the complete dataset.
        alpha (float): The Dirichlet concentration parameter. Smaller values yield higher non-IID data.
        num_clients (int): The total number of clients to split the data among.
        num_classes (int): The total number of distinct classes in the dataset.
        seed (int): The random seed for the NumPy random number generator.
        batch_size (int): The strict minimum number of samples each client must receive.

    Returns:
        list[np.ndarray]: A list of 1D NumPy arrays, where the i-th array contains the indices
            assigned to client i.

    Raises:
        RuntimeError: If the algorithm fails to satisfy the `batch_size` minimum requirement
            for all clients after the maximum number of retries (`_MAX_RETRIES`).
    """
    min_size = 0
    rng = np.random.default_rng(seed)

    attempt = 0

    # Index look-up table
    sorted_indices = np.argsort(targets)
    class_counts = np.bincount(targets, minlength=num_classes)
    split_points = np.cumsum(class_counts)[:-1]
    indices_per_class = np.split(sorted_indices, split_points)

    while min_size < batch_size:
        attempt += 1
        if attempt > _MAX_RETRIES:
            raise RuntimeError(
                f"Partition failed: Could not satisfy min_batch_size={batch_size} "
                f"after {_MAX_RETRIES} attempts. Try increasing alpha."
            )

        batch_accumulators: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = indices_per_class[k].copy()
            rng.shuffle(idx_k)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / (proportions.sum() + np.finfo(float).eps)

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_indices = np.split(idx_k, split_points)

            for i in range(num_clients):
                if len(split_indices[i]) > 0:
                    batch_accumulators[i].append(split_indices[i])

        current_indices = [
            np.concatenate(batches) if batches else np.array([], dtype=np.int64)
            for batches in batch_accumulators
        ]

        min_size = min(len(idx) for idx in current_indices)
        final_indices = current_indices

    return final_indices


def _load_partition(path: Path) -> list[np.ndarray]:
    """
    Loads a previously serialized data partition from a `.npz` file.

    Args:
        path (Path): The file path to the compressed `.npz` partition file.

    Returns:
        list[np.ndarray]: The reconstructed list of client index arrays.
    """
    with np.load(path) as data:
        return [data[f"client_{i}"] for i in range(len(data.files))]
