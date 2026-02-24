import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from afl_sim.enums import DatasetType
from afl_sim.types import PathCollection
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
    Generates a dictionary of data split parameters.
    """
    return {
        "num_clients": num_clients,
        "dataset": dataset,
        "alpha": alpha,
        "batch_size": batch_size,
        "seed": seed,
    }


def id_to_client_indices(partition: DataSplit, client_id: int) -> NDArray[np.int64]:
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
    """Orchestrates loading existing splits or generating new ones."""
    partitions_dir = data_root / "partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)

    partition_dict = _create_partition_dict(
        num_clients=num_clients,
        dataset=dataset,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )
    split_hash = compute_hash_from_dict(partition_dict)
    paths = PathCollection.from_hash(partitions_dir, split_hash)

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
    paths: PathCollection,
    num_clients: int,
    num_classes: int,
    meta_data: dict[str, Any],
    targets: np.ndarray,
    visualize: bool,
) -> None:
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
    Create non-iid data split using Dirichlet distribution.
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
    """Loads partition from file."""
    with np.load(path) as data:
        return [data[f"client_{i}"] for i in range(len(data.files))]
