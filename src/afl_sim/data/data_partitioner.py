import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from afl_sim.enums import DatasetType
from afl_sim.utils import compute_hash_from_dict, save_partition_plot

_MAX_RETRIES = 1000

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

    data_path = partitions_dir / f"{split_hash}.npz"
    plot_path = partitions_dir / f"{split_hash}.png"
    meta_path = partitions_dir / f"{split_hash}.json"

    if data_path.exists():
        logger.info(f"Loading existing partition: {data_path.name}")
        return _load_partition(data_path)

    logger.info(f"Generating new partition (Alpha={alpha})...")
    client_indices = _generate_dirichlet_split(
        targets, alpha, num_clients, seed, batch_size
    )

    logger.info(f"Saving partition to: {data_path.name}")
    paths = (data_path, meta_path, plot_path)
    meta_data = {
        "split_hash": split_hash,
        "config_dump": partition_dict,
    }

    _save_split_packet(
        client_indices=client_indices,
        paths=paths,
        num_clients=num_clients,
        meta_data=meta_data,
        targets=targets,
        visualize=visualize,
    )

    return client_indices


def _save_split_packet(
    client_indices: DataSplit,
    paths: tuple[Path, Path, Path],
    num_clients: int,
    meta_data: dict[str, Any],
    targets: np.ndarray,
    visualize: bool,
) -> None:
    data_path, meta_path, plot_path = paths

    index_dict: dict[str, Any] = {
        f"client_{i}": client_indices[i] for i in range(num_clients)
    }

    # Save the data split
    np.savez_compressed(
        data_path,
        **index_dict,
    )

    # Save metadata
    with meta_path.open("w") as f:
        json.dump(meta_data, f, indent=4)

    # Save visualization
    if visualize:
        try:
            logger.info("Saving data split visualization...")
            save_partition_plot(
                targets=targets,
                client_indices=client_indices,
                num_clients=num_clients,
                filepath=plot_path,
            )
        except Exception as e:
            logger.warning(f"Skipping data split visualization due to error: {e}")


def _generate_dirichlet_split(
    targets: np.ndarray, alpha: float, num_clients: int, seed: int, min_batch_size: int
) -> list[np.ndarray]:
    """
    Create non-iid data split using Dirichlet distribution.
    """
    min_size = 0
    num_classes = len(np.unique(targets))

    rng = np.random.default_rng(seed)

    attempt = 0

    while min_size < min_batch_size:
        attempt += 1
        if attempt > _MAX_RETRIES:
            raise RuntimeError(
                f"Partition failed: Could not satisfy min_batch_size={min_batch_size} "
                f"after {_MAX_RETRIES} attempts. Try increasing alpha."
            )

        batch_accumulators: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            rng.shuffle(idx_k)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / (proportions.sum() + 1e-10)

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_indices = np.split(idx_k, split_points)

            for i in range(num_clients):
                if len(split_indices[i]) > 0:
                    batch_accumulators[i].append(split_indices[i])

        current_indices = [
            np.concatenate(batches) if batches else np.array([], dtype=np.int64)
            for batches in batch_accumulators
        ]

        min_size = min(len(idx) for idx in current_indices) if current_indices else 0
        final_indices = current_indices

    return final_indices


def _load_partition(path: Path) -> list[np.ndarray]:
    """Loads partition from file."""
    with np.load(path) as data:
        return [data[f"client_{i}"] for i in range(len(data.files))]
