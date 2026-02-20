from typing import TypedDict
from unittest.mock import patch

import numpy as np
import pytest

from afl_sim.data.data_partitioner import (
    _generate_dirichlet_split,
    _save_split_packet,
    get_partition,
)
from afl_sim.enums import DatasetType
from afl_sim.types import PathCollection


class PartitionConfig(TypedDict):
    num_clients: int
    alpha: float
    batch_size: int
    seed: int
    targets: np.ndarray


class ObjectForTesting(TypedDict):
    config: PartitionConfig
    dataset: DatasetType
    client_indices: list[np.ndarray]


@pytest.fixture(params=[DatasetType.MNIST, DatasetType.CIFAR100])
def dataset_type(request) -> DatasetType:
    return request.param  # type: ignore


@pytest.fixture
def valid_config(dataset_type) -> PartitionConfig:
    """Provides a valid configuration dictionary."""
    return PartitionConfig(
        num_clients=5,
        alpha=1000.0,
        seed=42,
        batch_size=1,
        targets=np.repeat(np.arange(dataset_type.num_classes), 100),
    )


@pytest.fixture
def valid_test_object(valid_config: PartitionConfig, dataset_type) -> ObjectForTesting:
    """Builds the full test object with a uniform index partition."""
    total_indices = np.arange(len(valid_config["targets"]))
    client_indices = np.array_split(total_indices, valid_config["num_clients"])

    return ObjectForTesting(
        config=valid_config,
        dataset=dataset_type,
        client_indices=client_indices,
    )


# --- Partition Integrity Tests ----
def test_partition_integrity(valid_config, dataset_type):
    """Test that valid inputs produce valid splits."""
    indices = _generate_dirichlet_split(
        **valid_config, num_classes=dataset_type.num_classes
    )
    all_assigned_indices = np.concatenate(indices)

    assert len(indices) == valid_config["num_clients"], "Non-existing client dataset!"
    assert (
        np.min([len(indices[i]) for i in range(valid_config["num_clients"])])
        >= valid_config["batch_size"]
    ), "Batch size exceeds minimum client dataset size!"
    assert len(all_assigned_indices) == len(np.unique(all_assigned_indices)), (
        "Found duplicate indices assigned to multiple clients!"
    )
    assert np.array_equal(
        np.sort(all_assigned_indices), np.arange(len(valid_config["targets"]))
    ), "Assigned indices do not match target indices!"


def test_partition_retry_logic(monkeypatch, valid_config, dataset_type):
    """Test that invalid inputs raise errors."""
    monkeypatch.setattr("afl_sim.data.data_partitioner._MAX_RETRIES", 2)
    new_alpha = 0.0001
    new_batch_size = len(valid_config["targets"]) // valid_config["num_clients"]
    with pytest.raises(RuntimeError) as excinfo:
        _generate_dirichlet_split(
            **(valid_config | {"alpha": new_alpha, "batch_size": new_batch_size}),
            num_classes=dataset_type.num_classes,
        )

    assert "Partition failed" in str(excinfo.value)
    assert f"min_batch_size={new_batch_size}" in str(excinfo.value)


# Helper function: Calculate coefficient of variation (CV)
def _CV_per_class(indices, num_clients, num_classes, targets) -> np.ndarray:
    all_counts = np.zeros((num_clients, num_classes))
    for i, client_idx in enumerate(indices):
        client_labels = targets[client_idx]
        all_counts[i, :] = np.bincount(client_labels, minlength=num_classes)

    return all_counts.std(axis=0) / all_counts.mean(axis=0)  # type: ignore


def test_dirichlet_distribution_properties(valid_config, dataset_type):
    """Verify that alpha controls label skewness effectively."""
    # Non-uniform case
    indices_skewed = _generate_dirichlet_split(
        **(valid_config | {"alpha": 0.0001, "batch_size": dataset_type.num_classes}),
        num_classes=dataset_type.num_classes,
    )

    CV_per_class_skewed = _CV_per_class(
        indices=indices_skewed,
        num_clients=valid_config["num_clients"],
        num_classes=dataset_type.num_classes,
        targets=valid_config["targets"],
    )

    assert np.all(CV_per_class_skewed > 1.9), (
        f"Client data not skewed enough! Min CV={CV_per_class_skewed.min()}"
    )

    # Uniform case
    indices_uniform = _generate_dirichlet_split(
        **valid_config, num_classes=dataset_type.num_classes
    )
    CV_per_class_uniform = _CV_per_class(
        indices=indices_uniform,
        num_clients=valid_config["num_clients"],
        num_classes=dataset_type.num_classes,
        targets=valid_config["targets"],
    )
    assert np.all(CV_per_class_uniform < 0.07), (
        f"Client data too skewed! Max CV={CV_per_class_uniform.max()}"
    )


def test_partition_reproducibility(valid_config, dataset_type):
    """Ensure that the same seed produces identical partitions."""
    config = valid_config.copy()
    config["seed"] = 12345

    indices_1 = _generate_dirichlet_split(
        **config, num_classes=dataset_type.num_classes
    )
    indices_2 = _generate_dirichlet_split(
        **config, num_classes=dataset_type.num_classes
    )

    config["seed"] = 99999
    indices_3 = _generate_dirichlet_split(
        **config, num_classes=dataset_type.num_classes
    )

    for i in range(len(indices_1)):
        np.testing.assert_array_equal(indices_1[i], indices_2[i])

    assert not np.array_equal(indices_1[0], indices_3[0])


# --- I/O & Visualization Tests ---
def test_partition_packet_is_saved(tmp_path, valid_test_object):
    """Test that a new partition is generated and saved if it doesn't exist."""

    get_partition(
        data_root=tmp_path,
        dataset=valid_test_object["dataset"],
        visualize=False,
        **valid_test_object["config"],
    )

    saved_data_files = list(tmp_path.rglob("*.npz"))
    saved_meta_files = list(tmp_path.rglob("*.json"))

    assert len(saved_data_files) == 1, "The partition data file (.npz) was not saved."
    assert len(saved_meta_files) == 1, (
        "The partition meta-data file (.json) was not saved."
    )


@pytest.mark.parametrize("visualize_flag", [True, False])
@patch("afl_sim.data.data_partitioner.save_partition_plot")
def test_save_split_packet_visualization_trigger(
    mock_save_plot,
    visualize_flag,
    tmp_path,
    valid_test_object,
):
    """Test that visualization is saved only when the flag is True."""
    hash_str = "test_hash"
    paths = PathCollection.from_hash(tmp_path, hash_str)

    _save_split_packet(
        client_indices=valid_test_object["client_indices"],
        paths=paths,
        num_clients=valid_test_object["config"]["num_clients"],
        num_classes=valid_test_object["dataset"].num_classes,
        meta_data={"test": "data"},
        targets=valid_test_object["config"]["targets"],
        visualize=visualize_flag,
    )

    if visualize_flag:
        mock_save_plot.assert_called_once()
    else:
        mock_save_plot.assert_not_called()


@patch("afl_sim.data.data_partitioner._generate_dirichlet_split")
def test_existing_partition_is_loaded(
    mock_generate,
    tmp_path,
    valid_test_object,
):
    """Test that if the partition file exists, we load it instead of generating."""
    mock_generate.return_value = valid_test_object["client_indices"]

    get_partition(
        data_root=tmp_path,
        dataset=valid_test_object["dataset"],
        visualize=False,
        **valid_test_object["config"],
    )

    mock_generate.reset_mock()

    get_partition(
        data_root=tmp_path,
        dataset=valid_test_object["dataset"],
        visualize=False,
        **valid_test_object["config"],
    )

    mock_generate.assert_not_called()
