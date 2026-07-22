from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from afl_sim.config import DataConfig, EvaluationConfig, OptimizationConfig
from afl_sim.data_factory.data_manager import DataManager
from afl_sim.data_factory.data_types import DatasetWrapperABC, DataSplit

MODULEPATH = "afl_sim.data_factory.data_manager"


@dataclass
class ValidTestObject:
    wrapper: DatasetWrapperABC
    split: DataSplit
    num_clients: int
    num_samples: int


@pytest.fixture
def valid_test_object():
    num_clients = 3
    num_features = 2
    num_samples = 2 * num_clients
    num_labels = 3

    class MockDataset(Dataset[Any]):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.rand(size=(num_samples, num_features))
            self.targets = torch.randint(low=0, high=num_labels, size=(num_samples,))

        def __len__(self) -> int:
            return num_samples

        def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
            return self.features[index], self.targets[index]

    mock_dataset_instance = MockDataset()

    class MockWrapper(DatasetWrapperABC):
        @property
        def train_size(self):
            return len(mock_dataset_instance)

        @property
        def train_dataset(self):
            return mock_dataset_instance

        @property
        def test_dataset(self):
            return mock_dataset_instance

        @property
        def targets(self):
            return mock_dataset_instance.targets

        @property
        def train_transform(self):
            return None

        @property
        def eval_transform(self):
            return None

        def get_subset(self, indices):
            return Subset(mock_dataset_instance, indices)

    data_split = np.split(np.arange(num_samples).astype(np.int64), num_clients)
    mock_wrapper_instance = MockWrapper()

    return ValidTestObject(
        wrapper=mock_wrapper_instance,
        split=data_split,
        num_clients=num_clients,
        num_samples=num_samples,
    )


@pytest.fixture(
    params=[
        ("cpu", EvaluationConfig(num_workers=8)),
        ("cpu", EvaluationConfig()),
        ("cuda", EvaluationConfig(num_workers=8)),
        ("cuda", EvaluationConfig()),
        ("mps", EvaluationConfig(num_workers=8)),
        ("mps", EvaluationConfig()),
    ]
)
def data_manager(tmp_path, request, mocker, valid_test_object):
    device, eval_config = request.param

    data_config = DataConfig()
    optim_config = OptimizationConfig()

    mocker.patch(
        f"{MODULEPATH}.DataManager._init_wrapper",
        return_value=valid_test_object.wrapper,
    )
    mocker.patch(f"{MODULEPATH}.get_data_split", return_value=valid_test_object.split)

    return DataManager(
        num_clients=valid_test_object.num_clients,
        data_config=data_config,
        eval_config=eval_config,
        optim_config=optim_config,
        data_dir=tmp_path,
        device_type=device,
        visualize=False,
    )


@pytest.mark.parametrize("data_manager", [("cpu", EvaluationConfig())], indirect=True)
def test_client_dataloader_random_sampler_inputs(data_manager, mocker):
    mock_sampler = mocker.patch(f"{MODULEPATH}.RandomSampler")
    local_steps = data_manager.optim_config.num_local_steps
    batch_size = data_manager.optim_config.batch_size

    data_manager.get_client_dataloader(client_id=0)

    mock_sampler.assert_called_once_with(
        mocker.ANY,
        replacement=True,
        num_samples=local_steps * batch_size,
    )


def test_client_dataloader_inputs(data_manager, mocker):
    mock_client_dataloader = mocker.patch(f"{MODULEPATH}.DataLoader")
    optim_config = data_manager.optim_config

    data_manager.get_client_dataloader(client_id=0)

    mock_client_dataloader.assert_called_once_with(
        mocker.ANY,
        batch_size=optim_config.batch_size,
        sampler=mocker.ANY,
        num_workers=0,
        pin_memory=data_manager.device_type == "cuda",
        persistent_workers=False,
    )


@pytest.mark.parametrize("data_manager", [("cpu", EvaluationConfig())], indirect=True)
def test_client_weight_calculation(data_manager, valid_test_object):
    for cid in range(valid_test_object.num_clients):
        num_local_samples = valid_test_object.split[cid].shape[0]
        expected_weight = num_local_samples / valid_test_object.num_samples
        assert data_manager.get_client_weight(cid) == pytest.approx(expected_weight)


def test_eval_dataloader_inputs(data_manager, mocker):
    mock_eval_dataloader = mocker.patch(f"{MODULEPATH}.DataLoader")
    eval_batch_size = data_manager.eval_config.batch_size
    num_workers = data_manager.eval_config.num_workers

    data_manager.get_evaluation_dataloader()

    mock_eval_dataloader.assert_called_once_with(
        mocker.ANY,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_manager.device_type == "cuda",
        persistent_workers=num_workers > 0,
    )
