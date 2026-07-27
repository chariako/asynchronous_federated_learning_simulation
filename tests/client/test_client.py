from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.v2 import Identity, Transform

from afl_sim.client import Client
from afl_sim.config import MemStrategyConfig, OptimizationConfig
from afl_sim.enums import MemoryType
from afl_sim.models.logistic_regression import LogisticRegression
from afl_sim.types import SimulationModel


@dataclass
class ValidTestObject:
    model: SimulationModel
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]


@pytest.fixture
def dataset_and_model_factory():
    def _factory(num_samples: int = 1):
        batch_size = 1
        img_size = 8
        num_classes = 3
        num_channels = 1

        dataset = TensorDataset(
            torch.rand(size=(num_samples, img_size, img_size)),
            torch.randint(low=0, high=num_classes, size=(num_samples,)),
        )

        model = LogisticRegression(
            in_channels=num_channels, num_classes=num_classes, image_size=img_size
        )
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        return ValidTestObject(model=model, dataloader=data_loader)  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def client_factory():
    def _factory(
        model: SimulationModel,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        memory_type: MemoryType = MemoryType.DISABLED,
        seed: int = 42,
        client_id: int = 0,
        transform: Transform | None = None,
    ):
        batch_size = dataloader.batch_size
        assert batch_size

        return Client(
            client_id=client_id,
            initial_model=model,
            dataloader=dataloader,
            weight=0.1,
            transform=transform,
            optim_config=OptimizationConfig(
                learning_rate=0.1,
                weight_decay=0,
                num_local_steps=2,
                batch_size=batch_size,
            ),
            memory_strategy=MemStrategyConfig(type=memory_type),
            base_seed=seed,
        )

    return _factory


@pytest.mark.parametrize("mem_type", list(MemoryType))
def test_client_state_property(dataset_and_model_factory, client_factory, mem_type):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader, memory_type=mem_type)

    match client._memory_type:
        case MemoryType.DISABLED:
            assert not client.memory
        case MemoryType.GRADS:
            for name, param in model.named_parameters():
                assert torch.equal(
                    client.memory[name], torch.zeros_like(param, requires_grad=False)
                )
        case MemoryType.MODELS:
            for name, param in model.named_parameters():
                assert torch.equal(client.memory[name], param)


def test_train_optimizer_inputs(dataset_and_model_factory, client_factory, mocker):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader)

    expected_lr = client._base_lr * client._weight
    expected_decay = client._weight_decay

    mock_optimizer = mocker.patch("torch.optim.SGD")

    client._train_local(model=model, device=torch.device("cpu"))
    mock_optimizer.assert_called_once_with(
        params=mocker.ANY, lr=expected_lr, weight_decay=expected_decay
    )

    actual_params = mock_optimizer.call_args[1]["params"]
    assert list(actual_params) == list(model.parameters())


def test_train_local_zero_grad_optimization(
    dataset_and_model_factory, client_factory, mocker
):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader)

    spy_optimizer = mocker.spy(torch.optim.SGD, "zero_grad")

    client._train_local(model=model, device=torch.device("cpu"))

    spy_optimizer.assert_called_with(mocker.ANY, set_to_none=True)


@pytest.mark.parametrize(
    ("train_transform", "expected_transform"),
    [(lambda x: torch.mul(x, 2), lambda x: torch.mul(x, 2)), (None, Identity())],
)
def test_train_transform_application(
    dataset_and_model_factory,
    client_factory,
    mocker,
    train_transform,
    expected_transform,
):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader
    orig_input, _ = next(iter(dataloader))

    client = client_factory(
        model=model, dataloader=dataloader, transform=train_transform
    )

    spy = mocker.spy(model, name="forward")

    client._train_local(model=model, device=torch.device("cpu"))

    spy_input = spy.call_args_list[0][0][0]
    assert torch.equal(spy_input, expected_transform(orig_input))


@pytest.mark.parametrize(
    ("device_type", "expected_flag"), [("cpu", False), ("mps", False), ("cuda", True)]
)
def test_train_local_non_blocking_logic(
    dataset_and_model_factory, client_factory, mocker, device_type, expected_flag
):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader)

    test_device = torch.device(device_type)

    mock_inputs = mocker.MagicMock()
    mock_labels = mocker.MagicMock()

    mock_inputs.to.return_value = mock_inputs
    mock_labels.to.return_value = mock_labels

    mocker.patch.object(client, "_data_loader", [(mock_inputs, mock_labels)])

    mocker.patch.object(model, "forward", return_value=mocker.MagicMock())
    mocker.patch("torch.nn.CrossEntropyLoss", return_value=mocker.MagicMock())
    mocker.patch("torch.optim.SGD")

    client._train_local(model=model, device=test_device)

    mock_inputs.to.assert_called_with(test_device, non_blocking=expected_flag)
    mock_labels.to.assert_called_with(test_device, non_blocking=expected_flag)


def test_train_local_smoke_weight_update(dataset_and_model_factory, client_factory):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader)

    first_param = next(model.parameters())
    weights_before = first_param.detach().clone()

    client._train_local(model=model, device=torch.device("cpu"))

    weights_after = next(model.parameters())
    assert not torch.equal(weights_before, weights_after), (
        "Model weights did not update during training."
    )


def test_train_local_sets_train_mode(dataset_and_model_factory, client_factory):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader)

    model.eval()
    assert not model.training

    client._train_local(model=model, device=torch.device("cpu"))

    assert model.training


@pytest.mark.parametrize("mem_type", list(MemoryType))
def test_derive_update_logic(dataset_and_model_factory, client_factory, mem_type):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader, memory_type=mem_type)

    if mem_type.has_memory:
        random_memory = {
            name: torch.rand_like(param, requires_grad=False)
            for name, param in model.named_parameters()
        }
        client._memory = {
            name: random_memory[name].detach().clone()
            for name, _ in model.named_parameters()
        }

    initial_model_dict = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }
    update = client._derive_update(model, initial_model_dict)

    match client._memory_type:
        case MemoryType.DISABLED:
            for name, param in model.named_parameters():
                assert torch.equal(update[name], param - initial_model_dict[name])
        case MemoryType.GRADS:
            for name, param in model.named_parameters():
                assert torch.equal(
                    client._memory[name], param - initial_model_dict[name]
                )
                assert torch.equal(
                    update[name], param - initial_model_dict[name] - random_memory[name]
                )
        case MemoryType.MODELS:
            for name, param in model.named_parameters():
                assert torch.equal(client._memory[name], param)
                assert torch.equal(update[name], param - random_memory[name])


@pytest.mark.parametrize("mem_type", list(MemoryType))
def test_memory_dict_loading(dataset_and_model_factory, client_factory, mem_type):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client = client_factory(model=model, dataloader=dataloader, memory_type=mem_type)

    random_dict = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }

    client.load_mem_state_dict(random_dict)
    if not mem_type.has_memory:
        assert not client.memory
    else:
        for name, param in client.memory.items():
            assert torch.equal(param, random_dict[name])


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "global_idx1",
        "global_idx2",
        "seed1",
        "seed2",
        "client_id1",
        "client_id2",
        "expect_equal",
    ),
    [
        (100, 100, 42, 42, 0, 0, True),
        (100, 101, 42, 42, 0, 0, False),
        (100, 100, 42, 43, 0, 0, False),
        (100, 100, 42, 42, 0, 1, False),
    ],
)
def test_compute_update_integration(
    dataset_and_model_factory,
    client_factory,
    global_idx1,
    global_idx2,
    seed1,
    seed2,
    client_id1,
    client_id2,
    expect_equal,
):
    valid_test_object = dataset_and_model_factory(num_samples=100)
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    client1 = client_factory(
        model=model, dataloader=dataloader, seed=seed1, client_id=client_id1
    )
    client2 = client_factory(
        model=model, dataloader=dataloader, seed=seed2, client_id=client_id2
    )

    random_dict = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }

    update1 = client1.compute_update(
        model_shell=model,
        device=torch.device("cpu"),
        global_idx=global_idx1,
        requested_state_dict=random_dict,
    )

    update2 = client2.compute_update(
        model_shell=model,
        device=torch.device("cpu"),
        global_idx=global_idx2,
        requested_state_dict=random_dict,
    )

    for name, param in update1.items():
        assert torch.equal(param, update2[name]) == expect_equal
