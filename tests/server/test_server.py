from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.transforms.v2 import Identity

from afl_sim.models.logistic_regression import LogisticRegression
from afl_sim.server import Server
from afl_sim.types import ServerState, SimulationModel

MODULEPATH = "afl_sim.server.server.Server"


@dataclass
class ValidTestObject:
    model: SimulationModel
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]]


@pytest.fixture
def dataset_and_model_factory():
    def _factory(num_samples=1):
        batch_size = 1
        img_size = 8
        num_classes = 100
        num_channels = 1

        dataset = TensorDataset(
            torch.rand(size=(num_samples, img_size, img_size)),
            torch.randint(low=0, high=num_classes, size=(num_samples,)),
        )

        model = LogisticRegression(
            in_channels=num_channels, num_classes=num_classes, image_size=img_size
        )

        subset_size = int(0.1 * num_samples) if num_samples > 10 else 1

        sampler = RandomSampler(
            data_source=dataset, replacement=True, num_samples=subset_size
        )

        data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler
        )

        return ValidTestObject(model=model, dataloader=data_loader)  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def server_factory():
    def _factory(
        model,
        dataloader,
        transform=None,
        reset_buffer=False,
        seed=42,
        current_count=0,
        aggregation_goal=3,
        current_version=0,
    ):
        server = Server(
            model=model,
            test_loader=dataloader,
            test_transform=transform,
            aggregation_goal=aggregation_goal,
            num_clients=10,
            reset_buffer=reset_buffer,
            base_seed=seed,
        )

        server._current_count = current_count
        server.current_version = current_version

        return server

    return _factory


def test_server_initialization(server_factory, dataset_and_model_factory):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    assert server._current_count == 0
    assert server.current_version == 0

    for name, param in model.named_parameters():
        assert torch.equal(param, server.global_model_dict[name])
        assert torch.equal(
            server._buffer[name], torch.zeros_like(param, requires_grad=False)
        )


def test_state_property(server_factory, dataset_and_model_factory):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    server_state = server.state
    assert isinstance(server_state, ServerState)

    for name, param in server.global_model_dict.items():
        assert torch.equal(param, server_state.model_state[name])
        assert torch.equal(server_state.buffer[name], server._buffer[name])

    assert server_state.current_count == server._current_count
    assert server_state.current_version == server.current_version
    assert server_state.current_acc == server.current_acc
    assert server_state.best_acc == server.best_acc


def test_aggregate_updates(server_factory, dataset_and_model_factory):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    random_dict = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }

    server.aggregate_update(random_dict)

    assert server._current_count == 1

    # Buffer values after update should equal the injected update
    for name, param in random_dict.items():
        assert torch.equal(param, server._buffer[name])


@pytest.mark.parametrize("reset_buffer", [True, False])
def test_apply_buffer_update(server_factory, dataset_and_model_factory, reset_buffer):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(
        model=model, dataloader=dataloader, reset_buffer=reset_buffer, current_count=10
    )

    random_dict = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in server._buffer.items()
    }

    server._buffer = {
        name: param.detach().clone() for name, param in random_dict.items()
    }

    divisor = 2
    server._apply_buffer_update(divisor=divisor)

    for name, param in model.named_parameters():
        true_val = server.global_model_dict[name]
        expected_val = torch.add(torch.div(random_dict[name], divisor), param)

        assert torch.all(torch.isclose(true_val, expected_val))

    assert server._current_count == 0

    if reset_buffer:
        for name, param in model.named_parameters():
            assert torch.equal(
                server._buffer[name], torch.zeros_like(param, requires_grad=False)
            )
    else:
        for name, param in random_dict.items():
            assert torch.equal(server._buffer[name], param)


def test_server_state_load(server_factory, dataset_and_model_factory):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    random_state = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }
    random_buffer = {
        name: torch.rand_like(param, requires_grad=False)
        for name, param in model.named_parameters()
    }
    new_version = 100
    new_current_acc = 0.8
    new_best_acc = 0.9
    new_current_count = 10

    new_server_state = ServerState(
        model_state={
            name: param.detach().clone() for name, param in random_state.items()
        },
        buffer={name: param.detach().clone() for name, param in random_buffer.items()},
        current_version=new_version,
        current_acc=new_current_acc,
        best_acc=new_best_acc,
        current_count=new_current_count,
    )

    server.load_state_dict(new_server_state)

    assert server._current_count == new_current_count
    assert server.best_acc == new_best_acc
    assert server.current_version == new_version
    assert server.current_acc == new_current_acc

    for name, param in random_state.items():
        assert torch.equal(param, server.global_model_dict[name])
        assert torch.equal(random_buffer[name], server._buffer[name])


@pytest.mark.parametrize(
    (
        "current_version",
        "new_version",
        "current_count",
        "aggregation_goal",
        "expected_update",
    ),
    [(10, 10, 2, 3, False), (10, 11, 3, 3, True), (10, 11, 4, 3, True)],
)
def test_global_update_logic(
    server_factory,
    dataset_and_model_factory,
    current_version,
    new_version,
    current_count,
    aggregation_goal,
    expected_update,
    mocker,
):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(
        model=model,
        dataloader=dataloader,
        current_count=current_count,
        aggregation_goal=aggregation_goal,
        current_version=current_version,
    )

    mock_eval = mocker.patch(f"{MODULEPATH}._evaluate", return_value=None)
    mock_apply_update = mocker.patch(
        f"{MODULEPATH}._apply_buffer_update", return_value=None
    )

    update_performed = server.global_update(
        model_shell=model, device=torch.device("cpu"), global_idx=10, sim_time=123.4
    )

    assert update_performed == expected_update
    assert mock_eval.called == expected_update
    assert mock_apply_update.called == expected_update
    assert server.current_version == new_version


def test_eval_sets_eval_mode(dataset_and_model_factory, server_factory):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    server = server_factory(model=model, dataloader=dataloader)

    model.train()
    assert model.training

    server._evaluate(
        model_shell=model, device=torch.device("cpu"), global_idx=10, sim_time=123.4
    )

    assert not model.training


@pytest.mark.parametrize(
    ("device_type", "expected_flag"), [("cpu", False), ("mps", False), ("cuda", True)]
)
def test_eval_local_non_blocking_logic(
    dataset_and_model_factory, server_factory, mocker, device_type, expected_flag
):
    valid_test_object = dataset_and_model_factory()
    model = valid_test_object.model
    dataloader = valid_test_object.dataloader

    server = server_factory(model=model, dataloader=dataloader)

    test_device = torch.device(device_type)

    mock_inputs = mocker.MagicMock()
    mock_labels = mocker.MagicMock()

    mock_inputs.to.return_value = mock_inputs
    mock_labels.to.return_value = mock_labels

    mock_labels.size.return_value = 1

    mocker.patch.object(server, "_test_loader", [(mock_inputs, mock_labels)])

    mocker.patch.object(model, "forward", return_value=mocker.MagicMock())
    mocker.patch("torch.nn.CrossEntropyLoss", return_value=mocker.MagicMock())

    mock_predicted = mocker.MagicMock()
    mock_predicted.__eq__.return_value.sum.return_value.item.return_value = 1

    mocker.patch("torch.argmax", return_value=mock_predicted)
    mocker.patch(
        f"{MODULEPATH}._compute_and_update_metrics",
        return_value=(0.0, 0.0),
    )
    mocker.patch(f"{MODULEPATH}._update_logger")

    server._evaluate(
        model_shell=model, device=test_device, global_idx=10, sim_time=123.4
    )

    mock_inputs.to.assert_called_with(test_device, non_blocking=expected_flag)
    mock_labels.to.assert_called_with(test_device, non_blocking=expected_flag)


@pytest.mark.parametrize(
    ("eval_transform", "expected_transform"),
    [(lambda x: torch.mul(x, 2), lambda x: torch.mul(x, 2)), (None, Identity())],
)
def test_eval_transform_application(
    server_factory,
    dataset_and_model_factory,
    mocker,
    eval_transform,
    expected_transform,
):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(
        model=model, dataloader=dataloader, transform=eval_transform
    )
    orig_input, _ = next(iter(dataloader))

    spy = mocker.spy(model, name="forward")

    server._evaluate(
        model_shell=model, device=torch.device("cpu"), global_idx=10, sim_time=123.4
    )

    spy_input = spy.call_args_list[0][0][0]
    assert torch.equal(spy_input, expected_transform(orig_input))


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "current_count",
        "aggregation_goal",
        "seed1",
        "seed2",
        "global_idx1",
        "global_idx2",
        "expect_equal",
    ),
    [
        (3, 3, 42, 42, 100, 100, True),
        (3, 3, 42, 43, 100, 100, False),
        (3, 3, 42, 42, 100, 101, False),
    ],
)
def test_global_update_reproducibility(
    server_factory,
    dataset_and_model_factory,
    current_count,
    aggregation_goal,
    seed1,
    seed2,
    global_idx1,
    global_idx2,
    expect_equal,
):
    valid_obj = dataset_and_model_factory(num_samples=2000)
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server1 = server_factory(
        model=model,
        dataloader=dataloader,
        seed=seed1,
        current_count=current_count,
        aggregation_goal=aggregation_goal,
    )

    server2 = server_factory(
        model=model,
        dataloader=dataloader,
        seed=seed2,
        current_count=current_count,
        aggregation_goal=aggregation_goal,
    )

    server1.global_update(
        model_shell=model,
        device=torch.device("cpu"),
        global_idx=global_idx1,
        sim_time=123.4,
    )

    server2.global_update(
        model_shell=model,
        device=torch.device("cpu"),
        global_idx=global_idx2,
        sim_time=123.4,
    )

    server1_state = server1.state
    server2_state = server2.state

    assert server1_state.current_count == server2_state.current_count
    assert server1_state.current_version == server2_state.current_version
    for name, param in server1_state.buffer.items():
        assert torch.equal(param, server2_state.buffer[name])
        assert torch.equal(
            server1_state.model_state[name], server2_state.model_state[name]
        )

    assert (
        pytest.approx(server1_state.current_acc) == server2_state.current_acc
    ) == expect_equal
    assert (
        pytest.approx(server1_state.best_acc) == server2_state.best_acc
    ) == expect_equal
    assert (pytest.approx(server1.current_loss) == server2.current_loss) == expect_equal


@pytest.mark.parametrize(
    (
        "total_loss",
        "correct",
        "total",
        "num_batches",
        "expected_loss",
        "expected_acc",
    ),
    [
        (10.0, 80, 100, 5, 2.0, 80.0),
        (0.0, 50, 50, 2, 0.0, 100.0),
        (0.0, 0, 0, 0, 0.0, 0.0),
    ],
)
def test_compute_and_update_metrics_math(
    server_factory,
    dataset_and_model_factory,
    total_loss,
    correct,
    total,
    num_batches,
    expected_loss,
    expected_acc,
):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    assert server.current_acc == -1.0
    assert server.current_loss == -1.0

    avg_loss, accuracy = server._compute_and_update_metrics(
        total_loss, correct, total, num_batches
    )

    assert avg_loss == expected_loss
    assert accuracy == expected_acc

    assert server.current_loss == expected_loss
    assert server.current_acc == expected_acc


def test_best_acc_updates_conditionally(server_factory, dataset_and_model_factory):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    server._compute_and_update_metrics(
        total_loss=10.0, correct=50, total=100, num_batches=1
    )
    assert server.best_acc == 50.0

    server._compute_and_update_metrics(
        total_loss=15.0, correct=40, total=100, num_batches=1
    )
    assert server.current_acc == 40.0
    assert server.best_acc == 50.0

    server._compute_and_update_metrics(
        total_loss=5.0, correct=75, total=100, num_batches=1
    )
    assert server.current_acc == 75.0
    assert server.best_acc == 75.0

    server._compute_and_update_metrics(
        total_loss=5.0, correct=75, total=100, num_batches=1
    )
    assert server.best_acc == 75.0


def test_evaluate_metric_accumulation_integration(
    dataset_and_model_factory, server_factory, mocker
):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader

    mock_inputs = torch.zeros((1, 1))
    mock_labels = torch.zeros((1,), dtype=torch.long)
    fake_loader = [(mock_inputs, mock_labels), (mock_inputs, mock_labels)]

    server = server_factory(model=model, dataloader=dataloader)

    mocker.patch.object(server, "_test_loader", fake_loader)
    mocker.patch.object(model, "forward", return_value=torch.tensor([[1.0, 0.0]]))

    mock_loss = mocker.MagicMock()
    mock_loss.item.return_value = 2.5
    mocker.patch(
        "torch.nn.CrossEntropyLoss",
        return_value=mocker.MagicMock(return_value=mock_loss),
    )

    spy = mocker.spy(server, "_compute_and_update_metrics")

    mocker.patch(f"{MODULEPATH}._update_logger")

    server._evaluate(
        model_shell=model, device=torch.device("cpu"), global_idx=0, sim_time=0.0
    )

    spy.assert_called_once_with(total_loss=5.0, correct=2, total=2, num_batches=2)


def test_update_logger(dataset_and_model_factory, server_factory, capture_logs):
    valid_obj = dataset_and_model_factory()
    model = valid_obj.model
    dataloader = valid_obj.dataloader
    server = server_factory(model=model, dataloader=dataloader)

    server._update_logger(global_idx=100, sim_time=123.4, avg_loss=2.1, accuracy=10.1)
    expected_log = "Event:    100 | Time: 123.40 | Loss: 2.1000 | Acc: 10.10%"

    assert expected_log in capture_logs.text
