import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from afl_sim.checkpointing import CheckpointManager
from afl_sim.client import Client
from afl_sim.config import CheckpointConfig, MemStrategyConfig, OptimizationConfig
from afl_sim.enums import MemoryType
from afl_sim.models import LogisticRegression
from afl_sim.server import Server
from afl_sim.simulation import Simulation
from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncModelHistory,
    AsyncStateManager,
)
from afl_sim.timing.clock_types import ClockData, SimulationClock
from afl_sim.utils import MetricsLogger

_NUM_CLIENTS = 2
_COMM_TYPE = "sync"
_HISTORY_VERSIONS = (9, 10)
_GLOBAL_FIRST_IDX = 0
_EVENT_NUM = 2
_NUM_SAMPLES = 1
_NUM_CLASSES = 10
_IMG_SIZE = 8
_NUM_CHANNELS = 1
_MEM_TYPE = MemoryType.DISABLED
_DEVICE_TYPE = "cpu"
_CLIENT_REQUESTS = {0: 9, 1: 10}  # must be same size as _NUM_CLIENTS
_CURRENT_VERSION = 0


@pytest.fixture
def async_state_factory(model_factory):
    def _factory(
        comm_type=_COMM_TYPE,
        num_clients=_NUM_CLIENTS,
        history_versions=_HISTORY_VERSIONS,
        client_requests=_CLIENT_REQUESTS,
        num_channels=_NUM_CHANNELS,
        num_classes=_NUM_CLASSES,
        img_size=_IMG_SIZE,
    ):
        if comm_type == "sync":
            return None

        model = model_factory(
            num_channels=num_channels, num_classes=num_classes, img_size=img_size
        )

        torch.manual_seed(43)

        num_clients = num_clients

        model_requests = AsyncClientModelRequests(num_clients)
        for cid in range(num_clients):
            model_requests.update_client(cid, client_requests[cid])

        model_history = AsyncModelHistory(initial_model_dict=None)
        for version in history_versions:
            version_dict = {
                name: torch.rand_like(param, requires_grad=False)
                for name, param in model.named_parameters()
            }
            model_history.add_version(version, version_dict)

        async_states = AsyncStateManager(
            model_history=model_history, model_requests=model_requests
        )

        return async_states

    return _factory


@pytest.fixture
def simulation_clock_factory():
    def _factory(
        comm_type=_COMM_TYPE,
        num_clients=_NUM_CLIENTS,
        global_first_idx=_GLOBAL_FIRST_IDX,
        event_num=_EVENT_NUM,
    ):
        np.random.seed(42)
        if comm_type == "sync":
            client_ids = np.repeat(
                np.expand_dims(np.arange(num_clients, dtype=np.int64), axis=0),
                event_num,
                axis=0,
            )
        else:
            client_ids = np.random.randint(
                low=0, high=num_clients, size=event_num, dtype=np.int64
            )

        clock_data = ClockData(
            timestamps=np.arange(1, event_num + 1, dtype=np.float64),
            client_ids=client_ids,
        )

        return SimulationClock(
            clock_data=clock_data,
            global_first_idx=global_first_idx,
        )

    return _factory


@pytest.fixture
def dataloader_factory():
    def _factory(
        num_samples=_NUM_SAMPLES, img_size=_IMG_SIZE, num_classes=_NUM_CLASSES
    ):
        batch_size = 1
        dataset = TensorDataset(
            torch.rand(size=(num_samples, img_size, img_size)),
            torch.randint(low=0, high=num_classes, size=(num_samples,)),
        )

        subset_size = int(0.1 * num_samples) if num_samples > 10 else 1

        sampler = RandomSampler(
            data_source=dataset, replacement=True, num_samples=subset_size
        )

        return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

    return _factory


@pytest.fixture
def model_factory():
    def _factory(
        num_channels=_NUM_CHANNELS, num_classes=_NUM_CLASSES, img_size=_IMG_SIZE
    ):
        return LogisticRegression(
            in_channels=num_channels, num_classes=num_classes, image_size=img_size
        )

    return _factory


@pytest.fixture
def clients_factory(dataloader_factory, model_factory):
    def _factory(
        mem_type=_MEM_TYPE,
        num_clients=_NUM_CLIENTS,
        num_channels=_NUM_CHANNELS,
        num_classes=_NUM_CLASSES,
        img_size=_IMG_SIZE,
        num_samples=_NUM_SAMPLES,
    ):
        torch.manual_seed(42)

        model = model_factory(
            num_channels=num_channels, num_classes=num_classes, img_size=img_size
        )
        dataloader = dataloader_factory(
            num_samples=num_samples, img_size=img_size, num_classes=num_classes
        )
        mem_strategy = MemStrategyConfig(type=mem_type)

        clients = []
        for cid in range(num_clients):
            client = Client(
                client_id=cid,
                initial_model=model,
                dataloader=dataloader,
                weight=1 / num_clients,
                transform=None,
                optim_config=OptimizationConfig(),
                memory_strategy=mem_strategy,
                base_seed=42,
            )
            if mem_type.has_memory:
                client._memory = {
                    name: torch.rand_like(param, requires_grad=False)
                    for name, param in model.named_parameters()
                }
            clients.append(client)

        return clients

    return _factory


@pytest.fixture
def server_factory(dataloader_factory, simulation_clock_factory, model_factory):
    def _factory(
        num_clients=_NUM_CLIENTS,
        num_channels=_NUM_CHANNELS,
        num_classes=_NUM_CLASSES,
        img_size=_IMG_SIZE,
        num_samples=_NUM_SAMPLES,
        comm_type=_COMM_TYPE,
        global_first_idx=_GLOBAL_FIRST_IDX,
        event_num=_EVENT_NUM,
        current_version=_CURRENT_VERSION,
    ):
        torch.manual_seed(44)

        model = model_factory(
            num_channels=num_channels, num_classes=num_classes, img_size=img_size
        )
        dataloader = dataloader_factory(
            num_samples=num_samples, img_size=img_size, num_classes=num_classes
        )

        simulation_clock = simulation_clock_factory(
            comm_type=comm_type,
            num_clients=num_clients,
            global_first_idx=global_first_idx,
            event_num=event_num,
        )

        client_ids = simulation_clock.clock_data.client_ids
        goal = client_ids.shape[1] if client_ids.ndim > 1 else 1

        server = Server(
            model=model,
            test_loader=dataloader,
            test_transform=None,
            aggregation_goal=goal,
            num_clients=num_clients,
            reset_buffer=True,
            base_seed=42,
        )

        server.current_version = current_version

        return server

    return _factory


@pytest.fixture
def metrics_logger(tmp_path):
    return MetricsLogger(run_dir=tmp_path.joinpath("output"))


@pytest.fixture
def checkpoint_manager(tmp_path):
    return CheckpointManager(
        checkpoint_dir=tmp_path.joinpath("checkpoints"),
        checkpoint_config=CheckpointConfig(),
    )


@pytest.fixture
def simulation_factory(
    simulation_clock_factory,
    model_factory,
    server_factory,
    clients_factory,
    metrics_logger,
    checkpoint_manager,
    async_state_factory,
):
    def _factory(
        mem_type=_MEM_TYPE,
        comm_type=_COMM_TYPE,
        client_requests=_CLIENT_REQUESTS,
        history_versions=_HISTORY_VERSIONS,
        device_type=_DEVICE_TYPE,
        num_clients=_NUM_CLIENTS,
        global_first_idx=_GLOBAL_FIRST_IDX,
        num_channels=_NUM_CHANNELS,
        num_classes=_NUM_CLASSES,
        img_size=_IMG_SIZE,
        num_samples=_NUM_SAMPLES,
        event_num=_EVENT_NUM,
        current_version=_CURRENT_VERSION,
        local_idx=0,
        timeout=1000,
        stop_requested=False,
    ):
        mem_strategy = MemStrategyConfig(type=mem_type)

        model = model_factory(
            num_channels=num_channels, num_classes=num_classes, img_size=img_size
        )
        async_states = async_state_factory(
            comm_type=comm_type,
            history_versions=history_versions,
            client_requests=client_requests,
            num_clients=num_clients,
            num_channels=num_channels,
            num_classes=num_classes,
            img_size=img_size,
        )

        simulation_clock = simulation_clock_factory(
            comm_type=comm_type,
            num_clients=num_clients,
            global_first_idx=global_first_idx,
            event_num=event_num,
        )

        server = server_factory(
            num_clients=num_clients,
            num_channels=num_channels,
            num_classes=num_classes,
            img_size=img_size,
            num_samples=num_samples,
            comm_type=comm_type,
            global_first_idx=global_first_idx,
            event_num=event_num,
            current_version=current_version,
        )

        clients = clients_factory(
            mem_type=mem_type,
            num_clients=num_clients,
            num_channels=num_channels,
            num_classes=num_classes,
            img_size=img_size,
            num_samples=num_samples,
        )
        simulation = Simulation(
            mem_strategy=mem_strategy,
            timeout=timeout,
            metrics_logger=metrics_logger,
            checkpoint_manager=checkpoint_manager,
            device=torch.device(device_type),
            server=server,
            clients=clients,
            clock=simulation_clock,
            model_shell=model,
            async_states=async_states,
        )

        simulation.local_idx = local_idx
        simulation.stop_requested = stop_requested

        return simulation

    return _factory
