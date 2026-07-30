import random
from dataclasses import dataclass

import pytest
from helpers import valid_tensor_dict  # type: ignore[import-not-found]

from afl_sim.checkpointing.checkpoint_manager import CheckpointManager
from afl_sim.checkpointing.checkpoint_path_provider import CheckpointPathProvider
from afl_sim.config import CheckpointConfig
from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncModelHistory,
    AsyncStateManager,
    ClientMemoryStates,
)
from afl_sim.types import ServerState


@pytest.fixture
def async_setup(request, get_valid_checkpoint_states):
    if request.param == "from_valid_states":
        valid_states = get_valid_checkpoint_states()
        states = valid_states.async_states
        return states, states.model_history.version_list
    return None, []


@pytest.fixture
def checkpoint_manager_factory(tmp_path):
    def _factory(subfolder="test", interval=100, keep_best=False):
        checkpoint_config = CheckpointConfig(
            interval_seconds=interval, keep_best=keep_best
        )
        return CheckpointManager(
            checkpoint_dir=tmp_path.joinpath(subfolder),
            checkpoint_config=checkpoint_config,
        )

    return _factory


@pytest.fixture
def path_provider_factory(tmp_path):
    def _factory(subfolder="test"):
        return CheckpointPathProvider(checkpoint_dir=tmp_path.joinpath(subfolder))

    return _factory


@dataclass
class ValidCheckpointStates:
    global_idx: int
    num_clients: int
    server_state: ServerState
    client_states: ClientMemoryStates
    async_states: AsyncStateManager


@pytest.fixture
def get_valid_checkpoint_states():
    def _factory(current_version=10, history_versions=(9, 10), num_clients=2):
        global_idx = 42
        num_clients = num_clients
        current_count = 3
        best_acc = 40
        current_acc = 30

        client_states = ClientMemoryStates()
        for cid in range(num_clients):
            client_states.add_client_mem_state(cid, valid_tensor_dict())

        model_requests = AsyncClientModelRequests(num_clients)
        for cid in range(num_clients - 1):
            model_requests.update_client(cid, random.randint(0, current_version))
        model_requests.update_client(num_clients - 1, current_version)

        model_history = AsyncModelHistory(initial_model_dict=None)
        for version in history_versions:
            model_history.add_version(version, valid_tensor_dict())

        model_state = (
            model_history.get_version(current_version)
            if current_version in model_history.version_list
            else valid_tensor_dict()
        )

        server_state = ServerState(
            model_state=model_state,
            buffer=valid_tensor_dict(),
            current_count=current_count,
            best_acc=best_acc,
            current_acc=current_acc,
            current_version=current_version,
        )

        async_states = AsyncStateManager(
            model_history=model_history, model_requests=model_requests
        )

        return ValidCheckpointStates(
            global_idx=global_idx,
            num_clients=num_clients,
            server_state=server_state,
            client_states=client_states,
            async_states=async_states,
        )

    return _factory
