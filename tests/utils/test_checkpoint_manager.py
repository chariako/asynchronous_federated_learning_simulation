from dataclasses import dataclass

import pytest
import torch

from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncModelHistory,
    AsyncStateManager,
    ClientMemoryStates,
)
from afl_sim.types import LatestMetadataSchema, ServerState
from afl_sim.utils.checkpoint_manager import CheckpointManager

MODULEPATH = "afl_sim.utils.checkpoint_manager.CheckpointManager"


def _valid_tensor_dict(float_list) -> dict[str, torch.Tensor]:
    return {"weights": torch.Tensor(float_list)}


def assert_tensor_dicts_equal(
    dict1: dict[str, torch.Tensor], dict2: dict[str, torch.Tensor]
):
    assert dict1.keys() == dict2.keys(), "Dictionary keys do not match."
    for key in dict1:
        assert torch.equal(dict1[key], dict2[key]), (
            f"Tensors for key '{key}' do not match."
        )


@dataclass
class ValidTestObject:
    global_idx: int
    num_clients: int
    server_state: ServerState
    client_states: ClientMemoryStates
    async_states: AsyncStateManager


@pytest.fixture
def checkpoint_manager(tmp_path):
    return CheckpointManager(checkpoint_dir=tmp_path)


@pytest.fixture
def get_valid_test_object():
    global_idx = 42
    num_clients = 2
    buffer = _valid_tensor_dict([0.3, 0.4])
    current_count = 3
    best_acc = 0.5
    current_acc = 0.4

    client_0_state = _valid_tensor_dict([0.5, 0.6])
    client_1_state = _valid_tensor_dict([0.7, 0.8])
    client_states = ClientMemoryStates()
    client_states.add_client_mem_state(0, client_0_state)
    client_states.add_client_mem_state(1, client_1_state)

    client_0_model_request = 9
    client_1_model_request = 10
    model_requests = AsyncClientModelRequests(num_clients)
    model_requests.update_client(0, client_0_model_request)
    model_requests.update_client(1, client_1_model_request)

    model_version_10 = _valid_tensor_dict([0.1, 0.2])
    model_version_9 = _valid_tensor_dict([0.9, 1.0])
    model_history = AsyncModelHistory(initial_model_dict=None)
    model_history.add_version(10, model_version_10)
    model_history.add_version(9, model_version_9)

    server_state = ServerState(
        model_state=model_version_10,
        buffer=buffer,
        current_count=current_count,
        best_acc=best_acc,
        current_acc=current_acc,
        current_version=client_1_model_request,
    )

    async_states = AsyncStateManager(
        model_history=model_history, model_requests=model_requests
    )

    return ValidTestObject(
        global_idx=global_idx,
        num_clients=num_clients,
        server_state=server_state,
        client_states=client_states,
        async_states=async_states,
    )


@pytest.fixture
def async_setup(request, get_valid_test_object):
    if request.param == "from_valid_object":
        states = get_valid_test_object.async_states
        return states, states.model_history.version_list
    return None, []


@pytest.mark.parametrize("async_setup", ["from_valid_object", "none"], indirect=True)
def test_save_latest_wrapper(
    checkpoint_manager, mocker, async_setup, get_valid_test_object
):
    mock_save_metadata = mocker.patch(f"{MODULEPATH}._save_latest_metadata")
    mock_save_server_states = mocker.patch(f"{MODULEPATH}._save_latest_server_dicts")
    mock_save_client_states = mocker.patch(f"{MODULEPATH}._save_latest_client_dicts")
    mock_save_async_states = mocker.patch(f"{MODULEPATH}._save_latest_async_states")

    valid_server_state = get_valid_test_object.server_state
    async_states, history_versions = async_setup
    global_idx = get_valid_test_object.global_idx

    checkpoint_manager.save_latest(
        server_state=valid_server_state,
        client_states=None,
        async_states=async_states,
        global_idx=global_idx,
    )

    mock_save_server_states.assert_called_once_with(
        server_state=valid_server_state, history_versions=history_versions
    )

    assert mock_save_metadata.called
    assert mock_save_client_states.called
    assert mock_save_async_states.called


def test_lastest_client_states_roundtrip(checkpoint_manager, get_valid_test_object):
    valid_client_states = get_valid_test_object.client_states
    num_clients = get_valid_test_object.num_clients

    checkpoint_manager._save_latest_client_dicts(client_states=valid_client_states)

    for cid in range(num_clients):
        loaded_dict = checkpoint_manager.load_client_memory_state(cid)
        assert_tensor_dicts_equal(
            loaded_dict, valid_client_states.get_client_mem_state(cid)
        )


def test_latest_client_state_none(checkpoint_manager, mocker):
    mock_atomic_write = mocker.patch(f"{MODULEPATH}._atomic_write")
    checkpoint_manager._save_latest_client_dicts(client_states=None)

    assert not mock_atomic_write.called


@pytest.fixture
def history_setup(request, get_valid_test_object):
    if request.param == "from_valid_object":
        return [get_valid_test_object.server_state.current_version]
    return []


@pytest.mark.parametrize("history_setup", ["from_valid_object", "none"], indirect=True)
def test_latest_server_states_roundtrip(
    checkpoint_manager, mocker, history_setup, get_valid_test_object
):
    valid_server_state = get_valid_test_object.server_state
    history_list = history_setup
    global_idx = get_valid_test_object.global_idx

    mocker.patch(
        f"{MODULEPATH}.load_latest_metadata",
        return_value=LatestMetadataSchema(
            global_idx=global_idx,
            best_acc=valid_server_state.best_acc,
            current_acc=valid_server_state.current_acc,
            current_version=valid_server_state.current_version,
            current_server_count=valid_server_state.current_count,
            history_version_list=history_list,
        ),
    )

    model_in_history = valid_server_state.current_version in history_list

    checkpoint_manager._save_latest_server_dicts(
        server_state=valid_server_state, history_versions=history_list
    )
    if model_in_history:
        torch.save(
            valid_server_state.model_state,
            checkpoint_manager._get_latest_history_version_filename(
                valid_server_state.current_version
            ),
        )

    loaded_server_state = checkpoint_manager.load_server_states()

    assert checkpoint_manager.latest_server_model_path.exists() != model_in_history

    assert valid_server_state.best_acc == loaded_server_state.best_acc
    assert valid_server_state.current_acc == loaded_server_state.current_acc
    assert valid_server_state.current_count == loaded_server_state.current_count
    assert valid_server_state.current_version == loaded_server_state.current_version

    assert_tensor_dicts_equal(valid_server_state.buffer, loaded_server_state.buffer)
    assert_tensor_dicts_equal(
        valid_server_state.model_state, loaded_server_state.model_state
    )


@pytest.mark.parametrize("async_setup", ["from_valid_object", "none"], indirect=True)
def test_latest_metadata_roundtrip(
    checkpoint_manager, async_setup, get_valid_test_object
):
    global_idx = get_valid_test_object.global_idx
    valid_server_state = get_valid_test_object.server_state
    async_states, _ = async_setup

    extracted_metadata = checkpoint_manager._extract_metadata(
        valid_server_state, async_states, global_idx=global_idx
    )
    checkpoint_manager._save_latest_metadata(
        valid_server_state, async_states, global_idx=global_idx
    )
    loaded_metadata = checkpoint_manager.load_latest_metadata()

    assert extracted_metadata.best_acc == loaded_metadata.best_acc
    assert extracted_metadata.current_acc == loaded_metadata.current_acc
    assert (
        extracted_metadata.current_server_count == loaded_metadata.current_server_count
    )
    assert extracted_metadata.current_version == loaded_metadata.current_version
    assert extracted_metadata.global_idx == loaded_metadata.global_idx
    assert (
        extracted_metadata.history_version_list == loaded_metadata.history_version_list
    )


def test_latest_async_states_roundtrip(checkpoint_manager, get_valid_test_object):
    num_clients = get_valid_test_object.num_clients
    valid_async_states = get_valid_test_object.async_states

    checkpoint_manager._save_latest_async_states(valid_async_states)
    loaded_requests = checkpoint_manager.load_model_requests(num_clients)

    for cid in range(num_clients):
        assert loaded_requests.get_client_request(
            cid
        ) == valid_async_states.model_requests.get_client_request(cid)

    for version in valid_async_states.model_history.version_list:
        loaded_version = checkpoint_manager.load_history_version(version)
        assert_tensor_dicts_equal(
            loaded_version, valid_async_states.model_history.get_version(version)
        )


def test_save_latest_async_states_none(checkpoint_manager, mocker):
    mock_atomic_write = mocker.patch(f"{MODULEPATH}._atomic_write")

    checkpoint_manager._save_latest_async_states(async_states=None)

    assert not mock_atomic_write.called


def test_unused_history_file_removal(checkpoint_manager, get_valid_test_object):
    valid_async_states = get_valid_test_object.async_states

    min_version = min(valid_async_states.model_history.version_list)

    old_version_paths = [
        checkpoint_manager._get_latest_history_version_filename(version)
        for version in range(min_version)
    ]

    for version_path in old_version_paths:
        version_path.touch()

    checkpoint_manager._save_latest_async_states(valid_async_states)

    for version_path in old_version_paths:
        assert not version_path.exists()


# -------- Raises --------


def test_load_tensordict_raises_file_not_found(checkpoint_manager):
    with pytest.raises(FileNotFoundError, match="could not be found"):
        checkpoint_manager._load_tensorDict(
            checkpoint_manager.latest_server_buffer_path
        )


def test_load_model_requests_raises_file_not_found(checkpoint_manager):
    with pytest.raises(FileNotFoundError, match="could not be found"):
        checkpoint_manager.load_model_requests(num_clients=10)


def test_load_model_requests_raises_key_error(checkpoint_manager):
    num_clients = 3

    incomplete_requests = AsyncClientModelRequests(2)
    incomplete_requests.update_client(0, 10)
    incomplete_requests.update_client(1, 10)
    filepath = checkpoint_manager.latest_model_requests_path

    checkpoint_manager._atomic_write(incomplete_requests, filepath)

    with pytest.raises(KeyError, match="request missing for client: 2"):
        checkpoint_manager.load_model_requests(num_clients)


def test_load_model_requests_raises_json_error(checkpoint_manager):
    corrupted_text = "{'0': corrupted_text}"
    checkpoint_manager.latest_model_requests_path.write_text(corrupted_text)

    with pytest.raises(
        ValueError, match="model request file is corrupted and cannot be parsed"
    ):
        checkpoint_manager.load_model_requests(num_clients=3)


def test_load_metadata_raises_file_not_found(checkpoint_manager):
    with pytest.raises(FileNotFoundError, match="could not be found"):
        checkpoint_manager.load_latest_metadata()
