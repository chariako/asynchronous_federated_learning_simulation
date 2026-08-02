import json

import pytest
import torch
from torch import testing

from afl_sim.checkpointing.checkpoint_helpers import (
    _extract_metadata,
    _save_latest_async_states_to_tmp_dir,
    _save_latest_client_dicts_to_tmp_dir,
    _save_latest_metadata_to_tmp_dir,
    _save_latest_server_dicts_to_tmp_dir,
)
from afl_sim.enums import CheckpointFile
from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
)
from afl_sim.types import LatestMetadataSchema
from tests.checkpointing.helpers import (
    valid_tensor_dict,
)


def test_latest_async_states_roundtrip(
    checkpoint_manager_factory, get_valid_checkpoint_states
):
    checkpoint_manager = checkpoint_manager_factory()
    ckpt_dir = checkpoint_manager.path_provider.ckpt_dir
    tmp_dir = checkpoint_manager.path_provider.tmp_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    num_clients = valid_states.num_clients
    valid_async_states = valid_states.async_states

    _save_latest_async_states_to_tmp_dir(
        valid_async_states, path_provider=checkpoint_manager.path_provider
    )

    tmp_dir.replace(ckpt_dir)

    loaded_requests = checkpoint_manager.load_model_requests(num_clients)

    for cid in range(num_clients):
        assert loaded_requests.get_client_request(
            cid
        ) == valid_async_states.model_requests.get_client_request(cid)

    for version in valid_async_states.model_history.version_list:
        loaded_version = checkpoint_manager.load_history_version(version)
        testing.assert_close(
            loaded_version,
            valid_async_states.model_history.get_version(version),
            atol=0.0,
            rtol=0.0,
        )


def test_lastest_client_states_roundtrip(
    checkpoint_manager_factory, get_valid_checkpoint_states
):
    checkpoint_manager = checkpoint_manager_factory()
    ckpt_dir = checkpoint_manager.path_provider.ckpt_dir
    tmp_dir = checkpoint_manager.path_provider.tmp_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    valid_client_states = valid_states.client_states
    num_clients = valid_states.num_clients

    _save_latest_client_dicts_to_tmp_dir(
        client_states=valid_client_states,
        path_provider=checkpoint_manager.path_provider,
    )

    tmp_dir.replace(ckpt_dir)

    for cid in range(num_clients):
        loaded_dict = checkpoint_manager.load_client_memory_state(cid)
        testing.assert_close(
            loaded_dict,
            valid_client_states.get_client_mem_state(cid),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    ("current_version", "history_versions"), [(10, (9, 10)), (10, (8, 9))]
)
def test_latest_server_states_roundtrip(
    current_version,
    history_versions,
    checkpoint_manager_factory,
    mocker,
    get_valid_checkpoint_states,
):
    checkpoint_manager = checkpoint_manager_factory()
    ckpt_dir = checkpoint_manager.path_provider.ckpt_dir
    tmp_dir = checkpoint_manager.path_provider.tmp_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states(
        current_version=current_version, history_versions=history_versions
    )

    valid_server_state = valid_states.server_state
    history_list = list(history_versions)
    global_idx = 42

    mocker.patch.object(
        checkpoint_manager,
        "load_latest_metadata",
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

    _save_latest_server_dicts_to_tmp_dir(
        server_state=valid_server_state,
        history_versions=history_list,
        path_provider=checkpoint_manager.path_provider,
    )

    if model_in_history:
        version_path = checkpoint_manager.path_provider.get_history_version_path(
            version=current_version, tmp=True
        )
        torch.save(valid_server_state.model_state, version_path)

    tmp_dir.replace(ckpt_dir)

    loaded_server_state = checkpoint_manager.load_server_states()

    model_path = checkpoint_manager.path_provider.get_path(CheckpointFile.SERVER_STATE)
    assert model_path.exists() != model_in_history

    assert valid_server_state.best_acc == loaded_server_state.best_acc
    assert valid_server_state.current_acc == loaded_server_state.current_acc
    assert valid_server_state.current_count == loaded_server_state.current_count
    assert valid_server_state.current_version == loaded_server_state.current_version

    testing.assert_close(
        valid_server_state.buffer, loaded_server_state.buffer, atol=0.0, rtol=0.0
    )
    testing.assert_close(
        valid_server_state.model_state,
        loaded_server_state.model_state,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("async_setup", ["from_valid_states", "none"], indirect=True)
def test_latest_metadata_roundtrip(
    checkpoint_manager_factory, async_setup, get_valid_checkpoint_states
):
    checkpoint_manager = checkpoint_manager_factory()
    ckpt_dir = checkpoint_manager.path_provider.ckpt_dir
    tmp_dir = checkpoint_manager.path_provider.tmp_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    global_idx = 42
    valid_server_state = valid_states.server_state
    async_states, _ = async_setup

    extracted_metadata = _extract_metadata(
        valid_server_state, async_states, global_idx=global_idx
    )
    _save_latest_metadata_to_tmp_dir(
        valid_server_state,
        async_states,
        global_idx=global_idx,
        path_provider=checkpoint_manager.path_provider,
    )
    tmp_dir.replace(ckpt_dir)
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


@pytest.mark.parametrize(
    (
        "checkpoint_interval",
        "sim_duration",
        "expected_save",
        "expected_last_chkpt_time",
        "expected_log",
    ),
    [(100, 101, True, 101, True), (100, 90, False, 0, False)],
)
def test_save_latest(
    checkpoint_interval,
    sim_duration,
    expected_save,
    expected_last_chkpt_time,
    expected_log,
    checkpoint_manager_factory,
    get_valid_checkpoint_states,
    capture_logs,
    mocker,
):
    checkpoint_manager = checkpoint_manager_factory(interval=checkpoint_interval)
    mock_atomic_write = mocker.patch(
        "afl_sim.checkpointing.checkpoint_manager.atomic_write_latest"
    )

    valid_states = get_valid_checkpoint_states()
    valid_server_state = valid_states.server_state

    checkpoint_manager.save_latest(
        server_state=valid_server_state,
        client_states=None,
        async_states=None,
        global_idx=42,
        sim_duration=sim_duration,
    )

    assert checkpoint_manager.last_checkpoint_time == expected_last_chkpt_time
    assert mock_atomic_write.called == expected_save
    assert ("checkpoint successfully saved" in capture_logs.text) == expected_log


def test_save_shutdown(
    checkpoint_manager_factory,
    get_valid_checkpoint_states,
    capture_logs,
    mocker,
):
    checkpoint_manager = checkpoint_manager_factory()
    mock_atomic_write = mocker.patch(
        "afl_sim.checkpointing.checkpoint_manager.atomic_write_latest"
    )

    valid_states = get_valid_checkpoint_states()
    valid_server_state = valid_states.server_state

    checkpoint_manager.save_shutdown(
        server_state=valid_server_state,
        client_states=None,
        async_states=None,
        global_idx=42,
    )

    assert mock_atomic_write.called
    assert "checkpoint successfully saved before global event: 42" in capture_logs.text


@pytest.mark.parametrize(
    ("keep_best", "current_acc", "best_acc", "expect_save"),
    [
        (False, 80.0, 40.0, False),
        (False, 40.0, 80.0, False),
        (True, 80.0, 40.0, True),
        (True, 40.0, 80.0, False),
    ],
)
def test_save_best(
    keep_best, current_acc, best_acc, expect_save, checkpoint_manager_factory
):
    checkpoint_manager = checkpoint_manager_factory(keep_best=keep_best)
    checkpoint_manager.path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager.save_best(
        model_state_dict=valid_tensor_dict(),
        current_acc=current_acc,
        best_acc=best_acc,
    )
    model_path = checkpoint_manager.path_provider.get_path(CheckpointFile.BEST_MODEL)
    metadata_path = checkpoint_manager.path_provider.get_path(
        CheckpointFile.BEST_METADATA
    )

    assert model_path.exists() == expect_save
    assert metadata_path.exists() == expect_save


# -------- Raises --------


def test_load_model_requests_raises_file_not_found(checkpoint_manager_factory):
    checkpoint_manager = checkpoint_manager_factory()
    with pytest.raises(FileNotFoundError, match="could not be found"):
        checkpoint_manager.load_model_requests(num_clients=10)


def test_load_model_requests_raises_key_error(checkpoint_manager_factory):
    checkpoint_manager = checkpoint_manager_factory()
    checkpoint_manager.path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)
    num_clients = 3

    incomplete_requests = AsyncClientModelRequests(2)
    incomplete_requests.update_client(0, 10)
    incomplete_requests.update_client(1, 10)
    req_file = checkpoint_manager.path_provider.get_path(CheckpointFile.MODEL_REQUESTS)

    with open(req_file, "w") as file:
        json.dump(incomplete_requests.state_dict, file, indent=4)

    with pytest.raises(KeyError, match="request missing for client: 2"):
        checkpoint_manager.load_model_requests(num_clients)


def test_load_model_requests_raises_json_error(checkpoint_manager_factory):
    checkpoint_manager = checkpoint_manager_factory()
    checkpoint_manager.path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)

    corrupted_text = "{'0': corrupted_text}"
    req_file = checkpoint_manager.path_provider.get_path(CheckpointFile.MODEL_REQUESTS)
    req_file.write_text(corrupted_text)

    with pytest.raises(
        ValueError, match="model request file is corrupted and cannot be parsed"
    ):
        checkpoint_manager.load_model_requests(num_clients=3)


def test_load_metadata_raises_file_not_found(checkpoint_manager_factory):
    checkpoint_manager = checkpoint_manager_factory()
    with pytest.raises(FileNotFoundError, match="could not be found"):
        checkpoint_manager.load_latest_metadata()
