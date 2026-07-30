import pytest
from helpers import (  # type: ignore[import-not-found]
    assert_copied_files,
    assert_tensor_dicts_equal,
    valid_tensor_dict,
)

from afl_sim.checkpointing.checkpoint_helpers import (
    _copy_best_checkpoint_to_tmp_dir,
    _save_all_latest_states_to_tmp_dir,
    _save_latest_async_states_to_tmp_dir,
    _save_latest_client_dicts_to_tmp_dir,
    _save_latest_metadata_to_tmp_dir,
    _save_latest_server_dicts_to_tmp_dir,
    atomic_tensor_dict_write,
    atomic_write_latest,
    load_tensor_dict,
)
from afl_sim.enums import CheckpointFile

MODULEPATH = "afl_sim.checkpointing.checkpoint_helpers"

# ======= Raises =======


def test_load_tensor_dict_raises_file_not_found(tmp_path):
    path_name = tmp_path.joinpath("test_tensor.pt")
    with pytest.raises(FileNotFoundError, match="could not be found"):
        load_tensor_dict(path_name)


def test_atomic_write_latest_raises_permission_error(
    path_provider_factory, get_valid_checkpoint_states, mocker, capture_logs
):
    path_provider = path_provider_factory()
    path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()

    mocker.patch(f"{MODULEPATH}._save_all_latest_states_to_tmp_dir")
    mocker.patch("pathlib.Path.replace", side_effect=PermissionError)

    with pytest.raises(RuntimeError, match="Failed to atomically replace checkpoint"):
        atomic_write_latest(
            server_state=valid_states.server_state,
            client_states=None,
            async_states=None,
            global_idx=42,
            path_provider=path_provider,
        )

    assert "Permission denied when overwriting" in capture_logs.text


def test_atomic_tensor_write_raises_runtime_error(tmp_path, mocker):
    orig_tensor_dict = valid_tensor_dict()
    path_name = tmp_path.joinpath("test_tensor.pt")
    tmp_path_name = path_name.parent / ("tmp_" + path_name.name)

    mocker.patch("pathlib.Path.replace", side_effect=Exception)

    with pytest.raises(RuntimeError, match="Critical checkpoint failure"):
        atomic_tensor_dict_write(orig_tensor_dict, path_name)

    assert not tmp_path_name.exists()


# ======= Main tests =======


def test_tensor_dict_round_trip(tmp_path):
    orig_tensor_dict = valid_tensor_dict()
    path_name = tmp_path.joinpath("test_tensor.pt")
    atomic_tensor_dict_write(orig_tensor_dict, path_name)

    loaded_tensor_dict = load_tensor_dict(path_name)

    assert_tensor_dicts_equal(orig_tensor_dict, loaded_tensor_dict)


def test_atomic_write_path_replacement(
    path_provider_factory, get_valid_checkpoint_states, mocker
):
    path_provider = path_provider_factory()
    path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()

    # Add mock files to tmp_dir
    path_provider.get_path(CheckpointFile.BEST_METADATA, tmp=True).touch()
    path_provider.get_history_version_path(version=10, tmp=True).touch()

    mocker.patch(f"{MODULEPATH}._save_all_latest_states_to_tmp_dir")

    atomic_write_latest(
        server_state=valid_states.server_state,
        client_states=None,
        async_states=None,
        global_idx=42,
        path_provider=path_provider,
    )

    assert not path_provider.tmp_dir.exists()
    assert path_provider.get_path(CheckpointFile.BEST_METADATA, tmp=False).exists()
    assert path_provider.get_history_version_path(version=10, tmp=False).exists()


@pytest.mark.parametrize("async_setup", ["from_valid_states", "none"], indirect=True)
def test_save_all_states_to_tmp(
    async_setup, path_provider_factory, get_valid_checkpoint_states, mocker
):
    path_provider = path_provider_factory()
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    async_states, history_versions = async_setup

    mock_save_metadata = mocker.patch(f"{MODULEPATH}._save_latest_metadata_to_tmp_dir")
    mock_save_server = mocker.patch(
        f"{MODULEPATH}._save_latest_server_dicts_to_tmp_dir"
    )
    mock_save_clients = mocker.patch(
        f"{MODULEPATH}._save_latest_client_dicts_to_tmp_dir"
    )
    mock_save_async_states = mocker.patch(
        f"{MODULEPATH}._save_latest_async_states_to_tmp_dir"
    )
    mock_copy_best = mocker.patch(f"{MODULEPATH}._copy_best_checkpoint_to_tmp_dir")

    _save_all_latest_states_to_tmp_dir(
        server_state=valid_states.server_state,
        client_states=None,
        async_states=async_states,
        global_idx=42,
        path_provider=path_provider,
    )

    mock_save_server.assert_called_once_with(
        server_state=valid_states.server_state,
        history_versions=history_versions,
        path_provider=path_provider,
    )

    assert mock_save_metadata.called
    assert mock_save_clients.called
    assert mock_save_async_states.called
    assert mock_copy_best.called


@pytest.mark.parametrize(
    ("current_version", "history_versions", "expect_model_save"),
    [(10, [8, 9], True), (10, [9, 10], False)],
)
def test_save_latest_server_dicts_to_tmp(
    current_version,
    history_versions,
    expect_model_save,
    path_provider_factory,
    get_valid_checkpoint_states,
):
    path_provider = path_provider_factory()
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states(current_version=current_version)
    server_state = valid_states.server_state

    _save_latest_server_dicts_to_tmp_dir(
        server_state=server_state,
        history_versions=history_versions,
        path_provider=path_provider,
    )

    assert (
        path_provider.get_path(CheckpointFile.SERVER_STATE, tmp=True).exists()
        == expect_model_save
    )
    assert path_provider.get_path(CheckpointFile.SERVER_BUFFER, tmp=True).exists()


def test_save_latest_metadata_to_tmp(
    path_provider_factory, get_valid_checkpoint_states
):
    path_provider = path_provider_factory()
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    server_state = valid_states.server_state

    _save_latest_metadata_to_tmp_dir(
        server_state=server_state,
        async_states=None,
        global_idx=42,
        path_provider=path_provider,
    )

    assert path_provider.get_path(CheckpointFile.LATEST_METADATA, tmp=True).exists()


@pytest.mark.parametrize(
    ("model_exists", "metadata_exists"),
    [(True, False), (True, True), (False, False), (False, True)],
)
def test_copy_best_to_tmp(model_exists, metadata_exists, path_provider_factory):
    path_provider = path_provider_factory()
    path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    orig_model_path = path_provider.get_path(CheckpointFile.BEST_MODEL, tmp=False)
    tmp_model_path = path_provider.get_path(CheckpointFile.BEST_MODEL, tmp=True)

    orig_metadata_path = path_provider.get_path(CheckpointFile.BEST_METADATA, tmp=False)
    tmp_metadata_path = path_provider.get_path(CheckpointFile.BEST_METADATA, tmp=True)

    if model_exists:
        orig_model_path.touch()
    if metadata_exists:
        orig_metadata_path.touch()

    _copy_best_checkpoint_to_tmp_dir(path_provider)

    assert tmp_model_path.exists() == model_exists
    assert tmp_metadata_path.exists() == metadata_exists

    if model_exists:
        assert_copied_files(orig_model_path, tmp_model_path)
    if metadata_exists:
        assert_copied_files(orig_metadata_path, tmp_metadata_path)


def test_save_latest_client_dicts_tmp(
    path_provider_factory, get_valid_checkpoint_states
):
    path_provider = path_provider_factory()
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states()
    client_states = valid_states.client_states
    num_clients = valid_states.num_clients

    _save_latest_client_dicts_to_tmp_dir(
        client_states=client_states, path_provider=path_provider
    )

    for cid in range(num_clients):
        assert path_provider.get_client_state_path(cid=cid, tmp=True).exists()


def test_save_latest_client_dicts_none(path_provider_factory, mocker):
    path_provider = path_provider_factory()
    mock_save = mocker.patch("torch.save")

    _save_latest_client_dicts_to_tmp_dir(
        client_states=None, path_provider=path_provider
    )

    assert not mock_save.called


@pytest.mark.parametrize(
    ("history_versions", "existing_version"),
    [
        ((9, 10), 0),
        ((9, 10), 10),
    ],
)
def test_save_latest_async_states_tmp(
    history_versions,
    existing_version,
    path_provider_factory,
    get_valid_checkpoint_states,
):
    path_provider = path_provider_factory()
    path_provider.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    valid_states = get_valid_checkpoint_states(history_versions=history_versions)
    async_states = valid_states.async_states

    path_provider.get_history_version_path(version=existing_version, tmp=False).touch()

    _save_latest_async_states_to_tmp_dir(
        async_states=async_states, path_provider=path_provider
    )

    assert path_provider.get_path(CheckpointFile.MODEL_REQUESTS, tmp=True).exists()

    if existing_version not in history_versions:
        assert not path_provider.get_history_version_path(
            version=existing_version, tmp=True
        ).exists()

    for version in history_versions:
        assert path_provider.get_history_version_path(
            version=version, tmp=True
        ).exists()

        if version == existing_version:
            src_file = path_provider.get_history_version_path(
                version=existing_version, tmp=False
            )
            dst_file = path_provider.get_history_version_path(
                version=existing_version, tmp=True
            )
            assert_copied_files(src_file, dst_file)


def test_save_latest_async_states_none(path_provider_factory, mocker):
    path_provider = path_provider_factory()
    mock_save = mocker.patch("torch.save")
    mock_dump = mocker.patch("json.dump")

    _save_latest_async_states_to_tmp_dir(async_states=None, path_provider=path_provider)

    assert not mock_save.called
    assert not mock_dump.called
