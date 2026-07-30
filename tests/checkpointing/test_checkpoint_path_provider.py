import pytest

from afl_sim.enums import CheckpointFile


@pytest.mark.parametrize(
    ("tmp", "subfolder", "target_file", "expected_path"),
    [
        (
            True,
            "test",
            CheckpointFile.SERVER_BUFFER,
            "tmp_test/latest_server_buffer.pt",
        ),
        (False, "test", CheckpointFile.SERVER_BUFFER, "test/latest_server_buffer.pt"),
    ],
)
def test_get_path(
    path_provider_factory, tmp_path, tmp, subfolder, target_file, expected_path
):
    path_provider = path_provider_factory(subfolder)
    assert path_provider.get_path(target_file, tmp=tmp) == tmp_path.joinpath(
        expected_path
    )


@pytest.mark.parametrize(
    ("tmp", "subfolder", "cid", "expected_path"),
    [
        (
            True,
            "test",
            1,
            "tmp_test/latest_client_1_state.pt",
        ),
        (False, "test", 1, "test/latest_client_1_state.pt"),
    ],
)
def test_get_client_path(
    tmp_path, path_provider_factory, tmp, subfolder, cid, expected_path
):
    path_provider = path_provider_factory(subfolder)
    assert path_provider.get_client_state_path(cid=cid, tmp=tmp) == tmp_path.joinpath(
        expected_path
    )


@pytest.mark.parametrize(
    ("tmp", "subfolder", "version", "expected_path"),
    [
        (
            True,
            "test",
            10,
            "tmp_test/latest_history_version_10.pt",
        ),
        (False, "test", 10, "test/latest_history_version_10.pt"),
        (
            True,
            "test",
            "*",
            "tmp_test/latest_history_version_*.pt",
        ),
        (False, "test", "*", "test/latest_history_version_*.pt"),
    ],
)
def test_get_history_version_path(
    tmp_path, path_provider_factory, tmp, subfolder, version, expected_path
):
    path_provider = path_provider_factory(subfolder)
    assert path_provider.get_history_version_path(
        version=version, tmp=tmp
    ) == tmp_path.joinpath(expected_path)
