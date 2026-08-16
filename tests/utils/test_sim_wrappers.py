import signal
import uuid

import pytest
import yaml

from afl_sim.config import AppConfig, OptimizationConfig, SimulationConfig
from afl_sim.paths import SimulationDirectories
from afl_sim.utils.sim_wrappers import (
    _create_run_directory,
    _load_dict_from_yaml,
    build_and_run_simulation,
    get_simulation_dirs_from_metadata,
    graceful_interrupt_handler,
    load_config_from_run_dir_with_overrides,
    load_config_with_overrides,
    save_effective_config,
    save_simulation_metadata,
    setup_simulation_directories,
)


@pytest.fixture
def get_mock_simulation(mocker):
    simulation = mocker.MagicMock(stop_requested=False)
    simulation.run.return_value = None
    return simulation


@pytest.mark.parametrize("kill_signal", [signal.SIGINT, signal.SIGTERM])
def test_graceful_interrupt(get_mock_simulation, mocker, kill_signal):
    simulation = get_mock_simulation
    mock_signal = mocker.patch("signal.signal")

    with graceful_interrupt_handler(simulation):
        for call in mock_signal.call_args_list:
            if call[0][0] == kill_signal:
                registered_handler = call[0][1]
                break

        registered_handler(kill_signal, None)

        assert simulation.stop_requested


def test_graceful_interrupt_restores_signals(mocker, get_mock_simulation):
    mocker.patch(
        "signal.getsignal",
        side_effect=["mock_original_sigint", "mock_original_sigterm"],
    )
    mock_signal = mocker.patch("signal.signal")

    with graceful_interrupt_handler(get_mock_simulation):
        assert mock_signal.call_count == 2

    assert mock_signal.call_count == 4

    mock_signal.assert_has_calls(
        [
            mocker.call(signal.SIGINT, "mock_original_sigint"),
            mocker.call(signal.SIGTERM, "mock_original_sigterm"),
        ],
        any_order=False,
    )


@pytest.mark.parametrize(
    ("tag", "expected_tag"),
    [("safe_tag", "_safe_tag"), ("not//safe\\ tag", "_not__safe_-tag"), (None, "")],
)
def test_create_run_directory(tmp_path, tag, expected_tag, mocker):
    uuid_spy = mocker.spy(uuid, "uuid4")
    dir = _create_run_directory(tmp_path, tag)

    short_hash = str(uuid_spy.spy_return)[:6]

    assert str(dir)[-len(expected_tag) - 6 :] == short_hash + expected_tag


def test_load_yaml(tmp_path):
    mock_dict = {"key1": 1, "key2": 2}
    mock_yaml = tmp_path / "mock.yaml"

    with open(mock_yaml, "w", encoding="utf-8") as f:
        yaml.dump(mock_dict, f)

    loaded_dict = _load_dict_from_yaml(mock_yaml)
    assert loaded_dict == mock_dict


def test_load_yaml_raises_value_error(tmp_path):
    mock_dict = "corrupt_string"
    mock_yaml = tmp_path / "mock.yaml"

    with open(mock_yaml, "w", encoding="utf-8") as f:
        yaml.dump(mock_dict, f)

    with pytest.raises(ValueError, match="must parse to a dictionary"):
        _load_dict_from_yaml(mock_yaml)


@pytest.mark.parametrize(
    (
        "timeout_init",
        "learning_rate_init",
        "timeout_updated",
        "learning_rate_updated",
        "timeout_expected",
        "learning_rate_expected",
    ),
    [(100.0, 0.1, None, None, 100.0, 0.1), (100.0, 0.1, 200.0, 0.2, 200.0, 0.2)],
)
def test_load_config_with_override(
    timeout_init,
    learning_rate_init,
    timeout_updated,
    learning_rate_updated,
    timeout_expected,
    learning_rate_expected,
    mocker,
    tmp_path,
):
    sim_config = SimulationConfig(timeout_seconds=timeout_init)
    optim_config = OptimizationConfig(learning_rate=learning_rate_init)

    config = AppConfig(simulation=sim_config, optimization=optim_config)

    mocker.patch(
        "afl_sim.utils.sim_wrappers._load_dict_from_yaml",
        return_value=config.model_dump(),
    )

    updated_config = load_config_with_overrides(
        tmp_path, learning_rate=learning_rate_updated, timeout=timeout_updated
    )

    assert updated_config.simulation.timeout_seconds == timeout_expected
    assert updated_config.optimization.learning_rate == learning_rate_expected


def test_setup_simulation_dirs(tmp_path, mocker):
    folder_name = "123"
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / folder_name
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"

    mocker.patch(
        "afl_sim.utils.sim_wrappers._create_run_directory", return_value=run_dir
    )

    sim_dirs = setup_simulation_directories(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        tag=None,
    )

    assert sim_dirs.output_dir == run_dir
    assert sim_dirs.checkpoint_dir == checkpoint_dir / folder_name
    assert sim_dirs.data_dir == data_dir

    assert sim_dirs.output_dir.exists()
    assert sim_dirs.data_dir.exists()
    assert sim_dirs.checkpoint_dir.exists()


def test_metadata_roundtrip(tmp_path):
    sim_dirs = SimulationDirectories(
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    sim_dirs.output_dir.mkdir(parents=True, exist_ok=True)
    sim_dirs.data_dir.mkdir(parents=True, exist_ok=True)
    sim_dirs.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_simulation_metadata(simulation_dirs=sim_dirs)
    loaded_dirs = get_simulation_dirs_from_metadata(sim_dirs.output_dir)

    assert loaded_dirs.checkpoint_dir == sim_dirs.checkpoint_dir
    assert loaded_dirs.data_dir == sim_dirs.data_dir


@pytest.mark.parametrize(
    ("missing_path", "is_dir", "expected_error"),
    [
        (
            lambda sim_dirs: sim_dirs.output_dir / "runtime.yaml",
            False,
            "Missing runtime.yaml",
        ),
        (lambda sim_dirs: sim_dirs.data_dir, True, "data directory missing"),
        (
            lambda sim_dirs: sim_dirs.checkpoint_dir,
            True,
            "Checkpoint directory missing",
        ),
    ],
)
def test_load_metadata_raises_file_not_found_error(
    missing_path, is_dir, expected_error, tmp_path
):
    sim_dirs = SimulationDirectories(
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    sim_dirs.output_dir.mkdir(parents=True, exist_ok=True)
    sim_dirs.data_dir.mkdir(parents=True, exist_ok=True)
    sim_dirs.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_simulation_metadata(simulation_dirs=sim_dirs)
    missing = missing_path(sim_dirs)
    if is_dir:
        missing.rmdir()
    else:
        missing.unlink()

    with pytest.raises(FileNotFoundError, match=expected_error):
        get_simulation_dirs_from_metadata(sim_dirs.output_dir)


@pytest.mark.parametrize(
    ("loaded_dict", "expected_error"),
    [
        (
            {"data_dir": "some_dir"},
            "runtime.yaml is corrupt/missing key: 'checkpoint_dir'",
        ),
        (
            {"checkpoint_dir": "some_dir"},
            "runtime.yaml is corrupt/missing key: 'data_dir'",
        ),
        (
            {"some_other_key": "some_dir"},
            "runtime.yaml is corrupt/missing key: 'data_dir'",
        ),
    ],
)
def test_load_metadata_raises_value_error(
    loaded_dict, expected_error, mocker, tmp_path
):
    metadata_path = tmp_path / "runtime.yaml"
    metadata_path.touch()

    mocker.patch(
        "afl_sim.utils.sim_wrappers._load_dict_from_yaml", return_value=loaded_dict
    )
    with pytest.raises(ValueError, match=expected_error):
        get_simulation_dirs_from_metadata(tmp_path)


def test_config_roundtrip(tmp_path):
    config = AppConfig()
    save_effective_config(tmp_path, config)
    loaded_config = load_config_from_run_dir_with_overrides(tmp_path, timeout=None)
    assert config == loaded_config


def test_load_config_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing config"):
        load_config_from_run_dir_with_overrides(tmp_path, timeout=None)


def test_build_and_run_simulation(tmp_path, mocker, get_mock_simulation):
    sim_dirs = SimulationDirectories(
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    resume = False
    config = AppConfig()
    mock_sim = get_mock_simulation

    mock_build = mocker.patch(
        "afl_sim.utils.sim_wrappers.build_simulation", return_value=mock_sim
    )
    mock_handler = mocker.patch("afl_sim.utils.sim_wrappers.graceful_interrupt_handler")
    mock_run = mocker.patch.object(mock_sim, "run")

    manager = mocker.Mock()
    manager.attach_mock(mock_handler.return_value.__enter__, "enter_context")
    manager.attach_mock(mock_run, "run_simulation")
    manager.attach_mock(mock_handler.return_value.__exit__, "exit_context")
    expected_calls = [
        mocker.call.enter_context(),
        mocker.call.run_simulation(),
        mocker.call.exit_context(None, None, None),
    ]

    build_and_run_simulation(config=config, resume=resume, simulation_dirs=sim_dirs)

    manager.assert_has_calls(expected_calls, any_order=False)

    mock_build.assert_called_once_with(
        config=config,
        output_dir=sim_dirs.output_dir,
        data_dir=sim_dirs.data_dir,
        checkpoint_dir=sim_dirs.checkpoint_dir,
        resume=resume,
    )

    mock_handler.assert_called_once_with(mock_sim)
