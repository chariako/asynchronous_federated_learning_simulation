from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from afl_sim.api import resume_simulation, run_simulation
from afl_sim.config import AppConfig
from afl_sim.paths import SimulationDirectories
from afl_sim.utils.sim_wrappers import save_effective_config


@pytest.fixture
def get_simulation_dirs(tmp_path):
    return SimulationDirectories(
        output_dir=tmp_path / "run_dir",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )


@dataclass
class ConfigInputs:
    config: AppConfig
    run_input: Callable[[Path], Path | str | AppConfig]
    config_file: str | None


@pytest.fixture(
    params=[
        (AppConfig(), lambda tmp_path: AppConfig(), None),
        (AppConfig(), lambda tmp_path: tmp_path / "config.yaml", "config.yaml"),
    ]
)
def get_config_inputs(request, tmp_path):
    config, run_input, config_file = request.param

    if config_file is not None:
        save_effective_config(tmp_path, config)

    return ConfigInputs(
        config=config, run_input=run_input(tmp_path), config_file=config_file
    )


@pytest.fixture(autouse=True)
def mock_api_dependencies(mocker, get_simulation_dirs):
    return {
        "run_load": mocker.patch(
            "afl_sim.api.load_config_with_overrides", return_value=AppConfig()
        ),
        "resume_load": mocker.patch(
            "afl_sim.api.load_config_from_run_dir_with_overrides",
            return_value=AppConfig(),
        ),
        "setup_dirs": mocker.patch(
            "afl_sim.api.setup_simulation_directories", return_value=get_simulation_dirs
        ),
        "get_sim_dirs": mocker.patch(
            "afl_sim.api.get_simulation_dirs_from_metadata",
            return_value=get_simulation_dirs,
        ),
        "build_and_run": mocker.patch("afl_sim.api.build_and_run_simulation"),
        "save_config": mocker.patch("afl_sim.api.save_effective_config"),
        "save_metadata": mocker.patch("afl_sim.api.save_simulation_metadata"),
    }


def test_dry_run(get_config_inputs, capture_logs):

    run_simulation(config=get_config_inputs.run_input, dry_run=True)

    assert "Dry Run: Configuration Validated Successfully" in capture_logs.text


def test_run_invalid_config_raises_validation_error():

    with pytest.raises(ValidationError):
        run_simulation(config={"num_clients": 3, "learning_rate": 0.1})  # type: ignore[arg-type]


def test_run_invalid_directory_types_raises_validation_error(get_config_inputs):
    with pytest.raises(ValidationError):
        run_simulation(
            config=get_config_inputs.run_input,
            output_dir=123,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        run_simulation(
            config=get_config_inputs.run_input,
            checkpoint_dir=["invalid", "list", "type"],  # type: ignore[arg-type]
        )


def test_run_does_not_override_app_config():
    with pytest.raises(RuntimeError, match="override only allowed for YAML configs"):
        run_simulation(config=AppConfig(), learning_rate=0.06)


def test_run_saves_simulation_files(
    get_config_inputs,
    get_simulation_dirs,
    mock_api_dependencies,
    mocker,
):

    sim_dirs = get_simulation_dirs

    mock_log_add = mocker.patch("loguru.logger.add", return_value=99)
    mocker.patch("loguru.logger.remove")

    run_simulation(config=get_config_inputs.run_input)

    mock_log_add.assert_called_with(sim_dirs.output_dir / "run.log", rotation="10 MB")
    mock_api_dependencies["save_config"].assert_called_once_with(
        run_dir=sim_dirs.output_dir, config=get_config_inputs.config
    )
    mock_api_dependencies["save_metadata"].assert_called_once_with(
        simulation_dirs=sim_dirs
    )


def test_run_app_sets_resume_false(
    get_config_inputs,
    get_simulation_dirs,
    mock_api_dependencies,
    capture_logs,
):
    run_simulation(config=get_config_inputs.run_input)

    assert "Starting Simulation" in capture_logs.text
    assert "Simulation terminated" in capture_logs.text
    mock_api_dependencies["build_and_run"].assert_called_once_with(
        config=get_config_inputs.config,
        simulation_dirs=get_simulation_dirs,
        resume=False,
    )


@pytest.mark.parametrize(
    "learning_rate_override",
    [0.123456, None],
)
@pytest.mark.parametrize(
    "get_config_inputs",
    [
        (AppConfig(), lambda tmp_path: tmp_path / "config.yaml", "config.yaml"),
    ],
    indirect=True,
)
def test_run_allows_yaml_overrides(
    get_config_inputs, learning_rate_override, tmp_path, mock_api_dependencies
):

    run_simulation(
        config=get_config_inputs.run_input, learning_rate=learning_rate_override
    )

    mock_api_dependencies["run_load"].assert_called_once_with(
        config_path=tmp_path / "config.yaml", learning_rate=learning_rate_override
    )


@pytest.mark.parametrize(
    "learning_rate_override", [None, AppConfig().optimization.learning_rate]
)
def test_run_allows_none_or_equal_overrides_with_app_config(
    learning_rate_override, capture_logs
):
    run_simulation(
        config=AppConfig(), learning_rate=learning_rate_override, dry_run=True
    )
    assert "Dry Run: Configuration Validated Successfully" in capture_logs.text


def test_run_passes_paths(get_config_inputs, tmp_path, mock_api_dependencies):

    output_dir = tmp_path / "mock_outputs"
    data_dir = tmp_path / "mock_data"
    checkpoint_dir = tmp_path / "mock_checkpoints"
    tag = "run_1"

    run_simulation(
        config=get_config_inputs.run_input,
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        tag=tag,
    )

    mock_api_dependencies["setup_dirs"].assert_called_once_with(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        tag=tag,
    )


def test_run_cleans_up_logger(get_config_inputs, mocker):
    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    run_simulation(config=get_config_inputs.run_input)

    mock_log_remove.assert_called_with(99)


def test_run_cleans_up_logger_on_exception(
    get_config_inputs, mock_api_dependencies, mocker
):

    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    mock_api_dependencies["build_and_run"].side_effect = Exception("Crash")

    with pytest.raises(Exception, match="Crash"):
        run_simulation(config=get_config_inputs.run_input)

    mock_log_remove.assert_called_with(99)


def test_run_accepts_strings_and_coerces_to_paths(
    get_config_inputs, mock_api_dependencies
):
    output_str = "mock_outputs"
    data_str = "mock_data"
    checkpoint_str = "mock_checkpoints"
    tag = "run_1"

    run_simulation(
        config=get_config_inputs.run_input,
        output_dir=output_str,
        data_dir=data_str,
        checkpoint_dir=checkpoint_str,
        tag=tag,
    )

    mock_api_dependencies["setup_dirs"].assert_called_once_with(
        output_dir=Path(output_str),
        checkpoint_dir=Path(checkpoint_str),
        data_dir=Path(data_str),
        tag=tag,
    )


def test_resume_app_sets_resume_true(
    tmp_path, get_simulation_dirs, mock_api_dependencies, capture_logs
):
    resume_simulation(output_path=tmp_path)

    assert "Resuming Simulation" in capture_logs.text
    assert "Simulation resumed and terminated" in capture_logs.text
    mock_api_dependencies["build_and_run"].assert_called_once_with(
        config=AppConfig(), simulation_dirs=get_simulation_dirs, resume=True
    )


@pytest.mark.parametrize(
    "timeout_override",
    [123.456, None],
)
def test_resume_passes_overrides(timeout_override, tmp_path, mock_api_dependencies):
    resume_simulation(output_path=tmp_path, timeout=timeout_override)

    mock_api_dependencies["resume_load"].assert_called_once_with(
        run_dir=tmp_path, timeout=timeout_override
    )


def test_resume_app_resumes_cleans_up_logger(tmp_path, mocker):
    mock_log_add = mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    resume_simulation(output_path=tmp_path)
    mock_log_add.assert_called_with(tmp_path / "run.log", rotation="10 MB", mode="a")
    mock_log_remove.assert_called_with(99)


def test_resume_cleans_up_logger_on_exception(tmp_path, mocker, mock_api_dependencies):
    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    mock_api_dependencies["build_and_run"].side_effect = Exception("Crash")

    with pytest.raises(Exception, match="Crash"):
        resume_simulation(output_path=tmp_path)

    mock_log_remove.assert_called_with(99)


def test_resume_invalid_output_path_raises_validation_error():
    with pytest.raises(ValidationError):
        resume_simulation(output_path={"wrong": "type"})  # type: ignore[arg-type]


def test_resume_accepts_strings_and_coerces_to_paths(mock_api_dependencies, mocker):
    mocker.patch("loguru.logger.add", return_value=99)
    mocker.patch("loguru.logger.remove")

    output_str = "mock_outputs_resume"

    resume_simulation(output_path=output_str)

    mock_api_dependencies["resume_load"].assert_called_once_with(
        run_dir=Path(output_str), timeout=None
    )
