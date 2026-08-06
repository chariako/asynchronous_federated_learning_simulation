import pytest
from typer.testing import CliRunner

from afl_sim.cli import app
from afl_sim.cli_helpers import save_effective_config
from afl_sim.config import AppConfig
from afl_sim.paths import SimulationDirectories

runner = CliRunner()


@pytest.fixture
def get_simulation_dirs(tmp_path):
    return SimulationDirectories(
        run_dir=tmp_path / "run_dir",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )


@pytest.fixture(autouse=True)
def mock_cli_dependencies(mocker, get_simulation_dirs):
    return {
        "run_load": mocker.patch(
            "afl_sim.cli.load_config_with_overrides", return_value=AppConfig()
        ),
        "resume_load": mocker.patch(
            "afl_sim.cli.load_effective_config_from_run_dir_with_overrides",
            return_value=AppConfig(),
        ),
        "setup_dirs": mocker.patch(
            "afl_sim.cli.setup_simulation_directories", return_value=get_simulation_dirs
        ),
        "get_sim_dirs": mocker.patch(
            "afl_sim.cli.get_simulation_dirs_from_metadata",
            return_value=get_simulation_dirs,
        ),
        "build_and_run": mocker.patch("afl_sim.cli.build_and_run_simulation"),
        "save_config": mocker.patch("afl_sim.cli.save_effective_config"),
        "save_metadata": mocker.patch("afl_sim.cli.save_simulation_metadata"),
    }


def test_help_menu():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_dry_run(tmp_path):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry Run: Configuration Validated Successfully" in result.output


def test_run_saves_simulation_files(
    tmp_path, get_simulation_dirs, mock_cli_dependencies, mocker
):
    config = AppConfig()
    save_effective_config(tmp_path, config)
    sim_dirs = get_simulation_dirs

    mock_log_add = mocker.patch("loguru.logger.add", return_value=99)
    mocker.patch("loguru.logger.remove")

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 0
    mock_log_add.assert_called_with(sim_dirs.run_dir / "run.log", rotation="10 MB")
    mock_cli_dependencies["save_config"].assert_called_once_with(
        run_dir=sim_dirs.run_dir, config=config
    )
    mock_cli_dependencies["save_metadata"].assert_called_once_with(
        simulation_dirs=sim_dirs
    )


def test_run_app_sets_resume_false(
    tmp_path, get_simulation_dirs, mock_cli_dependencies
):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 0
    assert "Starting Simulation" in result.output
    assert "Simulation terminated" in result.output
    mock_cli_dependencies["build_and_run"].assert_called_once_with(
        config=config, simulation_dirs=get_simulation_dirs, resume=False
    )


@pytest.mark.parametrize(
    ("learning_rate_override", "append_command"),
    [(0.123456, ["--lr", "0.123456"]), (None, [])],
)
def test_run_passes_overrides(
    learning_rate_override, append_command, tmp_path, mock_cli_dependencies
):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml", *append_command])

    mock_cli_dependencies["run_load"].assert_called_once_with(
        config_path=tmp_path / "config.yaml", learning_rate=learning_rate_override
    )
    assert result.exit_code == 0


def test_run_passes_paths(tmp_path, mock_cli_dependencies):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoints"
    tag = "run_1"

    result = runner.invoke(
        app,
        [
            "run",
            f"{tmp_path}/config.yaml",
            "--output-dir",
            f"{output_dir}",
            "--data-dir",
            f"{data_dir}",
            "--checkpoint-dir",
            f"{checkpoint_dir}",
            "--tag",
            tag,
        ],
    )

    mock_cli_dependencies["setup_dirs"].assert_called_once_with(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        tag=tag,
    )
    assert result.exit_code == 0


def test_run_cleans_up_logger(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 0
    mock_log_remove.assert_called_with(99)


def test_run_cleans_up_logger_on_exception(tmp_path, mock_cli_dependencies, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    mock_cli_dependencies["build_and_run"].side_effect = Exception("Crash")

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    mock_log_remove.assert_called_with(99)


def test_run_raises_config_error(tmp_path, mock_cli_dependencies):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mock_cli_dependencies["run_load"].side_effect = ValueError

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_run_raises_filesystem_error(tmp_path, mock_cli_dependencies):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mock_cli_dependencies["run_load"].side_effect = PermissionError

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Filesystem error" in result.output


def test_run_catches_exceptions(tmp_path, mock_cli_dependencies):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mock_cli_dependencies["run_load"].side_effect = Exception

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Simulation crashed" in result.output


def test_resume_raises_config_error(tmp_path, mock_cli_dependencies):
    mock_cli_dependencies["resume_load"].side_effect = ValueError

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Cannot resume simulation (Configuration or metadata error)" in result.output


def test_resume_raises_filesystem_error(tmp_path, mock_cli_dependencies):
    mock_cli_dependencies["resume_load"].side_effect = PermissionError

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Cannot resume simulation (Filesystem error)" in result.output


def test_resume_catches_exceptions(tmp_path, mock_cli_dependencies):
    mock_cli_dependencies["resume_load"].side_effect = Exception

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Resume Failed" in result.output


def test_resume_app_sets_resume_true(
    tmp_path, get_simulation_dirs, mock_cli_dependencies
):
    config = AppConfig()

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 0
    assert "Resuming Simulation" in result.output
    assert "Simulation resumed and terminated" in result.output
    mock_cli_dependencies["build_and_run"].assert_called_once_with(
        config=config, simulation_dirs=get_simulation_dirs, resume=True
    )


@pytest.mark.parametrize(
    ("timeout_override", "append_command"),
    [(123.456, ["--timeout", "123.456"]), (None, [])],
)
def test_resume_passes_overrides(
    timeout_override, append_command, tmp_path, mock_cli_dependencies
):
    result = runner.invoke(app, ["resume", f"{tmp_path}", *append_command])

    mock_cli_dependencies["resume_load"].assert_called_once_with(
        run_dir=tmp_path, timeout=timeout_override
    )
    assert result.exit_code == 0


def test_resume_app_resumes_cleans_up_logger(tmp_path, mocker):
    mock_log_add = mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    mock_log_add.assert_called_with(tmp_path / "run.log", rotation="10 MB", mode="a")
    mock_log_remove.assert_called_with(99)
    assert result.exit_code == 0


def test_resume_cleans_up_logger_on_exception(tmp_path, mocker, mock_cli_dependencies):
    mocker.patch("loguru.logger.add", return_value=99)
    mock_log_remove = mocker.patch("loguru.logger.remove")

    mock_cli_dependencies["build_and_run"].side_effect = Exception("Crash")

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    mock_log_remove.assert_called_with(99)
