import pytest
from typer.testing import CliRunner

from afl_sim.cli import app
from afl_sim.config import AppConfig
from afl_sim.paths import SimulationDirectories
from afl_sim.utils.sim_wrappers import save_effective_config

runner = CliRunner()


@pytest.fixture
def get_simulation_dirs(tmp_path):
    return SimulationDirectories(
        output_dir=tmp_path / "run_dir",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_help_menu():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_dry_run(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mock_run = mocker.patch("afl_sim.cli.run_simulation")

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml", "--dry-run"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        config=tmp_path / "config.yaml",
        output_dir=mocker.ANY,
        data_dir=mocker.ANY,
        checkpoint_dir=mocker.ANY,
        learning_rate=mocker.ANY,
        tag=mocker.ANY,
        dry_run=True,
    )


@pytest.mark.parametrize(
    ("learning_rate_override", "append_command"),
    [(0.123456, ["--lr", "0.123456"]), (None, [])],
)
def test_run_passes_overrides(learning_rate_override, append_command, tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mock_run = mocker.patch("afl_sim.cli.run_simulation")

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml", *append_command])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        config=tmp_path / "config.yaml",
        output_dir=mocker.ANY,
        data_dir=mocker.ANY,
        checkpoint_dir=mocker.ANY,
        learning_rate=learning_rate_override,
        tag=mocker.ANY,
        dry_run=False,
    )


def test_run_passes_paths(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    output_dir = tmp_path / "mock_outputs"
    data_dir = tmp_path / "mock_data"
    checkpoint_dir = tmp_path / "mock_checkpoints"
    tag = "run_1"

    mock_run = mocker.patch("afl_sim.cli.run_simulation")

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

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        config=tmp_path / "config.yaml",
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        learning_rate=mocker.ANY,
        tag=tag,
        dry_run=False,
    )


def test_run_raises_config_error(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mocker.patch("afl_sim.cli.run_simulation", side_effect=ValueError)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Configuration error" in result.output


def test_run_raises_filesystem_error(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mocker.patch("afl_sim.cli.run_simulation", side_effect=PermissionError)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Filesystem error" in result.output


def test_run_catches_exceptions(tmp_path, mocker):
    config = AppConfig()
    save_effective_config(tmp_path, config)

    mocker.patch("afl_sim.cli.run_simulation", side_effect=Exception)

    result = runner.invoke(app, ["run", f"{tmp_path}/config.yaml"])

    assert result.exit_code == 1
    assert "Simulation crashed" in result.output


def test_resume_raises_config_error(tmp_path, mocker):

    mocker.patch("afl_sim.cli.resume_simulation", side_effect=ValueError)

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Cannot resume simulation (Configuration or metadata error)" in result.output


def test_resume_raises_filesystem_error(tmp_path, mocker):
    mocker.patch("afl_sim.cli.resume_simulation", side_effect=PermissionError)

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Cannot resume simulation (Filesystem error)" in result.output


def test_resume_catches_exceptions(tmp_path, mocker):
    mocker.patch("afl_sim.cli.resume_simulation", side_effect=Exception)

    result = runner.invoke(app, ["resume", f"{tmp_path}"])

    assert result.exit_code == 1
    assert "Resume Failed" in result.output


@pytest.mark.parametrize(
    ("timeout_override", "append_command"),
    [(123.456, ["--timeout", "123.456"]), (None, [])],
)
def test_resume_passes_overrides(timeout_override, append_command, tmp_path, mocker):
    mock_resume = mocker.patch("afl_sim.cli.resume_simulation")

    result = runner.invoke(app, ["resume", f"{tmp_path}", *append_command])

    mock_resume.assert_called_once_with(output_path=tmp_path, timeout=timeout_override)
    assert result.exit_code == 0
