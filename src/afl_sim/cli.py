import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError
from yaml import YAMLError

from afl_sim.api import resume_simulation, run_simulation
from afl_sim.enums import DefaultDirs


def logger_setup() -> None:
    """
    Configures the default application logging to standard error.

    Removes any existing default Loguru handlers and establishes a new
    stderr handler with a custom timestamped format and an INFO logging level.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )


app = typer.Typer(pretty_exceptions_show_locals=True)


@app.callback()
def main() -> None:
    """
    Federated Learning Simulation CLI.

    Provides commands to configure, run, and resume discrete-event
    federated learning simulations.
    """
    logger_setup()


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Argument(exists=True, help="Path to YAML config.")
    ],
    output_dir: Annotated[Path, typer.Option(help="Base output directory.")] = Path(
        DefaultDirs.OUTPUTS
    ),
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Directory for saving input data, including datasets, data splits and simulated clocks."
        ),
    ] = Path(DefaultDirs.DATA),
    checkpoint_dir: Annotated[
        Path,
        typer.Option(help="Directory for saving and loading checkpoints."),
    ] = Path(DefaultDirs.CHECKPOINTS),
    learning_rate: Annotated[
        float | None, typer.Option("--lr", help="Override client learning rate.")
    ] = None,
    tag: Annotated[
        str | None, typer.Option(help="Optional label for this run (e.g. 'baseline')")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config and exit without running."),
    ] = False,
) -> None:
    """
    Starts a new federated learning simulation.

    This command loads a YAML configuration, creates a timestamped results directory,
    initializes the data partitions and simulation environment, and begins the run.

    Args:
        config_path (Path): Path to the YAML configuration file.
        output_dir (Path): Base directory for all output runs.
        data_dir (Path): Directory for saving/loading datasets, splits, and clocks.
        checkpoint_dir (Path): Base directory for saving checkpoints.
        learning_rate (float | None): Optional override for the client learning rate.
        tag (str | None): Optional label appended to the run directory name.
        dry_run (bool): If True, validates the config and exits without starting.

    Raises:
        typer.Exit: Exits with code 1 if configuration validation, filesystem operations,
            or the simulation run fails. Exits with code 0 on a successful dry run.
    """
    try:
        run_simulation(
            config=config_path,
            output_dir=output_dir,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            learning_rate=learning_rate,
            tag=tag,
            dry_run=dry_run,
        )

    except (FileNotFoundError, ValueError, YAMLError, ValidationError) as e:
        logger.error(f"Configuration error: {e}")
        raise typer.Exit(code=1) from e

    except (PermissionError, OSError) as e:
        logger.error(f"Filesystem error: {e}")
        raise typer.Exit(code=1) from e

    except Exception as e:
        logger.exception("Simulation crashed. Exiting without saving.")
        raise typer.Exit(code=1) from e


@app.command()
def resume(
    output_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Path to the output directory (e.g. 'outputs/2026...') containing config.yaml.",
        ),
    ],
    timeout: Annotated[
        float | None,
        typer.Option(
            help="Override the wall-clock timeout (in seconds) for this specific resume session."
        ),
    ] = None,
) -> None:
    """
    Resumes an existing simulation from a previously saved output directory.

    Restores the configuration, locates the appropriate datasets and checkpoints
    from the runtime metadata, and continues the simulation loop from the exact
    global index where it last stopped.

    Args:
        output_path (Path): Path to the existing run directory containing `config.yaml`.
        timeout (float | None): Optional override for the wall-clock timeout in seconds
            for this specific session.

    Raises:
        typer.Exit: Exits with code 1 if configuration/metadata validation, filesystem operations,
            or the simulation run fails.
    """
    try:
        resume_simulation(output_path=output_path, timeout=timeout)

    except (FileNotFoundError, ValueError, YAMLError, ValidationError) as e:
        logger.error(f"Cannot resume simulation (Configuration or metadata error): {e}")
        raise typer.Exit(code=1) from e

    except (PermissionError, OSError) as e:
        logger.error(f"Cannot resume simulation (Filesystem error): {e}")
        raise typer.Exit(code=1) from e

    except Exception as e:
        logger.exception("Resume Failed. Exiting without saving.")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":  # pragma: no cover
    app()
