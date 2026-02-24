import signal
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import typer
import yaml
from loguru import logger

from afl_sim.config import AppConfig
from afl_sim.simulation import Simulation, build_simulation


@contextmanager
def graceful_interrupt_handler(
    simulation: Simulation,
) -> Generator[None, None, None]:
    """
    Context manager that wires Ctrl+C to the simulation object's stop flag.
    """
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: FrameType | None) -> None:
        simulation.stop_requested = True

    signal.signal(signal.SIGINT, handler)

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


app = typer.Typer(pretty_exceptions_show_locals=False)

# Configure default stderr logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level="INFO",
)


def create_run_directory(base_dir: Path, tag: str | None = None) -> Path:
    """Creates a unique directory for this specific run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_hash = str(uuid.uuid4())[:6]

    folder_name = f"{timestamp}_{short_hash}"
    if tag:
        # Sanitize tag for filesystem safety
        safe_tag = tag.replace("/", "_").replace("\\", "_").replace(" ", "-")
        folder_name += f"_{safe_tag}"

    return base_dir / folder_name


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Safely loads a dictionary from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must parse to a dictionary.")
        return data


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Argument(exists=True, help="Path to YAML config.")
    ],
    output_dir: Annotated[Path, typer.Option(help="Base output directory.")] = Path(
        "outputs"
    ),
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Directory for saving input data, including datasets, data splits and simulated clocks."
        ),
    ] = Path("data"),
    checkpoint_dir: Annotated[
        Path,
        typer.Option(help="Directory for saving and loading checkpoints."),
    ] = Path("checkpoints"),
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
    Start a new federated learning simulation.

    This command loads a YAML configuration, creates a timestamped results directory,
    and initializes the simulation.
    """
    # Load & Validate Config
    try:
        config_data = load_yaml_config(config_path)
        # Apply Overrides
        if learning_rate is not None:
            config_data.setdefault("optimization", {})["learning_rate"] = learning_rate

        config = AppConfig(**config_data)

    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        raise typer.Exit(code=1) from e

    if dry_run:
        logger.success("Dry Run: Configuration Validated Successfully.")
        raise typer.Exit()

    # Setup Output Environment
    run_dir = create_run_directory(output_dir, tag)
    run_id = run_dir.name

    actual_checkpoint_dir = checkpoint_dir / run_id

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        actual_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission denied creating directories: {e}")
        raise typer.Exit(code=1) from e
    except OSError as e:
        logger.error(f"Filesystem error: {e}")
        raise typer.Exit(code=1) from e

    # Setup Logging
    log_file_id = logger.add(run_dir / "run.log", rotation="10 MB")
    logger.success(f"Output Directory Created: {run_dir}")

    try:
        # Save effective config and metadata for resuming
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, sort_keys=False)

        metadata = {
            "data_dir": str(data_dir.resolve()),
            "checkpoint_dir": str(actual_checkpoint_dir.resolve()),
            "command": " ".join(sys.argv),
            "timestamp": datetime.now().isoformat(),
        }

        with open(run_dir / "runtime.yaml", "w") as f:
            yaml.dump(metadata, f, sort_keys=False)

        # Build & Run Simulation
        logger.info("Starting Simulation...")
        simulation = build_simulation(
            config=config,
            run_dir=run_dir,
            data_dir=data_dir,
            checkpoint_dir=actual_checkpoint_dir,
            resume=False,
        )

        with graceful_interrupt_handler(simulation):
            simulation.run()
            simulation.save_shutdown_checkpoint()
            logger.success("Simulation terminated.")

    except Exception:
        logger.exception("Simulation crashed. Exiting without saving.")
        raise typer.Exit(code=1) from None

    finally:
        # logger cleanup
        logger.remove(log_file_id)


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
    sim_duration: Annotated[
        float | None,
        typer.Option(help="Set a new experiment duration in simulated time units."),
    ] = None,
) -> None:
    """
    Resume an existing simulation from folder.
    """
    # Setup Logging
    log_file_id = logger.add(output_path / "run.log", rotation="10 MB", mode="a")

    try:
        # Validate Files
        config_path = output_path / "config.yaml"
        metadata_path = output_path / "runtime.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing config.yaml in {output_path}. Cannot recover simulation specs."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing runtime.yaml in {output_path}. Cannot recover location of saved input data and checkpoints."
            )

        # Load & Patch Config
        config_data = load_yaml_config(config_path)
        if timeout is not None:
            config_data.setdefault("simulation", {})["timeout_seconds"] = timeout
            logger.info(f"New session timeout override: {timeout}s")

        if sim_duration is not None:
            config_data.setdefault("simulation", {})["duration_sim_units"] = (
                sim_duration
            )
            logger.info(
                f"Experiment duration in simulated time units changed to: {sim_duration}"
            )

        config = AppConfig(**config_data)
        logger.success(f"Configuration loaded from: {config_path.name}")

        # Load & Validate Runtime Metadata
        metadata = load_yaml_config(metadata_path)

        required_keys = ["data_dir", "checkpoint_dir"]
        for k in required_keys:
            if k not in metadata:
                raise ValueError(f"runtime.yaml is corrupt/missing key: '{k}'")

        data_dir = Path(metadata["data_dir"])
        checkpoint_dir = Path(metadata["checkpoint_dir"])

        if not data_dir.exists():
            raise FileNotFoundError(f"Original data directory missing: {data_dir}")

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory missing: {checkpoint_dir}")

        # Resume Simulation
        logger.info(f"Resuming Simulation from: {output_path}")

        simulation = build_simulation(
            config=config,
            run_dir=output_path,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            resume=True,
        )

        with graceful_interrupt_handler(simulation):
            simulation.run()
            simulation.save_shutdown_checkpoint()
            logger.success("Simulation resumed and terminated.")

    except Exception as e:
        logger.exception(f"Resume Failed: {e}. Exiting without saving.")
        raise typer.Exit(code=1) from e

    finally:
        logger.remove(log_file_id)


if __name__ == "__main__":
    app()
