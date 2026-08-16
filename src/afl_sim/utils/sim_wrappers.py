import shlex
import signal
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from afl_sim.config import AppConfig
from afl_sim.paths import SimulationDirectories
from afl_sim.simulation import Simulation, build_simulation


@contextmanager
def graceful_interrupt_handler(
    simulation: Simulation,
) -> Generator[None, None, None]:
    """
    Context manager that wires termination signals to the simulation's stop flag.

    Intercepts both SIGINT (Ctrl+C) and SIGTERM (System Kill) to prevent an
    immediate hard crash, allowing the simulation loop to finish its current
    discrete event and save a final shutdown checkpoint.

    Args:
        simulation (Simulation): The active simulation object to be gracefully halted.

    Yields:
        None: Yields control back to the enclosed block.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handler(_signum: Any, _frame: Any) -> None:
        simulation.stop_requested = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def _create_run_directory(base_dir: Path, tag: str | None = None) -> Path:
    """
    Creates a unique, timestamped directory path string for the current simulation run.

    Args:
        base_dir (Path): The parent directory where the run folder should be created.
        tag (str | None, optional): An optional string label to append to the folder name
            for easier identification. Defaults to None.

    Returns:
        Path: The fully resolved path to the newly created run directory.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    short_hash = str(uuid.uuid4())[:6]

    folder_name = f"{timestamp}_{short_hash}"
    if tag:
        safe_tag = tag.replace("/", "_").replace("\\", "_").replace(" ", "-")
        folder_name += f"_{safe_tag}"

    return base_dir / folder_name


def _load_dict_from_yaml(path: Path) -> dict[str, Any]:
    """
    Safely loads and parses a YAML file into a dictionary.

    Args:
        path (Path): The YAML file path.

    Returns:
        dict[str, Any]: The parsed data.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        PermissionError: If the application lacks read permissions for the file.
        OSError: If a system-level file operation fails.
        ValueError: If the loaded YAML file does not parse into a top-level dictionary.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must parse to a dictionary.")
        return data


def load_config_with_overrides(
    config_path: Path, learning_rate: float | None = None, timeout: float | None = None
) -> AppConfig:
    """
    Loads a YAML configuration file and optionally applies runtime overrides.

    Reads the configuration, applies the provided learning rate and timeout
    overrides if they are not None, and validates the result using AppConfig.

    Args:
        config_path (Path): Path to the YAML configuration file.
        learning_rate (float | None, optional): Override value for the optimization learning rate. Defaults to None.
        timeout (float | None, optional): Override value for the simulation timeout in seconds. Defaults to None.

    Returns:
        AppConfig: The validated application configuration object.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        PermissionError: If the application lacks read permissions for the file.
        OSError: If a system-level file operation fails.
        ValueError: If the loaded YAML file does not parse into a top-level dictionary.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    config_data = _load_dict_from_yaml(config_path)

    if learning_rate is not None:
        config_data.setdefault("optimization", {})["learning_rate"] = learning_rate
        logger.info(f"New learning rate override: {learning_rate}")

    if timeout is not None:
        config_data.setdefault("simulation", {})["timeout_seconds"] = timeout
        logger.info(f"New session timeout override: {timeout}s")

    config = AppConfig(**config_data)
    logger.success(f"Configuration loaded from: {config_path.name}")

    return config


def setup_simulation_directories(
    output_dir: Path, checkpoint_dir: Path, data_dir: Path, tag: str | None = None
) -> SimulationDirectories:
    """
    Creates and validates the necessary directories for a new simulation run.

    Generates a unique run directory based on the output directory and optional tag,
    then ensures the run, checkpoint, and data directories exist on the filesystem.

    Args:
        output_dir (Path): The base directory for generating the specific run directory.
        checkpoint_dir (Path): The base directory for saving checkpoints.
        data_dir (Path): The directory for input data and datasets.
        tag (str | None, optional): An optional label appended to the run directory name. Defaults to None.

    Returns:
        SimulationDirectories: A dataclass instance containing the resolved paths.

    Raises:
        PermissionError: If the application lacks write permissions for the directory.
        OSError: If a system-level file operation fails.
    """
    run_dir = _create_run_directory(output_dir, tag)
    run_id = run_dir.name

    actual_checkpoint_dir = checkpoint_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.success(f"Output Directory Created: {run_dir}")
    actual_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return SimulationDirectories(
        output_dir=run_dir, checkpoint_dir=actual_checkpoint_dir, data_dir=data_dir
    )


def save_effective_config(run_dir: Path, config: AppConfig) -> None:
    """
    Serializes and saves the active configuration to the run directory.

    Dumps the validated AppConfig model into a 'config.yaml' file to preserve
    the exact specifications used for the simulation.

    Args:
        run_dir (Path): The directory where the 'config.yaml' file will be saved.
        config (AppConfig): The validated application configuration to serialize.

    Raises:
        PermissionError: If the application lacks write permissions for the directory.
        OSError: If a system-level file operation fails.
        yaml.YAMLError: If the configuration data cannot be serialized to YAML.
    """
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(mode="json"), f, sort_keys=False)


def save_simulation_metadata(simulation_dirs: SimulationDirectories) -> None:
    """
    Saves runtime metadata necessary for resuming the simulation.

    Writes the resolved data directory, checkpoint directory, exact CLI command,
    and timestamp to a 'runtime.yaml' file inside the run directory.

    Args:
        simulation_dirs (SimulationDirectories): The directory paths used in the simulation.

    Raises:
        PermissionError: If the application lacks write permissions for the directory.
        OSError: If a system-level file operation fails.
        yaml.YAMLError: If the metadata dictionary cannot be serialized to YAML.
    """
    metadata = {
        "data_dir": str(simulation_dirs.data_dir.resolve()),
        "checkpoint_dir": str(simulation_dirs.checkpoint_dir.resolve()),
        "command": shlex.join(sys.argv),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    with open(simulation_dirs.output_dir / "runtime.yaml", "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, sort_keys=False)


def get_simulation_dirs_from_metadata(run_dir: Path) -> SimulationDirectories:
    """
    Recovers simulation directory paths from saved runtime metadata.

    Reads 'runtime.yaml' in the provided run directory to locate the original
    data and checkpoint directories used in a previous session.

    Args:
        run_dir (Path): The existing simulation run directory containing the metadata.

    Returns:
        SimulationDirectories: A dataclass instance containing the recovered paths.

    Raises:
        FileNotFoundError: If 'runtime.yaml', the data directory, or the checkpoint directory is missing.
        PermissionError: If the application lacks read permissions for the file.
        OSError: If a system-level file operation fails.
        ValueError: If 'runtime.yaml' is corrupt or missing required keys.
        yaml.YAMLError: If the 'runtime.yaml' file contains invalid YAML syntax.
    """
    metadata_path = run_dir / "runtime.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing runtime.yaml in {run_dir}. Cannot recover location "
            "of saved input data and checkpoints."
        )

    metadata = _load_dict_from_yaml(metadata_path)

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

    return SimulationDirectories(
        output_dir=run_dir, checkpoint_dir=checkpoint_dir, data_dir=data_dir
    )


def load_config_from_run_dir_with_overrides(
    run_dir: Path, timeout: float | None = None
) -> AppConfig:
    """
    Loads a previously saved simulation configuration from a run directory.

    Locates 'config.yaml' within the run directory and optionally overrides
    the wall-clock timeout for the resumed session.

    Args:
        run_dir (Path): The existing simulation run directory containing 'config.yaml'.
        timeout (float | None, optional): An optional override for the timeout in seconds. Defaults to None.

    Returns:
        AppConfig: The validated application configuration.

    Raises:
        FileNotFoundError: If 'config.yaml' does not exist in the specified run directory.
        PermissionError: If the application lacks read permissions for the file.
        OSError: If a system-level file operation fails.
        ValueError: If the loaded YAML file does not parse into a top-level dictionary.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    config_path = run_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config.yaml in {run_dir}. Cannot recover simulation specs."
        )

    return load_config_with_overrides(config_path=config_path, timeout=timeout)


def build_and_run_simulation(
    config: AppConfig, resume: bool, simulation_dirs: SimulationDirectories
) -> None:
    """
    Initializes and executes the simulation loop.

    Builds the simulation object using the provided configuration and directories,
    then executes its run loop within a graceful interrupt context manager.

    Args:
        config (AppConfig): The application configuration.
        resume (bool): Flag indicating whether to resume from a checkpoint (True) or start fresh (False).
        simulation_dirs (SimulationDirectories): The directories for data, checkpoints, and output.
    """
    simulation = build_simulation(
        config=config,
        output_dir=simulation_dirs.output_dir,
        data_dir=simulation_dirs.data_dir,
        checkpoint_dir=simulation_dirs.checkpoint_dir,
        resume=resume,
    )

    with graceful_interrupt_handler(simulation):
        simulation.run()
