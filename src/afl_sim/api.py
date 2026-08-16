from pathlib import Path

from loguru import logger
from pydantic import validate_call

from afl_sim.config import AppConfig
from afl_sim.enums import DefaultDirs
from afl_sim.utils.sim_wrappers import (
    build_and_run_simulation,
    get_simulation_dirs_from_metadata,
    load_config_from_run_dir_with_overrides,
    load_config_with_overrides,
    save_effective_config,
    save_simulation_metadata,
    setup_simulation_directories,
)


@validate_call
def run_simulation(
    config: Path | str | AppConfig,
    output_dir: Path | str = DefaultDirs.OUTPUTS,
    data_dir: Path | str = DefaultDirs.DATA,
    checkpoint_dir: Path | str = DefaultDirs.CHECKPOINTS,
    learning_rate: float | None = None,
    tag: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Orchestrates and starts a new federated learning simulation.

    This function accepts either a path to a YAML configuration file or a pre-instantiated
    AppConfig object. It creates a timestamped results directory, initializes the data
    partitions and simulation environment, and begins the run.

    Args:
        config (Path | str | AppConfig): Path to the YAML configuration file, or an AppConfig instance.
        output_dir (Path | str, optional): Base directory for all output runs.
        data_dir (Path | str, optional): Directory for saving/loading datasets, splits, and clocks.
        checkpoint_dir (Path | str, optional): Base directory for saving checkpoints.
        learning_rate (float | None, optional): Optional override for the YAML client learning rate.
        tag (str | None, optional): Optional label appended to the run directory name.
        dry_run (bool, optional): If True, validates the config and exits without starting.

    Raises:
        RuntimeError: If a YAML learning rate override is requested without a YAML path.
        TypeError: If the provided config is not a Path, string, or AppConfig.
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If there is an invalid parameter value.
        YAMLError: If the YAML configuration is malformed.
        ValidationError: If the configuration fails Pydantic validation.
        PermissionError: If there are insufficient permissions to create directories.
        OSError: If a general filesystem error occurs.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)

    if isinstance(config, (Path, str)):
        resolved_config = load_config_with_overrides(
            config_path=Path(config),
            learning_rate=learning_rate,
        )
    else:
        resolved_config = config
        if (
            learning_rate is not None
            and learning_rate != config.optimization.learning_rate
        ):
            raise RuntimeError(
                "Config Error: Learning rate override only allowed "
                "for YAML configs. To modify the learning rate, "
                "edit the AppConfig object directly and re-run."
            )

    if dry_run:
        logger.success("Dry Run: Configuration Validated Successfully.")
        return

    simulation_dirs = setup_simulation_directories(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        tag=tag,
    )

    log_file_id = None
    try:
        log_file_id = logger.add(
            simulation_dirs.output_dir / "run.log", rotation="10 MB"
        )

        save_effective_config(
            run_dir=simulation_dirs.output_dir, config=resolved_config
        )
        save_simulation_metadata(simulation_dirs=simulation_dirs)

        logger.info("Starting Simulation...")
        build_and_run_simulation(
            config=resolved_config, simulation_dirs=simulation_dirs, resume=False
        )
        logger.success("Simulation terminated.")

    finally:
        if log_file_id is not None:  # pragma: no branch
            logger.remove(log_file_id)


@validate_call
def resume_simulation(
    output_path: Path | str,
    timeout: float | None = None,
) -> None:
    """
    Resumes an existing simulation from a previously saved output directory.

    Restores the configuration, locates the appropriate datasets and checkpoints
    from the runtime metadata, and continues the simulation loop from the exact
    global index where it last stopped.

    Args:
        output_path (Path): Path to the existing run directory containing `config.yaml`.
        timeout (float | None, optional): Optional override for the wall-clock timeout in seconds
            for this specific session.

    Raises:
        FileNotFoundError: If the configuration or metadata files cannot be found.
        ValueError: If there is an invalid parameter value.
        YAMLError: If the YAML configuration is malformed.
        ValidationError: If the configuration fails Pydantic validation.
        PermissionError: If there are insufficient permissions to read/write directories.
        OSError: If a general filesystem error occurs.
    """
    output_path = Path(output_path)

    log_file_id = None
    try:
        log_file_id = logger.add(output_path / "run.log", rotation="10 MB", mode="a")

        config = load_config_from_run_dir_with_overrides(
            run_dir=output_path, timeout=timeout
        )
        simulation_dirs = get_simulation_dirs_from_metadata(run_dir=output_path)

        logger.info(f"Resuming Simulation from: {output_path}")

        build_and_run_simulation(
            config=config, simulation_dirs=simulation_dirs, resume=True
        )

        logger.success("Simulation resumed and terminated.")

    finally:
        if log_file_id is not None:  # pragma: no branch
            logger.remove(log_file_id)
