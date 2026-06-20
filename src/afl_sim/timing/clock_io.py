import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from afl_sim.types import PathCollection
from afl_sim.utils import save_clock_plot

from .clock_types import ClockConfig, ClockData, ClockGenerators, ClockStates


def save_clock_and_visualize(
    clock_data: ClockData,
    clock_generators: ClockGenerators,
    chunk_num: int,
    paths: PathCollection,
    clock_config: ClockConfig,
    visualize: bool,
) -> None:
    """
    Orchestrates the saving of clock data and the generation of its visualization.

    Delegates the data serialization and plotting tasks to specialized internal functions
    to ensure the clock chunk and its visual representation are safely written to disk.

    Args:
        clock_data (ClockData): A foundational container for raw simulation clock events to save.
        clock_generators (ClockGenerators): A centralized container whose random states need preserving.
        chunk_num (int): The sequence number of the chunk being saved.
        paths (PathCollection): A collection of standardized file paths for simulation data I/O.
        clock_config (ClockConfig): A structured dictionary defining clock generation parameters.
        visualize (bool): Flag indicating whether to render and save the plot.
    """
    _save_clock_chunk(
        clock_data=clock_data,
        clock_generators=clock_generators,
        chunk_path=paths.get_clock_chunk_path(chunk_num=chunk_num),
    )
    _save_clock_visualization(
        clock=clock_data, paths=paths, config_dict=clock_config, visualize=visualize
    )


def _save_clock_chunk(
    clock_data: ClockData,
    clock_generators: ClockGenerators,
    chunk_path: Path,
) -> None:
    """
    Serializes and saves a single chunk of clock data alongside generator states.

    Extracts the current internal states of the random number generators and saves them
    as compressed numpy archives (`.npz`) along with the generated timestamps and client IDs,
    ensuring exact reproducibility upon resumption.

    Args:
        clock_data (ClockData): A foundational container for raw simulation clock events to save.
        clock_generators (ClockGenerators): A centralized container whose random states need preserving.
        chunk_path (Path): The designated filesystem path for the output `.npz` file.
    """
    # Extract the current states of the clock generators
    delay_state = clock_generators.delay_state
    select_state = clock_generators.select_state

    # Convert to string
    delay_state_str = json.dumps(delay_state)
    select_state_str = json.dumps(select_state)

    np.savez_compressed(
        chunk_path,
        timestamps=clock_data.timestamps,
        client_ids=clock_data.client_ids,
        delay_state=delay_state_str,
        select_state=select_state_str,
        allow_pickle=False,
    )


def load_clock_data(chunk_path: Path) -> ClockData:
    """
    Loads raw clock event data from a compressed numpy archive on disk.

    Args:
        chunk_path (Path): The filesystem path pointing to the saved chunk data.

    Returns:
        ClockData: A foundational container for raw simulation clock events (timestamps and client IDs).
    """
    logger.info(f"Loading clock data from: {chunk_path}")
    with np.load(chunk_path) as data:
        clock_data = ClockData(
            timestamps=data["timestamps"],
            client_ids=data["client_ids"],
        )
    return clock_data


def load_clock_generator_states(
    chunk_path: Path,
) -> ClockStates:
    """
    Reconstructs the random number generator states from a saved clock chunk.

    Reads the serialized JSON states and bounding timestamps from the archive to allow
    the simulation generators to resume exactly where they left off in the previous chunk.

    Args:
        chunk_path (Path): The filesystem path pointing to the saved chunk data.

    Returns:
        ClockStates: A snapshot of internal random generator states and temporal boundaries.
    """
    logger.info(f"Loading clock states from: {chunk_path}")
    with np.load(chunk_path) as data:
        delay_state_str = data["delay_state"]
        select_state_str = data["select_state"]
        start_time = data["timestamps"][0]
        end_time = data["timestamps"][-1]

    return ClockStates(
        delay_state=json.loads(delay_state_str.item()),
        select_state=json.loads(select_state_str.item()),
        start_time=start_time,
        end_time=end_time,
    )


def save_metadata(metadata: dict[str, Any], meta_path: Path) -> None:
    """
    Saves the clock's configuration metadata to a JSON file.

    Args:
        metadata (dict[str, Any]): A dictionary containing config parameters and hashes.
        meta_path (Path): The target filesystem path for the metadata JSON file.
    """
    logger.info(f"Saving clock metadata to: {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _save_clock_visualization(
    clock: ClockData,
    paths: PathCollection,
    config_dict: ClockConfig,
    visualize: bool,
) -> None:
    """
    Generates and saves a visual plot of the clock data if requested.

    Formats the timestamps and client IDs appropriately based on whether the simulation
    is synchronous or asynchronous, and utilizes the internal plotting utility to write
    the visualization to disk. Fails gracefully if plotting errors occur.

    Args:
        clock (ClockData): A foundational container for raw simulation clock events to visualize.
        paths (PathCollection): A collection of standardized file paths for simulation data I/O.
        config_dict (ClockConfig): A structured dictionary defining clock generation parameters.
        visualize (bool): Flag to toggle the visualization execution.
    """
    # Visualization
    if visualize and not paths.plot_path.exists():
        try:
            # Extract plot data
            timestamps = clock.timestamps
            client_ids = clock.client_ids

            # Prepare sync data for visualization
            if config_dict["sample_size"]:
                timestamps = np.repeat(timestamps, config_dict["sample_size"])
                client_ids = client_ids.flatten().astype(int)

            logger.info(f"Saving clock visualization to: {paths.plot_path}")

            save_clock_plot(
                timestamps=timestamps,
                client_ids=client_ids,
                num_clients=config_dict["num_clients"],
                filepath=paths.plot_path,
            )
        except Exception as e:
            logger.warning(f"Skipping clock visualization due to error: {e}")
