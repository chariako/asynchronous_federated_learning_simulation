from pathlib import Path

from loguru import logger

from afl_sim.config import AppConfig
from afl_sim.paths import ClockPathCollection
from afl_sim.utils import compute_hash_from_dict

from .clock_constructors import gen_clock_chunk_from_scratch
from .clock_io import (
    load_clock_data,
    load_clock_generator_states,
    save_clock_and_visualize,
    save_metadata,
)
from .clock_types import ClockConfig, ClockData, ClockGenerators, SimulationClock
from .clock_utils import (
    clock_merger,
    clock_slicer,
    extract_clock_config,
    get_clock_generators,
    package_simulation_clock,
)

_EVENTS_PER_CHUNK = 3000
_CHUNK_GEN_THRESHOLD = (
    1000  # number of leftover events in previous chunk triggering new chunk generation
)


def get_clock(
    config: AppConfig, data_dir: Path, global_next_idx: int
) -> SimulationClock:
    """
    Retrieves or generates a simulation clock based on the provided configuration and index.

    Handles the initialization or resumption of clock states by checking the global index.
    If resuming, it fetches the appropriate current (and potentially next) data chunks,
    merging or slicing them to maintain continuity. It safely manages metadata storage
    and output directory creation.

    Args:
        config (AppConfig): The main application configuration object.
        data_dir (Path): The base directory path for storing and retrieving clock data.
        global_next_idx (int): The global index indicating the next simulation event to process.

    Returns:
        SimulationClock: A packaged simulation clock object ready for use in the environment.
    """
    # Do not visualize if simualtion is resumed
    visualize = not global_next_idx and config.visualization.visualize_client_arrivals

    # Create unique clock hash string
    clock_config = extract_clock_config(config)
    config_hash = compute_hash_from_dict(clock_config)

    # Create output dirs
    output_dir = data_dir / "clocks" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ClockPathCollection.from_clock_specs(
        data_dir=output_dir,
        hash_str=config_hash,
    )

    # Always save/refresh clock metadata
    metadata = {
        "config": clock_config,
        "config_hash": config_hash,
        "events_per_chunk": _EVENTS_PER_CHUNK,
    }
    save_metadata(metadata, paths.meta_path)

    # Get decoupled random streams for rate and clock generation
    clock_generators = get_clock_generators(
        num_clients=clock_config["num_clients"],
        sigma_rate=clock_config["sigma"],
        seed=clock_config["seed"],
    )

    if global_next_idx:
        # Locate resume chunk
        current_chunk = global_next_idx // _EVENTS_PER_CHUNK
        logger.info(
            f"Attempting to fetch current clock chunk with chunk_num: {current_chunk}..."
        )
        current_clock_data = _fetch_or_generate_chunk(
            config=clock_config,
            chunk_num=current_chunk,
            clock_generators=clock_generators,
            paths=paths,
            visualize=visualize,
        )
        next_clock_data = None

        # Check if the next chunk is needed
        local_index = global_next_idx % _EVENTS_PER_CHUNK
        leftover_events = _EVENTS_PER_CHUNK - local_index

        if leftover_events < _CHUNK_GEN_THRESHOLD:
            logger.info(
                f"Less than {_CHUNK_GEN_THRESHOLD} ({leftover_events}) events remaining "
                f"in chunk {current_chunk}. Attempting to fetch next chunk..."
            )
            next_clock_data = _fetch_or_generate_chunk(
                config=clock_config,
                chunk_num=current_chunk + 1,
                clock_generators=clock_generators,
                paths=paths,
                visualize=visualize,
            )

        if next_clock_data is not None:
            merged_data = clock_merger(
                current_clock_chunk=current_clock_data,
                local_next_idx=local_index,
                next_clock_chunk=next_clock_data,
                events_per_chunk=_EVENTS_PER_CHUNK,
            )
            return package_simulation_clock(
                clock_data=merged_data,
                global_idx=global_next_idx,
            )

        sliced_clock = clock_slicer(
            current_clock_chunk=current_clock_data,
            local_next_idx=local_index,
            events_per_chunk=_EVENTS_PER_CHUNK,
        )
        return package_simulation_clock(
            clock_data=sliced_clock,
            global_idx=global_next_idx,
        )

    logger.info("Attempting to fetch clock chunk 0...")

    chunk_0 = _fetch_or_generate_chunk(
        config=clock_config,
        chunk_num=0,
        clock_generators=clock_generators,
        paths=paths,
        visualize=visualize,
    )

    return package_simulation_clock(
        clock_data=chunk_0,
        global_idx=0,
    )


def _fetch_or_generate_chunk(
    config: ClockConfig,
    chunk_num: int,
    clock_generators: ClockGenerators,
    paths: ClockPathCollection,
    visualize: bool,
) -> ClockData:
    """
    Retrieves a specific clock data chunk from disk or triggers its generation if missing.

    Checks the filesystem for the requested chunk. If it does not exist, it initiates a
    recursive generation process to build out the required data chronologically before
    loading and returning it.

    Args:
        config (ClockConfig): A structured dictionary defining clock generation parameters.
        chunk_num (int): The sequential index of the chunk to fetch or generate.
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        paths (ClockPathCollection): A collection of standardized file paths for simulation data I/O.
        visualize (bool): Flag indicating whether to generate visual plots of the data.

    Returns:
        ClockData: A foundational container for the raw simulation clock events of the specified chunk.
    """
    chunk_path = paths.get_clock_chunk_path(chunk_num)

    if not chunk_path.exists():
        logger.info(
            f"Existing data for chunk {chunk_num} not found. Generating new chunk..."
        )
        _recursive_chunk_generation(
            config=config,
            chunk_num=chunk_num,
            clock_generators=clock_generators,
            paths=paths,
            visualize=visualize,
        )

    return load_clock_data(chunk_path=chunk_path)


def _recursive_chunk_generation(
    config: ClockConfig,
    chunk_num: int,
    clock_generators: ClockGenerators,
    paths: ClockPathCollection,
    visualize: bool,
) -> None:
    """
    Recursively generates missing clock data chunks up to the targeted chunk number.

    Ensures chronological continuity is maintained by generating chunks sequentially
    from the last available state. For subsequent chunks, it loads the previous chunk's
    end state to accurately seed the generators for the current chunk.

    Args:
        config (ClockConfig): A structured dictionary defining clock generation parameters.
        chunk_num (int): The target chunk number to generate.
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        paths (ClockPathCollection): A collection of standardized file paths for simulation data I/O.
        visualize (bool): Flag indicating whether to generate visual plots of the data.
    """
    # Base case: chunk_num = 0
    if chunk_num == 0:
        logger.info("Generating clock data for base chunk 0...")
        _generate_chunk_and_save(
            config=config,
            start_time=0.0,
            chunk_num=0,
            clock_generators=clock_generators,
            paths=paths,
            visualize=visualize,
        )
        return

    # Previous chunk check and recursive generation
    prev_chunk_path = paths.get_clock_chunk_path(chunk_num=chunk_num - 1)

    if not prev_chunk_path.exists():
        logger.warning(f"Clock data for chunk {chunk_num - 1} missing. Regenerating...")
        _recursive_chunk_generation(
            config=config,
            chunk_num=chunk_num - 1,
            clock_generators=clock_generators,
            paths=paths,
            visualize=visualize,
        )

    # Extract previous clock states and update clock generator
    prev_chunk_states = load_clock_generator_states(chunk_path=prev_chunk_path)
    clock_generators.update_states(states=prev_chunk_states)

    logger.info(f"Generating clock data for chunk {chunk_num}...")
    _generate_chunk_and_save(
        config=config,
        start_time=prev_chunk_states.end_time,
        chunk_num=chunk_num,
        clock_generators=clock_generators,
        paths=paths,
        visualize=visualize,
    )


def _generate_chunk_and_save(
    config: ClockConfig,
    start_time: float,
    chunk_num: int,
    clock_generators: ClockGenerators,
    paths: ClockPathCollection,
    visualize: bool,
) -> None:
    """
    Generates a single block of clock events from scratch and persists it to disk.

    Utilizes the provided clock generators and a starting timestamp to calculate a new
    batch of clock events. It handles saving the chunk to the filesystem, preserving
    generator states, and executing visualization if configured.

    Args:
        config (ClockConfig): A structured dictionary defining clock generation parameters.
        start_time (float): The simulation timestamp at which this specific chunk begins.
        chunk_num (int): The sequential index assigned to this newly generated chunk.
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        paths (ClockPathCollection): A collection of standardized file paths for simulation data I/O.
        visualize (bool): Flag indicating whether to generate visual plots of the data.
    """
    # Generate and save clock data
    clock_data = gen_clock_chunk_from_scratch(
        config=config,
        clock_generators=clock_generators,
        start_time=start_time,
        event_num=_EVENTS_PER_CHUNK,
    )

    save_clock_and_visualize(
        clock_data=clock_data,
        clock_generators=clock_generators,
        chunk_num=chunk_num,
        paths=paths,
        clock_config=config,
        visualize=visualize,
    )

    logger.success(
        f"New clock data generated and saved for chunk {chunk_num} "
        f"with start_time: {start_time:.3f}"
    )
