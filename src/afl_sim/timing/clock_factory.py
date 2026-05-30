from pathlib import Path

from loguru import logger
from numpy.random import Generator, SeedSequence, default_rng

from afl_sim.config import AppConfig, SyncStrategy
from afl_sim.types import PathCollection
from afl_sim.utils import compute_hash_from_dict

from .clock_constructors import gen_clock_chunk_from_scratch, get_client_rates
from .clock_io import (
    load_clock_data,
    load_clock_generator_states,
    save_clock_and_visualize,
    save_metadata,
)
from .clock_types import ClockConfig, ClockData, ClockGenerators, SimulationClock

_EVENTS_PER_CHUNK = 3000
_CHUNK_GEN_THRESHOLD = (
    1000  # number of leftover events in previous chunk triggering new chunk generation
)


def _extract_clock_config(config: AppConfig) -> ClockConfig:
    """
    Generates a canonical dictionary of clock parameters.
    """
    return ClockConfig(
        num_clients=config.simulation.num_clients,
        sigma=config.simulation.client_rate_std,
        seed=config.simulation.rate_seed,
        comm_strategy=config.comm_strategy.type,
        sample_size=config.comm_strategy.sample_size
        if isinstance(config.comm_strategy, SyncStrategy)
        else None,
    )


def get_clock(
    config: AppConfig, data_dir: Path, global_next_idx: int
) -> SimulationClock:
    """
    Retrieves or generates a simulation clock.
    """
    # Do not visualize if simualtion is resumed
    visualize = not global_next_idx and config.visualization.visualize_client_arrivals

    # Create unique clock hash string
    clock_config = _extract_clock_config(config)
    config_hash = compute_hash_from_dict(clock_config)

    # Create output dirs
    output_dir = data_dir / "clocks" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = PathCollection.from_clock_specs(
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
    clock_generators = _get_clock_generators(
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
            merged_data = _clock_merger(
                current_clock_chunk=current_clock_data,
                local_next_idx=local_index,
                next_clock_chunk=next_clock_data,
            )
            return _package_simulation_clock(
                clock_data=merged_data,
                global_idx=global_next_idx,
            )

        sliced_clock = _clock_slicer(
            current_clock_chunk=current_clock_data,
            local_next_idx=local_index,
        )
        return _package_simulation_clock(
            clock_data=sliced_clock,
            global_idx=global_next_idx,
        )

    logger.info("Attempting to fetch chunk 0...")

    chunk_0 = _fetch_or_generate_chunk(
        config=clock_config,
        chunk_num=0,
        clock_generators=clock_generators,
        paths=paths,
        visualize=visualize,
    )

    return _package_simulation_clock(
        clock_data=chunk_0,
        global_idx=0,
    )


def _package_simulation_clock(
    clock_data: ClockData, global_idx: int
) -> SimulationClock:
    return SimulationClock(
        clock_data=clock_data,
        global_first_idx=global_idx,
    )


def _generate_decoupled_rngs(seed: int, rng_num: int) -> list[Generator]:
    ss = SeedSequence(entropy=seed)
    return [default_rng(s) for s in ss.spawn(rng_num)]


def _get_clock_generators(
    num_clients: int, sigma_rate: float, seed: int
) -> ClockGenerators:
    rng1, rng2, rng3 = _generate_decoupled_rngs(seed=seed, rng_num=3)
    rates = get_client_rates(
        num_clients=num_clients, sigma_rate=sigma_rate, rng_rate=rng1
    )
    return ClockGenerators(rates=rates, rng_delay=rng2, rng_select=rng3)


def _fetch_or_generate_chunk(
    config: ClockConfig,
    chunk_num: int,
    clock_generators: ClockGenerators,
    paths: PathCollection,
    visualize: bool,
) -> ClockData:
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
    paths: PathCollection,
    visualize: bool,
) -> None:
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
    paths: PathCollection,
    visualize: bool,
) -> None:
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


def _clock_slicer(
    current_clock_chunk: ClockData,
    local_next_idx: int,
) -> ClockData:
    if local_next_idx > _EVENTS_PER_CHUNK:
        raise ValueError(
            f"Next event_index '{local_next_idx}' exceeds "
            f"max events per chunk '{_EVENTS_PER_CHUNK}'."
        )
    logger.info(f"Slicing current clock chunk at local idx: {local_next_idx}...")

    return ClockData(
        timestamps=current_clock_chunk.timestamps[local_next_idx:],
        client_ids=current_clock_chunk.client_ids[local_next_idx:],
    )


def _clock_merger(
    current_clock_chunk: ClockData,
    local_next_idx: int,
    next_clock_chunk: ClockData,
) -> ClockData:
    if local_next_idx > _EVENTS_PER_CHUNK:
        raise ValueError(
            f"Next event_index '{local_next_idx}' exceeds "
            f"max events per chunk '{_EVENTS_PER_CHUNK}'."
        )
    logger.info(
        f"Merging current and next clock chunks starting at local idx: {local_next_idx}"
    )
    diff_idx = _EVENTS_PER_CHUNK - local_next_idx

    new_timestamps = current_clock_chunk.timestamps
    new_timestamps[:diff_idx] = current_clock_chunk.timestamps[local_next_idx:]
    new_timestamps[diff_idx:] = next_clock_chunk.timestamps[:local_next_idx]

    new_client_ids = current_clock_chunk.client_ids
    new_client_ids[:diff_idx] = current_clock_chunk.client_ids[local_next_idx:]
    new_client_ids[diff_idx:] = next_clock_chunk.client_ids[:local_next_idx]

    return ClockData(timestamps=new_timestamps, client_ids=new_client_ids)
