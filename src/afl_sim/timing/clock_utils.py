from loguru import logger
from numpy.random import Generator, SeedSequence, default_rng

from afl_sim.config import AppConfig, SyncStrategy

from .clock_constructors import get_client_rates
from .clock_types import ClockConfig, ClockData, ClockGenerators, SimulationClock


def extract_clock_config(config: AppConfig) -> ClockConfig:
    """
    Generates a canonical, flattened dictionary of essential clock parameters.

    Extracts pertinent simulation and communication properties from the broader
    application configuration to simplify clock generation logic and hashing.

    Args:
        config (AppConfig): The comprehensive application configuration object.

    Returns:
        ClockConfig: A structured dictionary defining clock generation parameters.
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


def package_simulation_clock(clock_data: ClockData, global_idx: int) -> SimulationClock:
    """
    Wraps raw clock data and its global starting index into a finalized simulation clock.

    Args:
        clock_data (ClockData): A foundational container for raw simulation clock events.
        global_idx (int): The absolute starting index of this clock segment in the broader simulation.

    Returns:
        SimulationClock: A packaged, ready-to-use clock object for the environment.
    """
    return SimulationClock(
        clock_data=clock_data,
        global_first_idx=global_idx,
    )


def _generate_decoupled_rngs(seed: int, rng_num: int) -> list[Generator]:
    """
    Spawns multiple statistically independent random number generators from a single master seed.

    Args:
        seed (int): The root entropy integer for the sequence.
        rng_num (int): The number of separate generator instances to spawn.

    Returns:
        list[Generator]: A list of decoupled numpy random `Generator` instances.
    """
    ss = SeedSequence(entropy=seed)
    return [default_rng(s) for s in ss.spawn(rng_num)]


def get_clock_generators(
    num_clients: int, sigma_rate: float, seed: int
) -> ClockGenerators:
    """
    Initializes the specific decoupled generators required for clock operations.

    Spawns independent random streams for determining base rates, calculating delays,
    and selecting clients to ensure robust stochastic behavior.

    Args:
        num_clients (int): The number of clients participating in the simulation.
        sigma_rate (float): The variance parameter for client rate distributions.
        seed (int): The master random seed used to spawn the decoupled generators.

    Returns:
        ClockGenerators: A centralized container for decoupled random number generators and client rates.
    """
    rng1, rng2, rng3 = _generate_decoupled_rngs(seed=seed, rng_num=3)
    rates = get_client_rates(
        num_clients=num_clients, sigma_rate=sigma_rate, rng_rate=rng1
    )
    return ClockGenerators(rates=rates, rng_delay=rng2, rng_select=rng3)


def clock_slicer(
    current_clock_chunk: ClockData,
    local_next_idx: int,
    events_per_chunk: int,
) -> ClockData:
    """
    Extracts a subset of events from a clock chunk starting at a specific local index.

    Used primarily when resuming a simulation mid-chunk, allowing the engine to skip
    already-processed events.

    Args:
        current_clock_chunk (ClockData): A foundational container for raw simulation clock events currently active.
        local_next_idx (int): The starting index within the current chunk.
        events_per_chunk (int): The maximum expected length of a valid chunk.

    Returns:
        ClockData: A foundational container for raw simulation clock events containing the sliced data.

    Raises:
        ValueError: If the requested starting index exceeds the chunk's capacity.
    """
    if local_next_idx > events_per_chunk:
        raise ValueError(
            f"Next event_index '{local_next_idx}' exceeds "
            f"max events per chunk '{events_per_chunk}'."
        )
    logger.info(f"Slicing current clock chunk at local idx: {local_next_idx}...")

    return ClockData(
        timestamps=current_clock_chunk.timestamps[local_next_idx:],
        client_ids=current_clock_chunk.client_ids[local_next_idx:],
    )


def clock_merger(
    current_clock_chunk: ClockData,
    local_next_idx: int,
    next_clock_chunk: ClockData,
    events_per_chunk: int,
) -> ClockData:
    """
    Stitches together remaining events from the current chunk with the beginning of the next.

    Ensures that the returned block of clock data maintains a consistent length
    (`events_per_chunk`) when a resumption point is close to a chunk boundary.

    Args:
        current_clock_chunk (ClockData): A foundational container for raw simulation clock events currently active.
        local_next_idx (int): The starting index within the current chunk.
        next_clock_chunk (ClockData): The subsequent generated chunk to pull padding events from.
        events_per_chunk (int): The target constant length for the merged arrays.

    Returns:
        ClockData: A foundational container for raw simulation clock events combining both chunks.

    Raises:
        ValueError: If the local index is out of bounds for the current chunk.
    """
    if local_next_idx > events_per_chunk:
        raise ValueError(
            f"Next event_index '{local_next_idx}' exceeds "
            f"max events per chunk '{events_per_chunk}'."
        )
    logger.info(
        f"Merging current and next clock chunks starting at local idx: {local_next_idx}"
    )
    diff_idx = events_per_chunk - local_next_idx

    new_timestamps = current_clock_chunk.timestamps
    new_timestamps[:diff_idx] = current_clock_chunk.timestamps[local_next_idx:]
    new_timestamps[diff_idx:] = next_clock_chunk.timestamps[:local_next_idx]

    new_client_ids = current_clock_chunk.client_ids
    new_client_ids[:diff_idx] = current_clock_chunk.client_ids[local_next_idx:]
    new_client_ids[diff_idx:] = next_clock_chunk.client_ids[:local_next_idx]

    return ClockData(timestamps=new_timestamps, client_ids=new_client_ids)
