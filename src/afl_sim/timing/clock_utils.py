from loguru import logger
from numpy.random import Generator, SeedSequence, default_rng

from afl_sim.config import AppConfig, SyncStrategy

from .clock_constructors import get_client_rates
from .clock_types import ClockConfig, ClockData, ClockGenerators, SimulationClock


def extract_clock_config(config: AppConfig) -> ClockConfig:
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


def package_simulation_clock(clock_data: ClockData, global_idx: int) -> SimulationClock:
    return SimulationClock(
        clock_data=clock_data,
        global_first_idx=global_idx,
    )


def _generate_decoupled_rngs(seed: int, rng_num: int) -> list[Generator]:
    ss = SeedSequence(entropy=seed)
    return [default_rng(s) for s in ss.spawn(rng_num)]


def get_clock_generators(
    num_clients: int, sigma_rate: float, seed: int
) -> ClockGenerators:
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
