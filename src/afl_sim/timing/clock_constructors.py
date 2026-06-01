import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .clock_types import ClockConfig, ClockData, ClockGenerators


def gen_clock_chunk_from_scratch(
    config: ClockConfig,
    clock_generators: ClockGenerators,
    start_time: float,
    event_num: int,
) -> ClockData:
    if event_num == 0:
        raise ValueError("Clock Generation Error: Requested 0 (zero) total events.")

    if config["comm_strategy"] == "async":
        clock_data = _generate_async(
            clock_generators=clock_generators,
            start_time=start_time,
            event_num=event_num,
        )
    elif config["comm_strategy"] == "sync" and config["sample_size"] is not None:
        clock_data = _generate_sync(
            sample_size=config["sample_size"],
            clock_generators=clock_generators,
            start_time=start_time,
            event_num=event_num,
        )

    return clock_data


def get_client_rates(
    num_clients: int,
    sigma_rate: float,
    rng_rate: Generator,
) -> NDArray[np.float64]:
    """
    Generates deterministic client rates (Poisson parameters)
    by sampling a zero-mean lognormal distribution.
    """
    return rng_rate.lognormal(0.0, sigma_rate, num_clients)


def _generate_async(
    clock_generators: ClockGenerators,
    start_time: float,
    event_num: int,
) -> ClockData:
    """
    Generates asynchronous events using Poisson Thinning.
    """
    rates = clock_generators.rates
    rng_delay = clock_generators.rng_delay
    rng_select = clock_generators.rng_select

    aggregate_rate = rates.sum()  # Poisson rate of superimposed process

    # Generate the superimposed Poisson process
    intervals = rng_delay.exponential(1.0 / aggregate_rate, size=event_num)
    timestamps = np.cumsum(intervals) + start_time

    # Label each event as client arrival according to client probability
    client_probs = rates / aggregate_rate
    cumulative_probs = np.cumsum(client_probs)
    cumulative_probs[-1] = 1.0

    uniform_draws = rng_select.random(size=event_num)
    client_ids = np.searchsorted(cumulative_probs, uniform_draws)

    return ClockData(
        timestamps=timestamps,
        client_ids=client_ids,
    )


def _generate_sync(
    sample_size: int,
    clock_generators: ClockGenerators,
    start_time: float,
    event_num: int,
) -> ClockData:
    """
    Generates synchronous rounds.
    """
    rates = clock_generators.rates
    rng_delay = clock_generators.rng_delay
    rng_select = clock_generators.rng_select

    # Select clients at each round uniformly
    rand_matrix = rng_select.random((event_num, len(rates)))

    if sample_size == rates.shape[0]:
        selections = np.repeat(np.expand_dims(np.arange(sample_size), 0), event_num, 0)
    else:
        selections = np.argpartition(rand_matrix, sample_size, axis=1)[:, :sample_size]

    selections = selections.astype(np.int64)

    # Set round length equal to maximum sampled client delay
    sel_rates = rates[selections]
    delays = rng_delay.exponential(1.0 / sel_rates)

    round_durations = np.max(delays, axis=1)
    round_ends = np.cumsum(round_durations) + start_time

    return ClockData(timestamps=round_ends, client_ids=selections)
