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
    """
    Generates a new chunk of clock events from scratch based on the communication strategy.

    Routes the generation process to either synchronous or asynchronous event constructors
    depending on the provided configuration.

    Args:
        config (ClockConfig): A structured dictionary defining clock generation parameters.
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        start_time (float): The simulation timestamp at which to begin generation.
        event_num (int): The total number of events to generate in this chunk.

    Returns:
        ClockData: A foundational container for raw simulation clock events (timestamps and client IDs).

    Raises:
        ValueError: If the requested total events (`event_num`) is zero.
    """
    if event_num == 0:
        raise ValueError("Clock Generation Error: Requested 0 (zero) total events.")

    if config["comm_strategy"] == "async":
        clock_data = _generate_async(
            clock_generators=clock_generators,
            start_time=start_time,
            event_num=event_num,
        )
    elif config["sample_size"] is not None:
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
    Generates deterministic client participation rates (Poisson parameters).

    Samples a zero-mean lognormal distribution to determine the base arrival or
    participation rate for each individual client in the simulated environment.

    Args:
        num_clients (int): The total number of clients to generate rates for.
        sigma_rate (float): The standard deviation of the lognormal distribution.
        rng_rate (Generator): The specific numpy random generator instance for rates.

    Returns:
        NDArray[np.float64]: An array containing the calculated rate for each client.
    """
    return rng_rate.lognormal(0.0, sigma_rate, num_clients)


def _generate_async(
    clock_generators: ClockGenerators,
    start_time: float,
    event_num: int,
) -> ClockData:
    """
    Generates asynchronous simulation events using Poisson Thinning.

    Calculates an aggregate rate for a superimposed Poisson process, generates the
    arrival intervals, and assigns each event to a specific client based on their
    individual probability weights derived from the generator container.

    Args:
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        start_time (float): The timestamp from which to start the event generation.
        event_num (int): The number of asynchronous events to generate.

    Returns:
        ClockData: A foundational container for raw simulation clock events (timestamps and scalar client IDs).
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
    Generates synchronous simulation rounds where multiple clients participate simultaneously.

    Selects a uniformly random subset of clients for each round based on the sample size.
    The duration of each round is determined by the maximum delay among the selected clients.

    Args:
        sample_size (int): The number of clients participating in each synchronous round.
        clock_generators (ClockGenerators): A centralized container for decoupled random number generators and client rates.
        start_time (float): The timestamp from which the synchronous rounds begin.
        event_num (int): The number of synchronous rounds to generate.

    Returns:
        ClockData: A foundational container for raw simulation clock events (timestamps and matrices of selected client IDs).
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
