import itertools
import math

import numpy as np
import pytest

from afl_sim.config import AsyncStrategy, SyncStrategy
from afl_sim.timing.clock_constructors import (
    _generate_async,
    _generate_sync,
    gen_clock_chunk_from_scratch,
)
from afl_sim.timing.clock_types import ClockConfig, ClockGenerators
from afl_sim.timing.clock_utils import _generate_decoupled_rngs, get_clock_generators


# -- Router test --
@pytest.fixture(
    params=[
        (SyncStrategy(sample_size=3), "_generate_sync", "_generate_async"),
        (AsyncStrategy(), "_generate_async", "_generate_sync"),
    ],
    ids=["sync_strategy", "async_strategy"],
)
def router_test_input(request):
    comm_strategy, expected_call, unexpected_call = request.param
    seed = 42
    num_clients = 4
    sigma = 0.1

    return {
        "config": ClockConfig(
            num_clients=num_clients,
            sigma=sigma,
            seed=seed,
            comm_strategy=comm_strategy.type,
            sample_size=comm_strategy.sample_size
            if isinstance(comm_strategy, SyncStrategy)
            else None,
        ),
        "clock_generators": get_clock_generators(
            num_clients=num_clients, sigma_rate=sigma, seed=seed
        ),
        "expected_call": expected_call,
        "unexpected_call": unexpected_call,
    }


def test_constructor_routing(mocker, router_test_input):
    mock_expected_call = mocker.patch(
        f"afl_sim.timing.clock_constructors.{router_test_input['expected_call']}"
    )
    mock_unexpected_call = mocker.patch(
        f"afl_sim.timing.clock_constructors.{router_test_input['unexpected_call']}"
    )

    gen_clock_chunk_from_scratch(
        config=router_test_input["config"],
        clock_generators=router_test_input["clock_generators"],
        start_time=0.0,
        event_num=10,
    )

    mock_expected_call.assert_called_once()
    mock_unexpected_call.assert_not_called()


@pytest.mark.parametrize(
    "router_test_input",
    [(SyncStrategy(sample_size=3), "_generate_sync", "_generate_async")],
    indirect=True,
)
def test_zero_events_request_raises_error(router_test_input):
    with pytest.raises(ValueError, match="Clock Generation Error"):
        gen_clock_chunk_from_scratch(
            config=router_test_input["config"],
            clock_generators=router_test_input["clock_generators"],
            start_time=0.0,
            event_num=0,
        )


# -- Data integrity --
@pytest.fixture(
    params=[
        np.array([1.0, 0.1, 0.05, 0.01]),
        np.array([1.0, 1.0, 1.0, 1.0]),
    ],
    ids=["heterogeneous_rates", "homogeneous_rates"],
)
def clock_input(request):
    rng1, rng2 = _generate_decoupled_rngs(seed=42, rng_num=2)
    return {
        "clock_generators": ClockGenerators(
            rates=request.param, rng_delay=rng1, rng_select=rng2
        ),
        "start_time": 123.0,
    }


def verify_timestamp_integrity(clock_data, start_time, event_num) -> None:
    time_diffs = np.diff(clock_data.timestamps)

    assert np.all(clock_data.timestamps > 0), "Timestamps are non-positive."
    assert np.all(clock_data.timestamps > start_time), (
        "Timestamps do not exceed start time."
    )
    assert np.all(time_diffs > 0), "Timestamps are not strictly increasing."
    assert clock_data.timestamps.shape[0] == event_num, (
        "Timestamp length does not match event_num."
    )


def test_async_clock_data(clock_input):
    event_num = 50

    clock_data = _generate_async(
        clock_generators=clock_input["clock_generators"],
        start_time=clock_input["start_time"],
        event_num=event_num,
    )

    verify_timestamp_integrity(
        clock_data=clock_data, start_time=clock_input["start_time"], event_num=event_num
    )

    assert clock_data.client_ids.shape[0] == event_num, (
        "Client_ids length does not match event_num."
    )
    assert clock_data.client_ids.ndim == 1, (
        "Async clock clients per event does not match 1."
    )


@pytest.mark.parametrize("sample_size", [1, 2, 3, 4])
def test_sync_clock_data(clock_input, sample_size):
    event_num = 50

    clock_data = _generate_sync(
        clock_generators=clock_input["clock_generators"],
        start_time=clock_input["start_time"],
        event_num=event_num,
        sample_size=sample_size,
    )

    verify_timestamp_integrity(
        clock_data=clock_data, start_time=clock_input["start_time"], event_num=event_num
    )

    assert clock_data.client_ids.shape[0] == event_num, (
        "Client_ids length does not match event_num."
    )
    assert clock_data.client_ids.ndim == 2, "Sync clock is not 2-dimensional."
    assert clock_data.client_ids.shape[1] == sample_size, (
        "Sync clock clients per event does not match sample_size."
    )


# -- Statistical tests --
def cosine_similarity(x, y) -> float:
    return np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)  # type: ignore


@pytest.mark.slow
def test_async_clock_stats(clock_input):
    rates = clock_input["clock_generators"].rates
    num_clients = len(rates)

    clock_data = _generate_async(
        clock_generators=clock_input["clock_generators"],
        start_time=clock_input["start_time"],
        event_num=100_000,
    )

    aggregate_rate = rates.sum()

    time_diffs = np.diff(clock_data.timestamps)
    counts, _ = np.histogram(clock_data.client_ids, bins=range(num_clients + 1))
    unique_ids = np.unique(clock_data.client_ids)

    assert np.isclose(time_diffs.mean(), 1 / aggregate_rate, rtol=0.01), (
        "Average interarrival time does not match aggregate rate."
    )

    assert unique_ids.min() >= 0 and unique_ids.max() < num_clients, (
        "Client IDs out of bounds."
    )
    assert cosine_similarity(rates, counts) > 0.99, (
        "Arrival distribution does not match client rates."
    )


def _inclusion_exclusion_max(rates) -> float:
    k = len(rates)
    expected_max = 0.0

    for i in range(1, k + 1):
        sign = (-1) ** (i - 1)
        for subset in itertools.combinations(rates, i):
            expected_max += sign * (1.0 / sum(subset))

    return expected_max


def exact_sampled_expected_time(rates, sample_size) -> float:
    num_clients = len(rates)

    total_expected_time = 0.0

    combinations = itertools.combinations(rates, sample_size)
    num_combinations = math.comb(num_clients, sample_size)

    for sample in combinations:
        total_expected_time += _inclusion_exclusion_max(sample)

    return total_expected_time / num_combinations


@pytest.mark.slow
@pytest.mark.parametrize("sample_size", [1, 2, 3, 4])
def test_sync_clock_stats(clock_input, sample_size):
    rates = clock_input["clock_generators"].rates
    num_clients = len(rates)

    clock_data = _generate_sync(
        clock_generators=clock_input["clock_generators"],
        start_time=clock_input["start_time"],
        event_num=100_000,
        sample_size=sample_size,
    )

    time_diffs = np.diff(clock_data.timestamps)
    counts, _ = np.histogram(clock_data.client_ids, bins=range(num_clients + 1))
    unique_ids = np.unique(clock_data.client_ids)

    exp_time = exact_sampled_expected_time(rates=rates, sample_size=sample_size)

    assert np.isclose(time_diffs.mean(), exp_time, rtol=0.01), (
        "Average interarrival time does not match expected rate."
    )

    assert unique_ids.min() >= 0 and unique_ids.max() < num_clients, (
        "Client IDs out of bounds."
    )
    assert np.all(np.diff(np.sort(clock_data.client_ids, axis=1), axis=1) != 0), (
        "Duplicate clients per sample found."
    )
    assert cosine_similarity(np.ones(num_clients), counts) > 0.99, (
        "Client distribution violates uniform sampling."
    )
