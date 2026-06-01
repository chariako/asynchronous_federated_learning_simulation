from copy import deepcopy

import numpy as np
import pytest

from afl_sim.timing.clock_types import ClockData
from afl_sim.timing.clock_utils import (
    clock_merger,
    clock_slicer,
    get_clock_generators,
)

_NUM_EVENTS_SHORT = 10


# --- Helper functions ---


def assert_state_equality(state_1, state_2) -> None:
    assert state_1 == state_2, "Expected equal states are not equal."


def assert_state_inequality(state_1, state_2) -> None:
    assert state_1 != state_2, "Expected non-equal states are equal."


def assert_timestamp_or_rate_equality(x, y) -> None:
    assert np.allclose(x, y, atol=1e-9), (
        "Expected equal rates or timestamps are not equal."
    )


def assert_timestamp_or_rate_inequality(x, y) -> None:
    assert not np.allclose(x, y, atol=1e-4), "Expected non-equal timestamps are equal."


def assert_id_equality(ids_1, ids_2) -> None:
    assert np.array_equal(ids_1, ids_2), "Expected equal client_ids are not equal."


def assert_id_inequality(ids_1, ids_2) -> None:
    assert not np.array_equal(ids_1, ids_2), "Expected non-equal client_ids are equal."


def mock_clock_data_generator(num_events, chunk_num, sample_size) -> ClockData:
    if sample_size == 1:
        output_size = num_events
    elif sample_size > 1:
        output_size = (num_events, sample_size)
    return ClockData(
        timestamps=np.arange(
            chunk_num * num_events, (1 + chunk_num) * num_events
        ).astype(np.float64),
        client_ids=np.random.choice(10, size=output_size).astype(np.int64),
    )


# --- Reproducibility tests ---


@pytest.mark.parametrize("seed_1, seed_2", [(42, 42), (42, 43)])
def test_clock_generator_reproducibility(seed_1, seed_2):
    num_clients = 4
    sigma = 0.1

    gen_1 = get_clock_generators(
        num_clients=num_clients,
        sigma_rate=sigma,
        seed=seed_1,
    )

    gen_2 = get_clock_generators(
        num_clients=num_clients,
        sigma_rate=sigma,
        seed=seed_2,
    )

    if seed_1 == seed_2:
        assert_timestamp_or_rate_equality(gen_1.rates, gen_2.rates)
        assert_state_equality(gen_1.delay_state, gen_2.delay_state)
        assert_state_equality(gen_1.select_state, gen_2.select_state)
    else:
        assert_timestamp_or_rate_inequality(gen_1.rates, gen_2.rates)
        assert_state_inequality(gen_1.delay_state, gen_2.delay_state)
        assert_state_inequality(gen_1.select_state, gen_2.select_state)


# --- Slicing and merging tests ---


@pytest.mark.parametrize("sample_size", [1, 3])
def test_clock_slicer(sample_size):
    num_events = _NUM_EVENTS_SHORT
    next_index = 6
    mock_chunk = mock_clock_data_generator(
        num_events=num_events, chunk_num=0, sample_size=sample_size
    )

    sliced_clock = clock_slicer(
        current_clock_chunk=mock_chunk,
        local_next_idx=next_index,
        events_per_chunk=num_events,
    )

    assert_timestamp_or_rate_equality(
        sliced_clock.timestamps, mock_chunk.timestamps[next_index:]
    )
    assert_id_equality(sliced_clock.client_ids, mock_chunk.client_ids[next_index:])


@pytest.mark.parametrize("sample_size", [1, 3])
def test_clock_slicer_failure(sample_size):
    num_events = _NUM_EVENTS_SHORT

    local_next_index = num_events + 1
    mock_chunk = mock_clock_data_generator(
        num_events=num_events, chunk_num=0, sample_size=sample_size
    )
    with pytest.raises(
        ValueError,
        match=f"Next event_index '{local_next_index}' exceeds max events per chunk '{num_events}'.",
    ):
        clock_slicer(
            current_clock_chunk=mock_chunk,
            local_next_idx=local_next_index,
            events_per_chunk=num_events,
        )


@pytest.mark.parametrize("sample_size", [1, 3])
def test_clock_merger(sample_size):
    num_events = _NUM_EVENTS_SHORT

    local_next_index = 6
    long_chunk = mock_clock_data_generator(
        num_events=2 * num_events, chunk_num=0, sample_size=sample_size
    )

    merged_clock = clock_merger(
        current_clock_chunk=ClockData(
            timestamps=deepcopy(long_chunk.timestamps[:num_events]),
            client_ids=deepcopy(long_chunk.client_ids[:num_events]),
        ),
        local_next_idx=local_next_index,
        next_clock_chunk=ClockData(
            timestamps=deepcopy(long_chunk.timestamps[num_events:]),
            client_ids=deepcopy(long_chunk.client_ids[num_events:]),
        ),
        events_per_chunk=num_events,
    )

    assert_timestamp_or_rate_equality(
        merged_clock.timestamps,
        long_chunk.timestamps[local_next_index : num_events + local_next_index],
    )
    assert_id_equality(
        merged_clock.client_ids,
        long_chunk.client_ids[local_next_index : num_events + local_next_index],
    )


@pytest.mark.parametrize("sample_size", [1, 3])
def test_clock_merger_failure(sample_size):
    num_events = _NUM_EVENTS_SHORT

    local_next_index = num_events + 1
    long_chunk = mock_clock_data_generator(
        num_events=2 * num_events, chunk_num=0, sample_size=sample_size
    )

    with pytest.raises(
        ValueError,
        match=f"Next event_index '{local_next_index}' exceeds max events per chunk '{num_events}'.",
    ):
        clock_merger(
            current_clock_chunk=ClockData(
                timestamps=long_chunk.timestamps[:num_events],
                client_ids=long_chunk.client_ids[:num_events],
            ),
            local_next_idx=local_next_index,
            next_clock_chunk=ClockData(
                timestamps=long_chunk.timestamps[num_events:],
                client_ids=long_chunk.client_ids[num_events:],
            ),
            events_per_chunk=num_events,
        )
