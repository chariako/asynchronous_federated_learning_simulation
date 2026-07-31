import numpy as np

from afl_sim.timing.clock_io import load_clock_data, load_clock_generator_states


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


def assert_chunk_state_equality(clockpath_1, clockpath_2) -> None:
    clock_1_states = load_clock_generator_states(clockpath_1)
    clock_2_states = load_clock_generator_states(clockpath_2)

    assert_state_equality(clock_1_states.delay_state, clock_2_states.delay_state)
    assert_state_equality(clock_1_states.select_state, clock_2_states.select_state)


def assert_chunk_state_inequality(clockpath_1, clockpath_2) -> None:
    clock_1_states = load_clock_generator_states(clockpath_1)
    clock_2_states = load_clock_generator_states(clockpath_2)

    assert_state_inequality(clock_1_states.delay_state, clock_2_states.delay_state)
    assert_state_inequality(clock_1_states.select_state, clock_2_states.select_state)


def assert_chunk_data_equality(
    clockpath_1, idx_range_1, clockpath_2, idx_range_2
) -> None:
    clock_1_data = load_clock_data(clockpath_1)
    clock_2_data = load_clock_data(clockpath_2)

    assert_timestamp_or_rate_equality(
        clock_1_data.timestamps[idx_range_1], clock_2_data.timestamps[idx_range_2]
    )
    assert_id_equality(
        clock_1_data.client_ids[idx_range_1], clock_2_data.client_ids[idx_range_2]
    )


def assert_chunk_data_inequality(
    clockpath_1, idx_range_1, clockpath_2, idx_range_2
) -> None:
    clock_1_data = load_clock_data(clockpath_1)
    clock_2_data = load_clock_data(clockpath_2)

    assert_timestamp_or_rate_inequality(
        clock_1_data.timestamps[idx_range_1], clock_2_data.timestamps[idx_range_2]
    )
    assert_id_inequality(
        clock_1_data.client_ids[idx_range_1], clock_2_data.client_ids[idx_range_2]
    )
