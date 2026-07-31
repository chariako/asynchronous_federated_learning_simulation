from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from afl_sim.config import AppConfig, AsyncStrategy, SimulationConfig, SyncStrategy
from afl_sim.paths import ClockPathCollection
from afl_sim.timing.clock_factory import (
    _fetch_or_generate_chunk,
    _generate_chunk_and_save,
    _recursive_chunk_generation,
    get_clock,
)
from afl_sim.timing.clock_types import ClockConfig, ClockData, ClockGenerators
from afl_sim.timing.clock_utils import (
    extract_clock_config,
    get_clock_generators,
)
from tests.timing.helpers import (
    assert_chunk_data_equality,
    assert_chunk_data_inequality,
    assert_chunk_state_equality,
    assert_chunk_state_inequality,
)

_NUM_EVENTS_SHORT = 10


@dataclass
class ClockTestContext:
    clock_generators: ClockGenerators
    paths: ClockPathCollection
    clock_config: ClockConfig
    app_config: AppConfig


@pytest.fixture(
    params=[
        SyncStrategy(sample_size=3),
        AsyncStrategy(),
    ],
    ids=["sync_strategy", "async_strategy"],
)
def make_base_input(request):
    def _factory(seed, hash_str, file_dir):
        num_clients = 4
        sigma = 0.1
        comm_strategy = request.param

        data_dir = file_dir / hash_str
        data_dir.mkdir(parents=True, exist_ok=True)
        app_config = AppConfig(
            comm_strategy=comm_strategy,
            simulation=SimulationConfig(
                num_clients=num_clients,
                client_rate_std=sigma,
                rate_seed=seed,
            ),
        )

        return ClockTestContext(
            clock_generators=get_clock_generators(
                num_clients=num_clients,
                sigma_rate=sigma,
                seed=seed,
            ),
            paths=ClockPathCollection.from_clock_specs(
                data_dir=file_dir / hash_str, hash_str=hash_str
            ),
            clock_config=extract_clock_config(app_config),
            app_config=app_config,
        )

    return _factory


# --- Recursive generation tests ---


def test_recursive_chunk_generation(
    monkeypatch, tmp_path, make_base_input, capture_logs
):
    # Create cofing and file directories
    base_input = make_base_input(seed=42, hash_str="test_hash", file_dir=tmp_path)

    num_events = _NUM_EVENTS_SHORT
    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", num_events)

    # Base case (chunk_num = 0)
    _recursive_chunk_generation(
        config=base_input.clock_config,
        chunk_num=0,
        clock_generators=deepcopy(base_input.clock_generators),
        paths=base_input.paths,
        visualize=False,
    )
    chunk_0_path = base_input.paths.get_clock_chunk_path(0)
    assert chunk_0_path.exists(), "Base chunk 0 not generated."

    # Chunk 2 requires recursive generation of chunk 1
    _recursive_chunk_generation(
        config=base_input.clock_config,
        chunk_num=2,
        clock_generators=deepcopy(base_input.clock_generators),
        paths=base_input.paths,
        visualize=False,
    )

    chunk_2_path = base_input.paths.get_clock_chunk_path(2)
    chunk_1_path = base_input.paths.get_clock_chunk_path(1)

    assert "Clock data for chunk 1 missing" in capture_logs.text, (
        "No warning raised for missing chunk."
    )

    assert chunk_1_path.exists(), (
        "Missing non-base chunk dependency (chunk_num > 0) not generated."
    )
    assert chunk_2_path.exists(), (
        "Requested non-base chunk (chunk_num > 0) not generated."
    )


def test_recursive_chunks_match_ordered(monkeypatch, make_base_input, tmp_path):
    num_events = _NUM_EVENTS_SHORT
    idx_range = np.arange(num_events)

    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", num_events)

    recursive_input = make_base_input(
        seed=42, hash_str="recursive_hash", file_dir=tmp_path
    )
    ordered_input = make_base_input(seed=42, hash_str="ordered_hash", file_dir=tmp_path)

    # Generate chunk 1 recursively
    _recursive_chunk_generation(
        config=recursive_input.clock_config,
        chunk_num=1,
        clock_generators=deepcopy(recursive_input.clock_generators),
        paths=recursive_input.paths,
        visualize=False,
    )

    # Generate chunks 0, 1 in order and compare
    for chunk_num in range(2):
        _recursive_chunk_generation(
            config=ordered_input.clock_config,
            chunk_num=chunk_num,
            clock_generators=deepcopy(ordered_input.clock_generators),
            paths=ordered_input.paths,
            visualize=False,
        )

        recursive_path = recursive_input.paths.get_clock_chunk_path(chunk_num)
        ordered_path = ordered_input.paths.get_clock_chunk_path(chunk_num)

        assert_chunk_state_equality(recursive_path, ordered_path)
        assert_chunk_data_equality(recursive_path, idx_range, ordered_path, idx_range)


# --- Reproducibility tests ---


def test_chunk_rng_reproducibility(monkeypatch, make_base_input, tmp_path):
    base_events = _NUM_EVENTS_SHORT
    total_events = 2 * base_events

    short_input = make_base_input(seed=42, hash_str="short_hash", file_dir=tmp_path)

    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", base_events)

    _recursive_chunk_generation(
        config=short_input.clock_config,
        chunk_num=1,
        clock_generators=deepcopy(short_input.clock_generators),
        paths=short_input.paths,
        visualize=False,
    )

    long_input = make_base_input(seed=42, hash_str="long_hash", file_dir=tmp_path)
    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", total_events)

    _recursive_chunk_generation(
        config=long_input.clock_config,
        chunk_num=0,
        clock_generators=deepcopy(long_input.clock_generators),
        paths=long_input.paths,
        visualize=False,
    )

    long_path = long_input.paths.get_clock_chunk_path(0)

    short_path_0 = short_input.paths.get_clock_chunk_path(0)
    short_path_1 = short_input.paths.get_clock_chunk_path(1)

    idx_range_short = np.arange(base_events)

    assert_chunk_data_equality(
        long_path, np.arange(0, 10), short_path_0, idx_range_short
    )
    assert_chunk_data_equality(
        long_path, np.arange(10, 20), short_path_1, idx_range_short
    )

    assert_chunk_state_equality(long_path, short_path_1)


@pytest.mark.parametrize(("seed_1", "seed_2"), [(42, 42), (42, 43)])
def test_chunk_seed_reproducibility(
    seed_1,
    seed_2,
    monkeypatch,
    make_base_input,
    tmp_path,
):
    chunk_num = 0
    start_time = 0.0
    num_events = _NUM_EVENTS_SHORT
    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", num_events)

    input_1 = make_base_input(seed=seed_1, hash_str="hash_1", file_dir=tmp_path)

    _generate_chunk_and_save(
        config=input_1.clock_config,
        start_time=start_time,
        chunk_num=chunk_num,
        clock_generators=input_1.clock_generators,
        paths=input_1.paths,
        visualize=False,
    )

    input_2 = make_base_input(seed=seed_2, hash_str="hash_2", file_dir=tmp_path)

    _generate_chunk_and_save(
        config=input_2.clock_config,
        start_time=start_time,
        chunk_num=chunk_num,
        clock_generators=input_2.clock_generators,
        paths=input_2.paths,
        visualize=False,
    )

    path_1 = input_1.paths.get_clock_chunk_path(chunk_num)
    path_2 = input_2.paths.get_clock_chunk_path(chunk_num)
    idx_range = np.arange(num_events)

    if seed_1 == seed_2:
        assert_chunk_state_equality(path_1, path_2)
        assert_chunk_data_equality(path_1, idx_range, path_2, idx_range)
    else:
        assert_chunk_state_inequality(path_1, path_2)
        assert_chunk_data_inequality(path_1, idx_range, path_2, idx_range)


# --- Fetching tests ---


@pytest.mark.parametrize(
    ("expected_generate", "error_message"),
    [
        (
            False,
            "New chunk generated from scratch when chunk matching specs exists.",
        ),
        (
            True,
            "New chunk not generated when existing chunk matching specs does not exist.",
        ),
    ],
)
def test_chunk_fetching_or_generation(
    expected_generate, error_message, mocker, make_base_input, tmp_path
):
    mock_generate = mocker.patch(
        "afl_sim.timing.clock_factory._recursive_chunk_generation"
    )
    mock_load = mocker.patch("afl_sim.timing.clock_factory.load_clock_data")

    chunk_num = 0
    base_input = make_base_input(seed=42, hash_str="test_hash", file_dir=tmp_path)

    chunk_path = base_input.paths.get_clock_chunk_path(chunk_num)
    if not expected_generate:
        chunk_path.touch()

    _fetch_or_generate_chunk(
        config=base_input.clock_config,
        chunk_num=chunk_num,
        clock_generators=base_input.clock_generators,
        paths=base_input.paths,
        visualize=False,
    )

    assert mock_generate.called == expected_generate, error_message

    (
        mock_load.assert_called_once_with(chunk_path=chunk_path),
        ("Requested path does not match chunk path."),
    )


# --- Chunk generation logic tests ---


def generate_dynamic_cases(
    num_events, threshold
) -> list[tuple[int, int, int, list[int], str]]:
    cases = []

    test_indices = [
        0,
        (num_events // 2) - 1,
        (num_events // 2) + 1,
        num_events + 2,
        (num_events * 10) + (num_events // 2) + 2,
    ]

    for global_idx in test_indices:
        chunk = global_idx // num_events
        local_idx = global_idx % num_events

        if global_idx == 0:
            action = "none"
            fetched = [chunk]
        elif local_idx >= threshold:
            action = "merge"
            fetched = [chunk, chunk + 1]
        else:
            action = "slice"
            fetched = [chunk]

        cases.append((num_events, threshold, global_idx, fetched, action))

    return cases


@pytest.mark.parametrize(
    (
        "num_events",
        "threshold",
        "global_next_idx",
        "expected_fetched",
        "expected_action",
    ),
    generate_dynamic_cases(
        num_events=_NUM_EVENTS_SHORT, threshold=_NUM_EVENTS_SHORT // 2
    ),
)
def test_leftover_triggers_generation(
    mocker,
    num_events,
    threshold,
    global_next_idx,
    expected_fetched,
    expected_action,
    monkeypatch,
    make_base_input,
    tmp_path,
):
    monkeypatch.setattr("afl_sim.timing.clock_factory._EVENTS_PER_CHUNK", num_events)
    monkeypatch.setattr("afl_sim.timing.clock_factory._CHUNK_GEN_THRESHOLD", threshold)

    def mock_fetch_or_generate_behavior(chunk_num, **kwargs):
        return ClockData(
            timestamps=np.arange(
                chunk_num * num_events, (1 + chunk_num) * num_events
            ).astype(np.float64),
            client_ids=np.arange(num_events).astype(np.int64),
        )

    # Mocks setup
    mock_fetch = mocker.patch(
        "afl_sim.timing.clock_factory._fetch_or_generate_chunk",
        side_effect=mock_fetch_or_generate_behavior,
    )
    mock_slicer = mocker.patch("afl_sim.timing.clock_factory.clock_slicer")
    mock_merger = mocker.patch("afl_sim.timing.clock_factory.clock_merger")
    mock_packager = mocker.patch(
        "afl_sim.timing.clock_factory.package_simulation_clock"
    )

    base_input = make_base_input(seed=42, hash_str="test_hash", file_dir=tmp_path)

    get_clock(
        config=base_input.app_config,
        data_dir=tmp_path,
        global_next_idx=global_next_idx,
    )

    actual_fetched = [
        call.kwargs.get("chunk_num") for call in mock_fetch.call_args_list
    ]
    assert actual_fetched == expected_fetched, (
        f"Expected chunks {expected_fetched}, but fetched {actual_fetched}."
    )

    if expected_action == "slice":
        mock_slicer.assert_called_once_with(
            current_clock_chunk=mocker.ANY,
            local_next_idx=global_next_idx % num_events,
            events_per_chunk=num_events,
        )
        mock_merger.assert_not_called()

    elif expected_action == "merge":
        mock_merger.assert_called_once_with(
            current_clock_chunk=mocker.ANY,
            local_next_idx=global_next_idx % num_events,
            next_clock_chunk=mocker.ANY,
            events_per_chunk=num_events,
        )
        mock_slicer.assert_not_called()

    else:
        mock_slicer.assert_not_called()
        mock_merger.assert_not_called()

    mock_packager.assert_called_once_with(
        clock_data=mocker.ANY, global_idx=global_next_idx
    )
