from dataclasses import dataclass

import numpy as np
import pytest

from afl_sim.config import AsyncStrategy, SyncStrategy
from afl_sim.timing.clock_io import (
    _save_clock_visualization,
    load_clock_data,
    load_clock_generator_states,
    save_clock_and_visualize,
)
from afl_sim.timing.clock_types import ClockConfig, ClockData, ClockGenerators
from afl_sim.timing.clock_utils import get_clock_generators
from afl_sim.types import PathCollection


@dataclass
class ValidTestObject:
    clock_data: ClockData
    clock_generators: ClockGenerators
    chunk_num: int
    paths: PathCollection
    clock_config: ClockConfig


_NUM_EVENTS = 10
_SYNC_SAMPLE_SIZE = 3
_NUM_CLIENTS = 4


@pytest.fixture(
    params=[
        (
            SyncStrategy(sample_size=_SYNC_SAMPLE_SIZE),
            ClockData(
                timestamps=np.arange(_NUM_EVENTS).astype(np.float64),
                client_ids=np.random.choice(
                    _NUM_CLIENTS, (_NUM_EVENTS, _SYNC_SAMPLE_SIZE)
                ),
            ),
        ),
        (
            AsyncStrategy(),
            ClockData(
                timestamps=np.arange(_NUM_EVENTS).astype(np.float64),
                client_ids=np.random.choice(_NUM_CLIENTS, _NUM_EVENTS),
            ),
        ),
    ],
    ids=["sync_strategy", "async_strategy"],
)
def test_obj(request, tmp_path):
    num_clients = _NUM_CLIENTS
    sigma = 0.1
    seed = 42
    hash_str = "test_hash"
    comm_strategy = request.param[0]
    clock_data = request.param[1]

    return ValidTestObject(
        clock_data=clock_data,
        clock_generators=get_clock_generators(
            num_clients=num_clients, sigma_rate=sigma, seed=seed
        ),
        chunk_num=0,
        paths=PathCollection.from_clock_specs(data_dir=tmp_path, hash_str=hash_str),
        clock_config=ClockConfig(
            num_clients=num_clients,
            sigma=sigma,
            seed=seed,
            comm_strategy=comm_strategy.type,
            sample_size=comm_strategy.sample_size
            if isinstance(comm_strategy, SyncStrategy)
            else None,
        ),
    )


def test_data_io_roundtrip(test_obj):
    save_clock_and_visualize(
        clock_data=test_obj.clock_data,
        clock_generators=test_obj.clock_generators,
        chunk_num=test_obj.chunk_num,
        paths=test_obj.paths,
        clock_config=test_obj.clock_config,
        visualize=False,
    )

    chunk_path = test_obj.paths.get_clock_chunk_path(test_obj.chunk_num)

    states_loaded = load_clock_generator_states(chunk_path=chunk_path)
    data_loaded = load_clock_data(chunk_path=chunk_path)

    assert states_loaded.delay_state == test_obj.clock_generators.delay_state, (
        "Saved and loaded RNG delay states do not match."
    )
    assert states_loaded.select_state == test_obj.clock_generators.select_state, (
        "Saved and loaded RNG select states do not match."
    )

    assert np.allclose(
        data_loaded.timestamps, test_obj.clock_data.timestamps, atol=1e-9
    ), "Saved and loaded timestamps do not match."
    assert np.array_equal(data_loaded.client_ids, test_obj.clock_data.client_ids), (
        "Saved and loaded client_ids do not match."
    )


@pytest.mark.parametrize(
    "visualize, error_message",
    [
        (True, "Clock plot not requested when visualize=True."),
        (False, "Clock plot requested when visualize=False."),
    ],
)
def test_visualization_trigger(visualize, error_message, mocker, test_obj):
    mock_plot = mocker.patch("afl_sim.timing.clock_io.save_clock_plot")

    _save_clock_visualization(
        clock=test_obj.clock_data,
        paths=test_obj.paths,
        config_dict=test_obj.clock_config,
        visualize=visualize,
    )
    assert mock_plot.called == visualize, error_message


def test_visualization_is_fed_1D_timestamps(mocker, test_obj):
    mock_plot = mocker.patch("afl_sim.timing.clock_io.save_clock_plot")

    _save_clock_visualization(
        clock=test_obj.clock_data,
        paths=test_obj.paths,
        config_dict=test_obj.clock_config,
        visualize=True,
    )

    mock_timestamps = mock_plot.call_args_list[0].kwargs.get("timestamps")

    assert mock_timestamps.ndim == 1
