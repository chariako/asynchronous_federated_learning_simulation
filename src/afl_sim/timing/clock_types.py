from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


@dataclass
class ClockData:
    timestamps: NDArray[np.float64]
    client_ids: NDArray[np.int64]


@dataclass
class SimulationClock:
    clock_data: ClockData
    global_first_idx: int

    @property
    def length(self) -> int:
        return len(self.clock_data.timestamps)

    def local_idx_to_sim_time(self, event_idx: int) -> float:
        return float(self.clock_data.timestamps[event_idx])

    def local_idx_to_incoming_clients(self, event_idx: int) -> list[int]:
        clients = self.clock_data.client_ids[event_idx]
        if clients.ndim > 0:
            return cast(list[int], clients.tolist())
        return [int(clients)]

    def local_to_global_idx(self, event_idx: int) -> int:
        return event_idx + self.global_first_idx


@dataclass
class ClockStates:
    delay_state: Mapping[str, Any]
    select_state: Mapping[str, Any]
    start_time: np.float64
    end_time: np.float64


@dataclass
class ClockGenerators:
    rates: NDArray[np.float64]
    rng_delay: Generator
    rng_select: Generator

    @property
    def delay_state(self) -> Mapping[str, Any]:
        return self.rng_delay.bit_generator.state

    @property
    def select_state(self) -> Mapping[str, Any]:
        return self.rng_select.bit_generator.state

    def update_states(
        self,
        states: ClockStates,
    ) -> None:
        self.rng_delay.bit_generator.state = states.delay_state
        self.rng_select.bit_generator.state = states.select_state


class ClockConfig(TypedDict):
    num_clients: int
    sigma: float
    seed: int
    comm_strategy: str
    sample_size: int | None
