from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ClockData:
    """
    A foundational container for raw simulation clock events.

    Attributes:
        timestamps (NDArray[np.float64]): An array of simulation times when events occur.
        client_ids (NDArray[np.int64]): An array of client IDs associated with each timestamp.
            This can be a 1D array for asynchronous arrivals or a 2D matrix for synchronous rounds.
    """

    timestamps: NDArray[np.float64]
    client_ids: NDArray[np.int64]


@dataclass(frozen=True)
class SimulationClock:
    """
    A packaged simulation clock providing helper methods for localized event traversal.

    Wraps raw clock data alongside its global starting index. This allows the simulation
    environment to easily query times and participating clients using a localized event
    index while maintaining global context.

    Attributes:
        clock_data (ClockData): A foundational container for raw simulation clock events.
        global_first_idx (int): The absolute starting sequence index of this chunk in the broader simulation.
    """

    clock_data: ClockData
    global_first_idx: int

    def __len__(self) -> int:
        """
        The total number of recorded events in this clock chunk.

        Returns:
            int: The length of the internal timestamps array.
        """
        return len(self.clock_data.timestamps)

    def local_idx_to_sim_time(self, event_idx: int) -> float:
        """
        Retrieves the exact simulation time for a given local event index.

        Args:
            event_idx (int): The local index within this specific clock chunk.

        Returns:
            float: The simulation timestamp corresponding to the requested index.
        """
        return float(self.clock_data.timestamps[event_idx])

    def local_idx_to_incoming_clients(self, event_idx: int) -> list[int]:
        """
        Retrieves the list of participating client IDs for a given local event index.

        Handles both scalar (asynchronous) and array (synchronous) client ID structures,
        ensuring the output is consistently formatted as a list of integers.

        Args:
            event_idx (int): The local index within this specific clock chunk.

        Returns:
            list[int]: A list of client IDs assigned to the specific event.
        """
        clients = self.clock_data.client_ids[event_idx]
        if clients.ndim > 0:
            return cast("list[int]", clients.tolist())
        return [int(clients)]

    def local_to_global_idx(self, event_idx: int) -> int:
        """
        Translates a local chunk index into its absolute global index within the simulation.

        Args:
            event_idx (int): The local index within this specific clock chunk.

        Returns:
            int: The corresponding global sequence index.
        """
        return event_idx + self.global_first_idx


@dataclass(frozen=True, slots=True)
class ClockStates:
    """
    A snapshot of internal random generator states and temporal boundaries.

    Used to serialize and deserialize the exact state of a clock generation sequence,
    ensuring deterministic reproducibility when generating subsequent chunks or resuming simulations.

    Attributes:
        delay_state (Mapping[str, Any]): The internal state dictionary of the interval delay generator.
        select_state (Mapping[str, Any]): The internal state dictionary of the client selection generator.
        start_time (np.float64): The first timestamp recorded in the associated clock chunk.
        end_time (np.float64): The final timestamp recorded in the associated clock chunk.
    """

    delay_state: Mapping[str, Any]
    select_state: Mapping[str, Any]
    start_time: np.float64
    end_time: np.float64


@dataclass
class ClockGenerators:
    """
    A centralized container for decoupled random number generators and client rates.

    Maintains pre-calculated client rates alongside independent generator instances
    for delays and client selection to ensure robust, isolated stochastic processes.

    Attributes:
        rates (NDArray[np.float64]): The individual participation or arrival rates for all clients.
        rng_delay (Generator): The numpy random generator instance used for calculating time intervals.
        rng_select (Generator): The numpy random generator instance used for sampling clients.
    """

    rates: NDArray[np.float64]
    rng_delay: Generator
    rng_select: Generator

    @property
    def delay_state(self) -> Mapping[str, Any]:
        """
        Retrieves the internal state mapping of the delay generator.

        Returns:
            Mapping[str, Any]: The bit generator state dictionary for `rng_delay`.
        """
        return self.rng_delay.bit_generator.state

    @property
    def select_state(self) -> Mapping[str, Any]:
        """
        Retrieves the internal state mapping of the client selection generator.

        Returns:
            Mapping[str, Any]: The bit generator state dictionary for `rng_select`.
        """
        return self.rng_select.bit_generator.state

    def update_states(
        self,
        states: ClockStates,
    ) -> None:
        """
        Overwrites the internal generator states with values from a provided snapshot.

        Used when resuming chunk generation to ensure the random streams continue
        exactly where the previous chunk left off.

        Args:
            states (ClockStates): A snapshot of internal random generator states and temporal boundaries.
        """
        self.rng_delay.bit_generator.state = states.delay_state
        self.rng_select.bit_generator.state = states.select_state


class ClockConfig(TypedDict):
    """
    A structured dictionary defining clock generation parameters.

    Attributes:
        num_clients (int): The total number of clients participating in the simulation.
        sigma (float): The variance parameter for the client rate distribution.
        seed (int): The master random seed for clock generation.
        comm_strategy (str): The communication strategy type (e.g., 'sync' or 'async').
        sample_size (int | None): The number of clients per round if synchronous, otherwise None.
    """

    num_clients: int
    sigma: float
    seed: int
    comm_strategy: Literal["async", "sync"]
    sample_size: int | None
