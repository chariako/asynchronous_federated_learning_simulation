import time

from loguru import logger
from torch import device

from afl_sim.checkpointing import CheckpointManager
from afl_sim.client import Client
from afl_sim.config import MemStrategyConfig
from afl_sim.server import Server
from afl_sim.timing import (
    SimulationClock,
)
from afl_sim.types import (
    SimulationModel,
    TensorDict,
)
from afl_sim.utils import (
    MetricsLogger,
)

from .simulation_states import (
    AsyncStateManager,
    ClientMemoryStates,
)


class Simulation:
    """
    Initializes the Simulation environment with injected dependencies.

    Args:
        mem_strategy (MemStrategyConfig): Strategy for local client memory-based tracking.
        timeout (float): The simulation duration in seconds.
        metrics_logger (MetricsLogger): The utility responsible for logging performance metrics.
        checkpoint_manager (CheckpointManager): The utility for saving and loading states.
        device (device): The PyTorch device (CPU/GPU) utilized for model operations.
        server (Server): The centralized federated learning server.
        clients (list[Client]): The collection of all participating client nodes.
        clock (SimulationClock): The timekeeping mechanism mapping events to simulated time and incoming clients.
        model_shell (SimulationModel): The underlying PyTorch model architecture.
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if the communication is synchronous.
    """

    def __init__(
        self,
        mem_strategy: MemStrategyConfig,
        timeout: float,
        metrics_logger: MetricsLogger,
        checkpoint_manager: CheckpointManager,
        device: device,
        server: Server,
        clients: list[Client],
        clock: SimulationClock,
        model_shell: SimulationModel,
        async_states: AsyncStateManager | None,
    ):
        """
        Initializes the Simulation environment with injected dependencies.

        Args:
            mem_strategy (MemStrategyConfig): Strategy for local client memory-based tracking.
            timeout (float): The simulation duration in seconds.
            metrics_logger (MetricsLogger): Logger for simulation metrics.
            checkpoint_manager (CheckpointManager): Manager for I/O state saving.
            device (device): Target hardware device for tensor operations.
            server (Server): The initialized federated server.
            clients (list[Client]): The initialized list of client nodes.
            clock (SimulationClock): The simulation timekeeper.
            model_shell (SimulationModel): The base PyTorch model architecture.
            async_states (AsyncStateManager | None): The centralized artifact database storing simulation states (async only).
        """
        self.mem_strategy = mem_strategy
        self.timeout = timeout
        self.local_idx = 0
        self.metrics_logger = metrics_logger
        self.device = device
        self.clock = clock

        # Client and server initialization
        self.server = server
        self.clients = clients

        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_time = time.time()
        self.stop_requested = False

        # Shell model and centralized version database
        self.model_shell = model_shell.to(device=device)
        self.async_states = async_states

    @property
    def global_idx(self) -> int:
        """
        Calculates the current global simulation index based on the local event counter.

        Returns:
            int: The current global iteration or round index.
        """
        return self.clock.local_to_global_idx(self.local_idx)

    @property
    def incoming_clients(self) -> list[int]:
        """
        Retrieves the list of client IDs scheduled to arrive at the current local step.

        Returns:
            list[int]: A list of unique client identifiers participating in this event.
        """
        return self.clock.local_idx_to_incoming_clients(self.local_idx)

    @property
    def sim_time(self) -> float:
        """
        Calculates the current elapsed time within the simulated environment.

        Returns:
            float: The simulated time corresponding to the current local step.
        """
        return self.clock.local_idx_to_sim_time(self.local_idx)

    def _build_client_state_dicts(self) -> ClientMemoryStates | None:
        """
        Aggregates the internal memory states of all participating clients.

        Returns:
            ClientMemoryStates | None: An object containing all client memory states,
                or None if the configured training strategy does not utilize client memory.
        """
        if not self.mem_strategy.type.has_memory:
            return None

        client_states = ClientMemoryStates()
        for client in self.clients:
            client_states.add_client_mem_state(
                client_id=client.client_id, mem_state=client.memory
            )

        return client_states

    def _fetch_requested_state_dict_to_client(self, client_id: int) -> TensorDict:
        """
        Retrieves the specific model weights requested by a target client.

        In asynchronous mode, fetches the specific historical version requested.
        In synchronous mode, fetches the latest global model directly from the server.

        Args:
            client_id (int): The unique client identifier.

        Returns:
            TensorDict: The state dictionary of the requested model version.
        """
        if self.async_states is not None:
            return self.async_states.fetch_historical_version_requested_by_client(
                cid=client_id
            )
        else:
            return self.server.global_model_dict

    def _async_post_local_update_book_keeping(self, client_id: int) -> None:
        """
        Overwrites the version requested by a client with the current global model version.

        Args:
            client_id (int): The unique client identifier.
        """
        if self.async_states is None:
            return

        self.async_states.update_version_requested_by_client(
            cid=client_id, requested_version=self.server.current_version
        )

    def _async_post_global_update_book_keeping(self) -> None:
        """
        Updates the simulation's historical model artifact during asynchronous training.

        Registers the newly aggregated global model into the centralized history database
        to provision future stale clients.
        """
        if self.async_states is None:
            return

        self.async_states.add_new_global_model_to_history(
            version=self.server.current_version,
            model_dict=self.server.global_model_dict,
        )

    def _handle_external_files_post_global_update(
        self, global_update_performed: bool
    ) -> None:
        """
        Handles disk I/O and external logging after a global update cycle.

        Logs metrics if a global update occurred and optionally saves the newly
        aggregated model if it achieved a new best accuracy.

        Args:
            global_update_performed (bool): Flag indicating if the server performed a
                global update during this step.
        """
        if not global_update_performed:
            return

        self._update_logger_post_global_update()

        self.checkpoint_manager.save_best(
            self.server.global_model_dict,
            current_acc=self.server.current_acc,
            best_acc=self.server.best_acc,
        )

        self.metrics_logger.log(
            global_idx=self.global_idx,
            loss=self.server.current_loss,
            accuracy=self.server.current_acc,
            sim_time=self.sim_time,
        )

    def _process_incoming_clients(self) -> None:
        """Handles the model dispatch, local training, and aggregation for active clients."""
        for client_id in self.incoming_clients:
            client: Client = self.clients[client_id]

            requested_state_dict = self._fetch_requested_state_dict_to_client(client_id)

            client_update = client.compute_update(
                model_shell=self.model_shell,
                device=self.device,
                global_idx=self.global_idx,
                requested_state_dict=requested_state_dict,
            )

            self._async_post_local_update_book_keeping(client_id)
            self.server.aggregate_update(client_update)

    def _process_global_update(self) -> bool:
        """
        Executes and processes global model update and adds the updated model
        to history if communication is asynchronous.
        """
        global_update_performed = self.server.global_update(
            model_shell=self.model_shell,
            device=self.device,
            global_idx=self.global_idx,
        )

        self._async_post_global_update_book_keeping()

        return global_update_performed

    def _step(self) -> bool:
        """
        Executes a single discrete event step in the simulation timeline.

        Processes updates for all incoming clients at the current timestamp,
        triggers global server aggregation, and updates execution logs,
        metrics logs and best checkpoints (optional).

        Returns:
            bool: True if the step executed successfully, False if the simulation
                clock has reached its end.
        """
        if self.local_idx >= len(self.clock):
            return False

        self._process_incoming_clients()
        global_update_performed = self._process_global_update()
        self._handle_external_files_post_global_update(
            global_update_performed=global_update_performed,
        )

        # Increase event counter
        self.local_idx += 1

        return True

    def _update_logger_post_global_update(self) -> None:
        """
        Updates the simulation log with the latest test accuracy and test loss.

        Args:
            global_update_performed (bool): Flag indicating if the server performed a
                global update during this step.
            sim_time (float): The current elapsed time within the simulated environment.
        """
        avg_loss = self.server.current_loss
        accuracy = self.server.current_acc

        logger.info(
            f"Global Update | Event: {self.global_idx:6d} | Time: {self.sim_time:5.2f} | "
            f"Loss: {avg_loss:2.4f} | Acc: {accuracy:3.2f}%"
        )

    def run(self) -> None:
        """
        Executes the main continuous simulation loop.

        Steps through the simulation clock until the maximum iterations are reached,
        a real-world timeout is exceeded, or a user interrupt is detected. Manages
        periodic interval-based checkpointing during the run.
        """
        logger.info(f"Starting Simulation Loop From Event: {self.global_idx}")

        start_time = time.time()

        with self.metrics_logger:
            while self._step():
                sim_duration = time.time() - start_time

                if sim_duration >= self.timeout:
                    logger.warning("Simulation Warning: Timeout exceeded.")
                    break

                if self.stop_requested:
                    logger.warning(
                        "Simulation Warning: Simulation interrupted by user or system."
                    )
                    break

                self.checkpoint_manager.save_latest(
                    server_state=self.server.state,
                    client_states=self._build_client_state_dicts(),
                    async_states=self.async_states,
                    global_idx=self.global_idx,
                    sim_duration=sim_duration,
                )

            self.checkpoint_manager.save_shutdown(
                server_state=self.server.state,
                client_states=self._build_client_state_dicts(),
                async_states=self.async_states,
                global_idx=self.global_idx,
            )
