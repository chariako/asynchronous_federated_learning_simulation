import time

from loguru import logger
from torch import device

from afl_sim.config import AppConfig
from afl_sim.core import Client, Server
from afl_sim.timing import (
    SimulationClock,
)
from afl_sim.types import (
    SimulationModel,
    TensorDict,
)
from afl_sim.utils import (
    CheckpointManager,
    MetricsLogger,
)

from .simulation_states import (
    AsyncStateManager,
    ClientMemoryStates,
)


class Simulation:
    """
    Orchestrates the federated learning simulation loop.

    This class manages the discrete-event progression of the simulation,
    coordinating the flow of models and updates between the central server
    and distributed clients. It handles both synchronous and asynchronous
    training workflows, periodic checkpointing, and metric logging.

    Attributes:
        config (AppConfig): The application configuration settings.
        local_idx (int): The internal counter for discrete simulation events.
        metrics_logger (MetricsLogger): The utility responsible for logging performance metrics.
        device (device): The PyTorch device (CPU/GPU) utilized for model operations.
        clock (SimulationClock): The timekeeping mechanism mapping events to simulated time and incoming clients.
        server (Server): The centralized federated learning server.
        clients (list[Client]): The collection of all participating client nodes.
        checkpoint_manager (CheckpointManager): The utility for saving and loading states.
        last_checkpoint_time (float): The real-world timestamp of the last saved checkpoint.
        stop_requested (bool): Flag indicating if an early termination (e.g., Ctrl+C) was triggered.
        model_shell (SimulationModel): The underlying PyTorch model architecture.
        async_states (AsyncStateManager | None): The centralized artifact database storing simulation states (async only).
    """

    def __init__(
        self,
        config: AppConfig,
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
            config (AppConfig): Application configuration parameters.
            metrics_logger (MetricsLogger): Logger for simulation metrics.
            checkpoint_manager (CheckpointManager): Manager for I/O state saving.
            device (device): Target hardware device for tensor operations.
            server (Server): The initialized federated server.
            clients (list[Client]): The initialized list of client nodes.
            clock (SimulationClock): The simulation timekeeper.
            model_shell (SimulationModel): The base PyTorch model architecture.
            async_states (AsyncStateManager | None): The centralized artifact database storing simulation states (async only).
        """
        self.config = config
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

    def _build_client_state_dicts(self) -> ClientMemoryStates | None:
        """
        Aggregates the internal memory states of all participating clients.

        Returns:
            ClientMemoryStates | None: An object containing all client memory states,
                or None if the configured training strategy does not utilize client memory.
        """
        if not self.config.mem_strategy.type.has_memory:
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
            client_id (int): The unique identifier of the requesting client.

        Returns:
            TensorDict: The state dictionary of the requested model version.
        """
        if self.async_states is not None:
            return self.async_states.fetch_historical_version_requested_by_client(
                cid=client_id
            )
        else:
            return self.server.global_model_dict

    def _async_client_book_keeping(self, client_id: int) -> None:
        """
        Overwrites the version requested by a client with the current global model version.

        Args:
            client_id (int): The unique identifier of the client being updated.
        """
        if self.async_states is not None:
            self.async_states.update_version_requested_by_client(
                cid=client_id, requested_version=self.server.current_version
            )

    def _async_server_book_keeping(self) -> None:
        """
        Updates the simulation's historical model artifact during asynchronous training.

        Registers the newly aggregated global model into the centralized history database
        to provision future stale clients.
        """
        if self.async_states is not None:
            self.async_states.add_new_global_model_to_history(
                version=self.server.current_version,
                model_dict=self.server.global_model_dict,
            )

    def _handle_external_files_post_global_update(
        self, global_update_performed: bool, current_simulated_time: float
    ) -> None:
        """
        Handles disk I/O and external logging after a global update cycle.

        Logs metrics if a global update occurred and optionally saves the newly
        aggregated model if it achieved a new best accuracy.

        Args:
            global_update_performed (bool): Flag indicating if the server performed a
                global update during this step.
            current_simulated_time (float): The current time in the simulated environment.
        """
        # Save best checkpoint if applicable
        if (
            self.config.checkpoints.keep_best
            and self.server.current_acc == self.server.best_acc
        ):
            self.checkpoint_manager.save_best(
                self.server.global_model_dict, current_acc=self.server.current_acc
            )

        # Update metrics logger
        if global_update_performed:
            self.metrics_logger.log(
                global_idx=self.global_idx,
                sim_time=current_simulated_time,
                loss=self.server.current_loss,
                accuracy=self.server.current_acc,
            )

    def _step(self) -> bool:
        """
        Executes a single discrete event step in the simulation timeline.

        Processes updates for all incoming clients at the current timestamp,
        triggers global server aggregation, and advances internal states.

        Returns:
            bool: True if the step executed successfully, False if the simulation
                clock has reached its end.
        """
        if self.local_idx >= len(self.clock):
            return False

        current_simulated_time = self.clock.local_idx_to_sim_time(self.local_idx)
        incoming_client_ids = self.clock.local_idx_to_incoming_clients(self.local_idx)

        for client_id in incoming_client_ids:
            client: Client = self.clients[client_id]

            # Pull requested model from history (async) or server (sync)
            requested_state_dict = self._fetch_requested_state_dict_to_client(
                client_id=client_id
            )

            # Train the requested model on local data and return update
            client_update = client.compute_update(
                model_shell=self.model_shell,
                device=self.device,
                global_idx=self.global_idx,
                requested_state_dict=requested_state_dict,
            )

            # Client book-keeping for async mode
            self._async_client_book_keeping(client_id=client_id)

            # Aggregate client update at server
            self.server.aggregate_updates(client_update)

        # Global update performed if buffer is full
        global_update_performed = self.server.global_update(
            model_shell=self.model_shell,
            device=self.device,
            global_idx=self.global_idx,
            sim_time=current_simulated_time,
        )

        # Server book-keeping for async mode
        self._async_server_book_keeping()

        self._handle_external_files_post_global_update(
            global_update_performed=global_update_performed,
            current_simulated_time=current_simulated_time,
        )

        # Increase event counter
        self.local_idx += 1

        return True

    def _save_checkpoint(self) -> None:
        """Instructs the CheckpointManager to save the current global simulation state."""
        self.checkpoint_manager.save_latest(
            server_state=self.server.state,
            client_states=self._build_client_state_dicts(),
            async_states=self.async_states,
            global_idx=self.global_idx,
        )

    def external_files_shutdown_update(self) -> None:
        """
        Saves a final checkpoint in the event of an interruption or termination
        and flushes the metrics log file.

        Guarantees that the simulation state is written to disk before the program
        exits due to a manual halt (Ctrl+C), a SIGKILL signal or completion.
        """
        logger.info(
            f"Saving shutdown checkpoint before global event: {self.global_idx}..."
        )
        self._save_checkpoint()
        logger.info("Flushing metrics log file...")
        self.metrics_logger.flush_log_file()

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
                current_time = time.time()

                if current_time - start_time >= self.config.simulation.timeout_seconds:
                    logger.warning("Timeout exceeded.")
                    break

                if self.stop_requested:
                    logger.warning("Simulation interrupted by user (Ctrl+C).")
                    break

                # --- Checkpointing ---
                time_since_last_ckpt = current_time - self.last_checkpoint_time
                if time_since_last_ckpt >= self.config.checkpoints.interval_seconds:
                    logger.info("Saving Checkpoint...")
                    self._save_checkpoint()
                    self.last_checkpoint_time = current_time
