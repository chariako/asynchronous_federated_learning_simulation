import time
from pathlib import Path
from typing import Any

from loguru import logger

from afl_sim.config import AppConfig
from afl_sim.core import Client, Server
from afl_sim.data import DataManager
from afl_sim.models import get_model
from afl_sim.timing import (
    ClockData,
    get_clock,
    get_clock_length,
    read_clock_event,
)
from afl_sim.utils import CheckpointManager, MetricsLogger, get_device


class Simulation:
    def __init__(
        self,
        config: AppConfig,
        data_manager: DataManager,
        clock: ClockData,
        run_dir: Path,
        checkpoint_dir: Path,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Metrics
        self.metrics_logger = MetricsLogger(run_dir=run_dir)

        # Device
        self.device = get_device(config.simulation.device)
        logger.info(f"Simulation running on device: {self.device}")

        # Clock
        self.clock = clock
        self.len_clock = get_clock_length(clock)

        # Server & Model
        logger.info("Initializing model & server...")
        raw_model = get_model(dataset=config.data.dataset, model_config=config.model)

        self.server = Server(
            model=raw_model,
            test_loader=data_manager.get_evaluation_dataloader(),
            device=self.device,
            aggregation_goal=config.comm_strategy.agg_target,
            num_clients=config.simulation.num_clients,
            reset_buffer=config.mem_strategy.type.requires_buffer_reset,
        )

        # Model shell for local client training
        shell_model = self.server.get_shell_model()

        # Clients
        logger.info("Initializing clients...")
        self.clients: list[Client] = [
            Client(
                client_id=i,
                initial_model=shell_model,
                dataloader=data_manager.get_client_dataloader(client_id=i),
                weight=data_manager.get_client_weight(client_id=i),
                optim_config=config.optimization,
                memory_strategy=config.mem_strategy,
            )
            for i in range(config.simulation.num_clients)
        ]

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir)
        self.event_idx = 0
        self.last_checkpoint_time = time.time()

    def state_dict(self) -> dict[str, Any]:
        """
        Orchestrates the saving strategy.
        Decides which client data to save based on config.
        """
        # Always save all server states
        state = {"server": self.server.get_state_dict(), "event_idx": self.event_idx}

        clients_data = {}
        save_stale = self.config.comm_strategy.type == "async"
        save_memory = self.config.mem_strategy.type.has_memory

        for cid, client in enumerate(self.clients):
            c_data = {}

            # Async: Save stale states
            if save_stale:
                c_data["stale_state"] = client.get_stale_state_dict()

            # Memory: Save if enabled
            if save_memory:
                mem = client.get_memory_dict()
                if mem:
                    c_data["memory"] = mem

            if c_data:
                clients_data[f"client_{cid}"] = c_data

        state["clients"] = clients_data
        return state

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Hydrates simulation from payload."""
        logger.info("Loading Server State...")
        self.server.load_state_dict(payload["server"])

        logger.info("Loading Client States...")
        client_states = payload["clients"]

        for cid, client in enumerate(self.clients):
            key = f"client_{cid}"

            if key in client_states:
                client.load_state_dict(client_states[key])

            # If Sync, initialize clients with the current server model
            if self.config.comm_strategy.type == "sync":
                client.receive_global_model(self.server.get_global_state_dict())

        # Sync with the server's best accuracy.
        self.checkpoint_manager.update_best_accuracy(payload["server"]["best_acc"])

    def step(self) -> bool:
        """Executes one clock tick."""
        if self.event_idx >= self.len_clock:
            return False

        current_simulated_time, incoming_client_ids = read_clock_event(
            self.clock, self.event_idx
        )

        # --- Client Processing ---
        for client_id in incoming_client_ids:
            if client_id == -1:
                continue

            client: Client = self.clients[client_id]

            # Compute
            client_update = client.compute_update(
                self.server.get_shell_model(), self.device
            )
            self.server.aggregate_updates(client_update)

            # Incoming client receives current global model
            if self.config.comm_strategy.type == "async":
                client.receive_global_model(self.server.get_global_state_dict())

        # --- Global Update ---
        # did_update returns True if a global update was performed
        did_update = self.server.global_update()

        if did_update:
            # Update metrics logger
            avg_loss = self.server.get_current_loss()
            accuracy = self.server.get_current_accuracy()
            self.metrics_logger.log(
                event_idx=self.event_idx,
                sim_time=current_simulated_time,
                loss=avg_loss,
                accuracy=accuracy,
            )

            logger.info(
                f"Global Update | Event: {self.event_idx:6d} | Time: {current_simulated_time:5.2f} | "
                f"Loss: {avg_loss:2.4f} | Acc: {accuracy:3.2f}%"
            )

            # Update best model checkpoint
            if self.config.checkpoints.keep_best:
                self.checkpoint_manager.save_best(
                    self.server.get_global_state_dict(), current_acc=accuracy
                )

        # Get next round's clients for sync mode
        if self.config.comm_strategy.type == "sync":
            self._sync_next_round_clients()

        # --- Checkpointing ---
        current_time = time.time()
        time_since_last_ckpt = current_time - self.last_checkpoint_time
        if time_since_last_ckpt >= self.config.checkpoints.interval_seconds:
            logger.info("Saving Checkpoint...")
            self.checkpoint_manager.save_latest(self.state_dict(), self.event_idx + 1)
            self.last_checkpoint_time = current_time

        self.event_idx += 1
        return True

    def _sync_next_round_clients(self) -> None:
        """
        Updates the clients participating in the next round with the new global model.
        Only used in Synchronous strategies.
        """
        if self.event_idx + 1 >= self.len_clock:
            return

        _, outgoing_client_ids = read_clock_event(self.clock, self.event_idx + 1)
        global_state = self.server.get_global_state_dict()
        for cid in outgoing_client_ids:
            if cid == -1:
                continue
            self.clients[cid].receive_global_model(global_state)

    def save_shutdown_checkpoint(self) -> None:
        """Saves shutdown checkpoint in case of interruption or termination."""
        logger.info(f"Saving shutdown checkpoint before event: {self.event_idx}...")
        self.checkpoint_manager.save_latest(self.state_dict(), self.event_idx)

    def resume(self) -> None:
        """Resume existing simulation from folder."""
        self.event_idx, payload = self.checkpoint_manager.load_latest()

        # Align metrics logger with new starting point
        self.metrics_logger.trim_history(resume_from_idx=self.event_idx)
        self.load_state_dict(payload)

    def run(self) -> None:
        """
        Executes the main simulation loop.
        """
        logger.info(f"Starting Simulation Loop From Event: {self.event_idx}")
        start_time = time.time()

        while self.step():
            if time.time() - start_time >= self.config.simulation.timeout_seconds:
                logger.warning("Timeout exceeded.")
                break


def build_simulation(
    config: AppConfig,
    run_dir: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> Simulation:
    data_manager = DataManager(
        config=config,
        data_dir=data_dir,
        visualize=config.visualization.visualize_data_split,
    )
    clock = get_clock(config=config, data_dir=data_dir)

    sim = Simulation(
        config=config,
        data_manager=data_manager,
        clock=clock,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
    )
    if resume:
        logger.info(f"Loading checkpoint payload from {checkpoint_dir.name}...")
        sim.resume()
    return sim
