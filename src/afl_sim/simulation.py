import time
from pathlib import Path

from loguru import logger

from afl_sim.config import AppConfig
from afl_sim.core import Client, Server
from afl_sim.data import DataManager
from afl_sim.models import get_model
from afl_sim.timing import (
    SimulationClock,
    get_clock,
)
from afl_sim.types import LatestCheckpoint, SimulationState
from afl_sim.utils import (
    CheckpointManager,
    MetricsLogger,
    get_device,
)


class Simulation:
    def __init__(
        self,
        config: AppConfig,
        data_manager: DataManager,
        checkpoint_manager: CheckpointManager,
        clock: SimulationClock,
        run_dir: Path,
    ):
        self.config = config

        # Metrics
        self.metrics_logger = MetricsLogger(run_dir=run_dir)

        # Device
        self.device = get_device(config.simulation.device)
        logger.info(f"Simulation running on device: {self.device}")

        # Clock
        self.clock = clock
        self.local_idx = 0  # local index

        # Server & Model
        raw_model = get_model(dataset=config.data.dataset, model_config=config.model)

        self.server = Server(
            model=raw_model,
            test_loader=data_manager.get_evaluation_dataloader(),
            device=self.device,
            aggregation_goal=config.comm_strategy.agg_target,
            num_clients=config.simulation.num_clients,
            reset_buffer=config.mem_strategy.type.requires_buffer_reset,
            base_seed=config.simulation.torch_seed,
        )

        # Model shell for local client training
        shell_model = self.server.get_shell_model()

        # Clients
        self.clients: list[Client] = [
            Client(
                client_id=i,
                initial_model=shell_model,
                dataloader=data_manager.get_client_dataloader(client_id=i),
                weight=data_manager.get_client_weight(client_id=i),
                optim_config=config.optimization,
                memory_strategy=config.mem_strategy,
                base_seed=config.simulation.torch_seed,
            )
            for i in range(config.simulation.num_clients)
        ]

        logger.success("Clients & server successfully initialized.")

        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_time = time.time()

        self.stop_requested = False

    @property
    def global_idx(self) -> int:
        return self.clock.local_to_global_idx(self.local_idx)

    @property
    def state(self) -> SimulationState:
        save_stale = self.config.comm_strategy.type == "async"
        save_memory = self.config.mem_strategy.type.has_memory
        client_states = {
            f"client_{cid}": client.state for cid, client in enumerate(self.clients)
        }

        return SimulationState(
            server=self.server.state,
            clients={
                k: {
                    "memory": v["memory"] if save_memory else None,
                    "stale_state": v["stale_state"] if save_stale else None,
                }
                for k, v in client_states.items()
            },
        )

    def load_state(self, state: SimulationState) -> None:
        logger.info("Loading Server & Client States...")

        self.server.load_state_dict(state["server"])

        client_states = state["clients"]

        for cid, client in enumerate(self.clients):
            key = f"client_{cid}"

            if key in client_states:
                client.load_state_dict(client_states[key])

            if self.config.comm_strategy.type == "sync":
                client.receive_global_model(self.server.get_global_state_dict())

        self.checkpoint_manager.update_best_accuracy(acc=state["server"]["best_acc"])

    def step(self) -> bool:
        """Executes one clock tick."""
        if self.local_idx >= self.clock.length:
            return False

        current_simulated_time = self.clock.local_idx_to_sim_time(self.local_idx)
        incoming_client_ids = self.clock.local_idx_to_incoming_clients(self.local_idx)

        # --- Client Processing ---
        for client_id in incoming_client_ids:
            client: Client = self.clients[client_id]

            # Compute
            client_update = client.compute_update(
                self.server.get_shell_model(),
                self.device,
                self.global_idx,
            )
            self.server.aggregate_updates(client_update)

            # Incoming client receives current global model
            if self.config.comm_strategy.type == "async":
                client.receive_global_model(self.server.get_global_state_dict())

        # --- Global Update ---
        self.server.global_update(self.global_idx)

        if self.server.just_performed_global_update():
            # Update metrics logger
            self.server.evaluate()
            avg_loss = self.server.get_current_loss()
            accuracy = self.server.get_current_accuracy()
            self.metrics_logger.log(
                global_idx=self.global_idx,
                sim_time=current_simulated_time,
                loss=avg_loss,
                accuracy=accuracy,
            )

            logger.info(
                f"Global Update | Event: {self.global_idx:6d} | Time: {current_simulated_time:5.2f} | "
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

        self.local_idx += 1

        return True

    def _sync_next_round_clients(self) -> None:
        """
        Updates the clients participating in the next round with the new global model.
        Only used in Synchronous strategies.
        """
        if self.local_idx + 1 >= self.clock.length:
            return

        outgoing_client_ids = self.clock.local_idx_to_incoming_clients(
            self.local_idx + 1
        )
        global_state = self.server.get_global_state_dict()
        for cid in outgoing_client_ids:
            self.clients[cid].receive_global_model(global_state)

    def save_shutdown_checkpoint(self) -> None:
        """Saves shutdown checkpoint in case of interruption or termination."""
        logger.info(
            f"Saving shutdown checkpoint before global event: {self.global_idx}..."
        )
        self.checkpoint_manager.save_latest(self.state, self.global_idx)

    def resume(self, latest_checkpoint: LatestCheckpoint) -> None:
        """Resume existing simulation from folder."""

        global_idx = latest_checkpoint["global_next_event"]
        simulation_state = latest_checkpoint["simulation_state"]

        # Align metrics logger with new starting point
        self.metrics_logger.trim_history(resume_from_idx=global_idx)
        self.load_state(simulation_state)

    def run(self) -> None:
        """
        Executes the main simulation loop.
        """
        logger.info(f"Starting Simulation Loop From Event: {self.global_idx}")
        start_time = time.time()

        while self.step():
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
                self.checkpoint_manager.save_latest(self.state, self.global_idx)
                self.last_checkpoint_time = current_time


def build_simulation(
    config: AppConfig,
    run_dir: Path,
    data_dir: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> Simulation:
    next_idx = 0
    data_manager = DataManager(
        config=config,
        data_dir=data_dir,
        visualize=config.visualization.visualize_data_split,
        base_seed=config.simulation.torch_seed,
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    if resume:
        logger.info(f"Loading checkpoint payload from {checkpoint_dir.name}...")
        latest_checkpoint = checkpoint_manager.load_latest()
        next_idx = latest_checkpoint["global_next_event"]

    clock = get_clock(
        config=config,
        data_dir=data_dir,
        global_next_idx=next_idx,
    )

    sim = Simulation(
        config=config,
        data_manager=data_manager,
        checkpoint_manager=checkpoint_manager,
        clock=clock,
        run_dir=run_dir,
    )

    # Overwrite initialization with checkpoint
    if resume:
        sim.resume(latest_checkpoint)

    return sim
