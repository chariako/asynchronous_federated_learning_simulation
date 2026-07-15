from pathlib import Path

from loguru import logger

from afl_sim.config import AppConfig, AsyncStrategy, CommStrategyConfig
from afl_sim.core import Client, Server
from afl_sim.data import DataManager
from afl_sim.models import get_model
from afl_sim.timing import (
    get_clock,
)
from afl_sim.types import (
    SimulationModel,
)
from afl_sim.utils import (
    CheckpointManager,
    MetricsLogger,
    get_device,
)

from .simulation import Simulation
from .simulation_states import (
    AsyncClientModelRequests,
    AsyncModelHistory,
    AsyncStateManager,
)


def build_simulation(
    config: AppConfig, data_dir: Path, checkpoint_dir: Path, run_dir: Path, resume: bool
) -> Simulation:
    """
    Constructs and initializes the complete federated learning simulation environment.

    This factory function acts as the central orchestration point for setting up a run.
    It provisions the device hardware, instantiates the core data managers and loggers,
    builds the model architecture, and links the Server, Clients, and discrete-event Clock.
    It also seamlessly handles restoring the entire ecosystem from a saved state if
    resuming a prior execution.

    Args:
        config (AppConfig): The comprehensive configuration object defining simulation parameters.
        data_dir (Path): The directory path containing the federated datasets.
        checkpoint_dir (Path): The directory path where checkpoint files are saved or loaded from.
        run_dir (Path): The target directory for logging metrics and simulation outputs.
        resume (bool): Flag indicating whether to initialize a fresh simulation (False)
            or restore the state from the most recent checkpoint (True).

    Returns:
        Simulation: The fully initialized, ready-to-run Simulation orchestration object.
    """
    data_manager = DataManager(config=config, data_dir=data_dir)
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    device = get_device(config.simulation.device)
    logger.info(f"Simulation running on device: {device}")

    model = get_model(dataset=config.data.dataset, model_config=config.model)

    server = _initialize_server(
        model=model,
        config=config,
        data_manager=data_manager,
        resume=resume,
        checkpoint_manager=checkpoint_manager,
    )

    clients = _initialize_clients(
        model=model,
        config=config,
        data_manager=data_manager,
        resume=resume,
        checkpoint_manager=checkpoint_manager,
    )

    next_global_index = _get_next_global_index(
        resume=resume, checkpoint_manager=checkpoint_manager
    )

    async_states = _initialize_async_states(
        num_clients=config.simulation.num_clients,
        comm_strategy=config.comm_strategy,
        resume=resume,
        checkpoint_manager=checkpoint_manager,
        initial_model=model,
    )

    clock = get_clock(
        config=config,
        data_dir=data_dir,
        global_next_idx=next_global_index,
    )

    metrics_logger = _initialize_metrics_logger(
        run_dir=run_dir, resume=resume, checkpoint_manager=checkpoint_manager
    )

    return Simulation(
        config=config,
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
        server=server,
        clients=clients,
        clock=clock,
        model_shell=model,
        async_states=async_states,
    )


def _initialize_server(
    model: SimulationModel,
    config: AppConfig,
    data_manager: DataManager,
    resume: bool,
    checkpoint_manager: CheckpointManager,
) -> Server:
    """
    Initializes the central federated learning server.

    Provisions the server with the global model architecture, evaluation dataloader,
    and aggregation parameters. If resuming, it restores the server's previous weights,
    buffers, and internal metrics from the checkpoint manager.

    Args:
        model (SimulationModel): The underlying PyTorch model architecture.
        config (AppConfig): The application configuration parameters.
        data_manager (DataManager): The utility for accessing global evaluation datasets.
        resume (bool): Flag indicating whether to restore the server from a checkpoint.
        checkpoint_manager (CheckpointManager): The utility for loading checkpointed states.

    Returns:
        Server: The initialized server node.
    """
    server = Server(
        model=model,
        test_loader=data_manager.get_evaluation_dataloader(),
        aggregation_goal=config.comm_strategy.agg_target,
        num_clients=config.simulation.num_clients,
        reset_buffer=config.mem_strategy.type.requires_buffer_reset,
        base_seed=config.simulation.torch_seed,
    )

    if resume:
        server.load_state_dict(checkpoint_manager.load_server_states())

    return server


def _initialize_clients(
    model: SimulationModel,
    config: AppConfig,
    data_manager: DataManager,
    resume: bool,
    checkpoint_manager: CheckpointManager,
) -> list[Client]:
    """
    Initializes the distributed client nodes for the simulation.

    Generates the specified number of clients, provisioning each with its unique
    local dataset partition, statistical weight, and optimization configuration.
    If resuming and the memory strategy is stateful, it restores each client's
    internal memory variables.

    Args:
        model (SimulationModel): The base PyTorch model architecture.
        config (AppConfig): The application configuration parameters.
        data_manager (DataManager): The utility for accessing client-specific dataset partitions.
        resume (bool): Flag indicating whether to restore client states from a checkpoint.
        checkpoint_manager (CheckpointManager): The utility for loading checkpointed memory.

    Returns:
        list[Client]: A list containing all fully initialized client objects.
    """
    clients = [
        Client(
            client_id=i,
            initial_model=model,
            dataloader=data_manager.get_client_dataloader(client_id=i),
            weight=data_manager.get_client_weight(client_id=i),
            optim_config=config.optimization,
            memory_strategy=config.mem_strategy,
            base_seed=config.simulation.torch_seed,
        )
        for i in range(config.simulation.num_clients)
    ]

    if resume and config.mem_strategy.type.has_memory:
        for client in clients:
            client.load_mem_state_dict(
                checkpoint_manager.load_client_memory_state(client.client_id)
            )

    return clients


def _get_next_global_index(resume: bool, checkpoint_manager: CheckpointManager) -> int:
    """
    Determines the starting index for the global simulation clock.

    Args:
        resume (bool): Flag indicating whether to resume from a checkpoint.
        checkpoint_manager (CheckpointManager): The utility containing historical metadata.

    Returns:
        int: The global event index to begin or resume the simulation loop from.
            Returns 0 for a fresh run.
    """
    if resume:
        latest_metadata = checkpoint_manager.load_latest_metadata()
        return latest_metadata.global_idx

    return 0


def _initialize_async_states(
    num_clients: int,
    comm_strategy: CommStrategyConfig,
    resume: bool,
    checkpoint_manager: CheckpointManager,
    initial_model: SimulationModel,
) -> AsyncStateManager | None:
    """
    Initializes the unified state manager for asynchronous simulations.

    If the communication strategy is asynchronous, it either restores the historical
    models and client requests from a checkpoint (if resuming) or creates fresh
    instances initialized with the starting model. Returns None for synchronous runs.

    Args:
        num_clients (int): The total number of participating clients.
        comm_strategy (CommStrategyConfig): The communication strategy configuration.
        resume (bool): Flag indicating whether to restore state from a checkpoint.
        checkpoint_manager (CheckpointManager): The utility for loading checkpointed states.
        initial_model (SimulationModel): The starting model to use if not resuming.

    Returns:
        AsyncStateManager | None: The initialized unified states manager, or None
            if using a synchronous communication strategy.
    """

    if isinstance(comm_strategy, AsyncStrategy):
        if resume:
            latest_metadata = checkpoint_manager.load_latest_metadata()
            version_list = latest_metadata.history_version_list
            model_history = AsyncModelHistory(initial_model_dict=None)
            for version in version_list:
                model_history.add_version(
                    version, checkpoint_manager.load_history_version(version)
                )
            return AsyncStateManager(
                model_history=model_history,
                model_requests=checkpoint_manager.load_model_requests(
                    num_clients=num_clients
                ),
            )

        return AsyncStateManager(
            model_history=AsyncModelHistory(
                initial_model_dict=initial_model.state_dict()
            ),
            model_requests=AsyncClientModelRequests(num_clients),
        )

    return None


def _initialize_metrics_logger(
    run_dir: Path, resume: bool, checkpoint_manager: CheckpointManager
) -> MetricsLogger:
    """
    Initializes the simulation's metrics logger.

    Provisions the logger in the specified run directory. If the simulation is
    resuming from a previous state, it queries the checkpoint manager for the
    latest global index and trims the metrics file to remove any orphaned entries
    recorded after that index, preventing duplicate logs.

    Args:
        run_dir (Path): The directory path where the metrics log file will be stored.
        resume (bool): Flag indicating whether to restore the state from a checkpoint.
        checkpoint_manager (CheckpointManager): The utility for loading historical
            metadata to determine the correct resume index.

    Returns:
        MetricsLogger: The fully initialized metrics logger.
    """
    metrics_logger = MetricsLogger(run_dir=run_dir)

    if resume:
        latest_metadata = checkpoint_manager.load_latest_metadata()
        metrics_logger.trim_history(next_global_idx=latest_metadata.global_idx)

    return metrics_logger
