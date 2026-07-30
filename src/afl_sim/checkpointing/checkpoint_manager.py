import json
from pathlib import Path

from loguru import logger

from afl_sim.config import CheckpointConfig
from afl_sim.enums import CheckpointFile
from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncStateManager,
    ClientMemoryStates,
)
from afl_sim.types import LatestMetadataSchema, ServerState, TensorDict

from .checkpoint_helpers import (
    atomic_tensor_dict_write,
    atomic_write_latest,
    load_tensor_dict,
)
from .checkpoint_path_provider import CheckpointPathProvider


class CheckpointManager:
    """
    Manages the saving and loading of simulation checkpoints and model weights.

    This class handles atomic file operations to prevent data corruption during
    checkpointing and tracks both the latest simulation state and the best-performing
    model weights.

    Attributes:
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
        last_checkpoint_time (float): The timestamp recording when the most recent latest checkpoint was successfully saved.
        checkpoint_config (CheckpointConfig): Configuration parameters governing checkpointing behavior and intervals.
    """

    def __init__(self, checkpoint_dir: Path, checkpoint_config: CheckpointConfig):
        """
        Initializes the CheckpointManager with the requisite file paths and configuration.

        Args:
            checkpoint_dir (Path): The root directory where all checkpoint files are stored.
            checkpoint_config (CheckpointConfig): Configuration parameters governing checkpointing behavior and intervals.
        """
        self.path_provider = CheckpointPathProvider(checkpoint_dir)
        self.last_checkpoint_time = 0.0
        self.checkpoint_config = checkpoint_config

    def save_latest(
        self,
        server_state: ServerState,
        client_states: ClientMemoryStates | None,
        async_states: AsyncStateManager | None,
        global_idx: int,
        sim_duration: float,
    ) -> None:
        """
        Saves the current global state of the simulation to the latest checkpoint.

        Saves metadata, server states, active client states, active model history
        versions, and model requests. Automatically removes stale model
        history files that are no longer tracked.

        Args:
            server_state (ServerState): The current state of the central server.
            client_states (ClientMemoryStates | None): An object containing the memory states of all clients, or None if a memory-less strategy is used.
            async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
            global_idx (int): The current global iteration or round index.
            sim_duration (float): The current elapsed duration of the simulation in seconds.
        """
        time_since_last = sim_duration - self.last_checkpoint_time

        if time_since_last < self.checkpoint_config.interval_seconds:
            return

        atomic_write_latest(
            server_state=server_state,
            client_states=client_states,
            async_states=async_states,
            global_idx=global_idx,
            path_provider=self.path_provider,
        )

        self.last_checkpoint_time = sim_duration
        logger.success(
            f"Latest resumable checkpoint successfully saved before global event: {global_idx}"
        )

    def save_shutdown(
        self,
        server_state: ServerState,
        client_states: ClientMemoryStates | None,
        async_states: AsyncStateManager | None,
        global_idx: int,
    ) -> None:
        """
        Saves the final global state of the simulation during a shutdown event.

        Saves metadata, server states, active client states, active model history
        versions, and model requests immediately prior to simulation termination.

        Args:
            server_state (ServerState): The current state of the central server.
            client_states (ClientMemoryStates | None): An object containing the memory states of all clients, or None if a memory-less strategy is used.
            async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
            global_idx (int): The current global iteration or round index.
        """
        atomic_write_latest(
            server_state=server_state,
            client_states=client_states,
            async_states=async_states,
            global_idx=global_idx,
            path_provider=self.path_provider,
        )
        logger.success(
            f"Shutdown checkpoint successfully saved before global event: {global_idx}"
        )

    def save_best(
        self, model_state_dict: TensorDict, current_acc: float, best_acc: float
    ) -> None:
        """
        Persists the model weights and associated metadata if the empirical test accuracy has improved.

        Args:
            model_state_dict (TensorDict): The state dictionary containing the model weights.
            current_acc (float): The evaluation accuracy of the current model.
            best_acc (float): The maximum evaluation accuracy recorded historically.
        """
        if not self.checkpoint_config.keep_best:
            return

        if current_acc < best_acc:
            return

        best_metadata_dict = {"best_model_acc": current_acc}

        with open(
            self.path_provider.get_path(CheckpointFile.BEST_METADATA), "w"
        ) as file:
            json.dump(best_metadata_dict, file, indent=4)

        atomic_tensor_dict_write(
            model_state_dict, self.path_provider.get_path(CheckpointFile.BEST_MODEL)
        )

    def load_latest_metadata(self) -> LatestMetadataSchema:
        """
        Loads and rigorously validates the simulation metadata from the latest checkpoint file.

        Returns:
            LatestMetadataSchema: The validated schema object encompassing simulation metadata.

        Raises:
            FileNotFoundError: If the metadata file is absent from the checkpoint directory.
        """
        try:
            with open(
                self.path_provider.get_path(CheckpointFile.LATEST_METADATA)
            ) as file:
                return LatestMetadataSchema.model_validate_json(file.read())

        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Critical Error: Metadata from previous "
                "run could not be found. Simulation aborted."
            ) from error

    def load_server_states(self) -> ServerState:
        """
        Restores the most recent server state from the designated checkpoint directory.

        Reconstructs the central server's model state, memory buffer, and internal tracking metrics,
        incorporating the best and current accuracy evaluations.

        Returns:
            ServerState: The structurally reconstructed state object for the server.
        """
        metadata = self.load_latest_metadata()

        if metadata.current_version in metadata.history_version_list:
            model_state = self.load_history_version(metadata.current_version)
        else:
            model_state = load_tensor_dict(
                self.path_provider.get_path(CheckpointFile.SERVER_STATE)
            )

        buffer = load_tensor_dict(
            self.path_provider.get_path(CheckpointFile.SERVER_BUFFER)
        )

        return ServerState(
            model_state=model_state,
            buffer=buffer,
            current_count=metadata.current_server_count,
            best_acc=metadata.best_acc,
            current_acc=metadata.current_acc,
            current_version=metadata.current_version,
        )

    def load_client_memory_state(self, cid: int) -> TensorDict:
        """
        Retrieves the saved memory state for a designated client.

        Args:
            cid (int): The unique client identifier.

        Returns:
            TensorDict: The persistent tensor dictionary corresponding to the requested client.
        """
        return load_tensor_dict(self.path_provider.get_client_state_path(cid))

    def load_history_version(self, version: int) -> TensorDict:
        """
        Retrieves a specific historical version of the global model weights.

        Args:
            version (int): The integer identifier specifying the target model version.

        Returns:
            TensorDict: The tensor dictionary encapsulating the requested model version.
        """
        return load_tensor_dict(self.path_provider.get_history_version_path(version))

    def load_model_requests(self, num_clients: int) -> AsyncClientModelRequests:
        """
        Loads and reconstructs the asynchronous model requests across all registered clients.

        Args:
            num_clients (int): The aggregate number of clients configured in the simulation.

        Returns:
            AsyncClientModelRequests: The systematically reconstructed client requests object.

        Raises:
            KeyError: If a client identifier is demonstrably missing from the saved requests file.
            FileNotFoundError: If the requests file is absent from the checkpoint directory.
            ValueError: If the requests JSON payload is malformed or fundamentally unparseable.
        """
        try:
            with open(
                self.path_provider.get_path(CheckpointFile.MODEL_REQUESTS)
            ) as file:
                requests_dict = json.load(file)

            model_requests = AsyncClientModelRequests(num_clients)

            for cid in range(num_clients):
                if str(cid) in requests_dict:
                    model_requests.update_client(
                        cid=cid, version=requests_dict[str(cid)]
                    )
                else:
                    raise KeyError(f"Model request missing for client: {cid}")

            return model_requests

        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Critical Error: Model request file from previous "
                "run could not be found. Simulation aborted."
            ) from error

        except json.JSONDecodeError as error:
            raise ValueError(
                "Critical Error: The model request file is corrupted and cannot be parsed."
            ) from error
