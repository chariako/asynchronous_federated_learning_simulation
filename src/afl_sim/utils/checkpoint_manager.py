import json
from pathlib import Path
from typing import Literal, cast

import torch
from loguru import logger

from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncStateManager,
    ClientMemoryStates,
)
from afl_sim.types import LatestMetadataSchema, ServerState, TensorDict


class CheckpointManager:
    """
    Manages the saving and loading of simulation checkpoints and model weights.

    This class handles atomic file operations to prevent data corruption during
    checkpointing and tracks both the latest simulation state and the best-performing
    model weights.

    Attributes:
        ckpt_dir (Path): The root directory where all checkpoint files are stored.
        best_path (Path): Path to the best model weights.
        latest_metadata_path (Path): Path to the simulation metadata JSON.
        latest_server_model_path (Path): Path to the server's latest model state.
        latest_server_buffer_path (Path): Path to the server's latest buffer state.
        latest_model_requests_path (Path): Path to the latest client model requests JSON.
        best_acc (float): The highest test accuracy recorded so far.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initializes the CheckpointManager with necessary file paths.

        Args:
            checkpoint_dir (Path): The directory path where checkpoint files
                will be saved and loaded from.
        """
        self.ckpt_dir = checkpoint_dir

        self.best_path = self.ckpt_dir / "model_best.pt"
        self.latest_metadata_path = self.ckpt_dir / "latest_metadata.json"
        self.latest_server_model_path = self.ckpt_dir / "latest_server_state.pt"
        self.latest_server_buffer_path = self.ckpt_dir / "latest_server_buffer.pt"
        self.latest_model_requests_path = self.ckpt_dir / "latest_model_requests.json"

        self.best_acc = -1.0

    def _get_latest_client_state_filename(self, cid: int) -> Path:
        """
        Generates the file path for a specific client's state.

        Args:
            cid (int): The unique client identifier.

        Returns:
            Path: The target path for the client's state file.
        """
        return self.ckpt_dir / f"latest_client_{cid}_state.pt"

    def _get_latest_history_version_filename(self, version: int | Literal["*"]) -> Path:
        """
        Generates the file path or glob pattern for a model history version.

        Args:
            version (int | Literal["*"]): The specific version integer, or a wildcard
                string ("*") used for globbing operations.

        Returns:
            Path: The target path or pattern for the history version file.
        """
        return self.ckpt_dir / f"latest_history_version_{version}.pt"

    def _atomic_write(
        self,
        data: TensorDict | LatestMetadataSchema | AsyncClientModelRequests,
        file_path: Path,
    ) -> None:
        """
        Safely writes data to a file using an atomic replace operation.

        Writes the data to a temporary file first, then replaces the target file
        to prevent data corruption in the event of an interruption.

        Args:
            data (TensorDict | LatestMetadataSchema | AsyncClientModelRequests):
                The data payload to be saved. Determines the save method (JSON or torch).
            file_path (Path): The final destination path for the saved file.

        Raises:
            RuntimeError: If the write operation fails, logging the error and
                cleaning up the temporary file before halting execution.
        """
        tmp_path = file_path.parent / ("tmp_" + file_path.name)

        try:
            if isinstance(data, LatestMetadataSchema):
                with open(tmp_path, "w") as file:
                    json_string = data.model_dump_json(indent=4)
                    file.write(json_string)

            elif isinstance(data, AsyncClientModelRequests):
                with open(tmp_path, "w") as file:
                    json.dump(data.state_dict, file, indent=4)

            else:
                torch.save(data, tmp_path)

            tmp_path.replace(file_path)

        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

            raise RuntimeError(f"Critical checkpoint failure at {file_path}") from e

    def _extract_metadata(
        self,
        server_state: ServerState,
        async_states: AsyncStateManager | None,
        global_idx: int,
    ) -> LatestMetadataSchema:
        """
        Extracts the current simulation metrics and state indices into a structured schema.

        Args:
            server_state (ServerState): The current state of the central server.
            async_states (AsyncStateManager | None): The object tracking asynchronous
                simulation states, or None if tracking is disabled.
            global_idx (int): The current global iteration or round index.

        Returns:
            LatestMetadataSchema: A validated schema object containing the extracted
                metadata parameters.
        """
        return LatestMetadataSchema(
            global_idx=global_idx,
            best_acc=server_state.best_acc,
            current_acc=server_state.current_acc,
            current_version=server_state.current_version,
            current_server_count=server_state.current_count,
            history_version_list=async_states.model_history.version_list
            if async_states is not None
            else [],
        )

    def _save_latest_metadata(
        self,
        server_state: ServerState,
        async_states: AsyncStateManager | None,
        global_idx: int,
    ) -> None:
        """
        Extracts and atomically saves the latest simulation metadata to disk.

        Args:
            server_state (ServerState): The current state of the central server.
            async_states (AsyncStateManager | None): The object tracking asynchronous
                simulation states, or None if tracking is disabled.
            global_idx (int): The current global iteration or round index.
        """
        metadata_dict = self._extract_metadata(
            server_state=server_state, async_states=async_states, global_idx=global_idx
        )
        self._atomic_write(metadata_dict, self.latest_metadata_path)

    def _save_latest_server_dicts(
        self, server_state: ServerState, history_versions: list[int]
    ) -> None:
        """
        Saves the server's model weights and buffer states to disk.

        The model state is only saved if its current version is not already
        tracked in the active history versions, avoiding duplicate file writes.

        Args:
            server_state (ServerState): The current state of the central server.
            history_versions (list[int]): A list of active model version identifiers
                currently tracked in the model history.
        """
        if server_state.current_version not in history_versions:
            self._atomic_write(server_state.model_state, self.latest_server_model_path)
        self._atomic_write(server_state.buffer, self.latest_server_buffer_path)

    def _save_latest_client_dicts(
        self, client_states: ClientMemoryStates | None
    ) -> None:
        """
        Saves the local memory states for all registered clients to disk.

        Args:
            client_states (ClientMemoryStates | None): An object containing the memory
                states of all clients. Does nothing if None.
        """
        if client_states is not None:
            for cid in client_states.client_ids:
                self._atomic_write(
                    client_states.get_client_mem_state(cid),
                    self._get_latest_client_state_filename(cid),
                )

    def _save_latest_async_states(self, async_states: AsyncStateManager | None) -> None:
        """
        Saves the asynchronous model histories and current client model requests.

        This method automatically writes all active historical model versions to disk
        and triggers the cleanup process to remove any stale versions no longer tracked.

        Args:
            async_states (AsyncStateManager | None): The object tracking asynchronous
                simulation states. Does nothing if None.
        """
        if async_states is not None:
            active_versions = async_states.model_history.version_list
            for version in active_versions:
                self._atomic_write(
                    async_states.model_history.get_version(version),
                    self._get_latest_history_version_filename(version),
                )

            self._remove_unused_history_files(active_versions=active_versions)
            self._atomic_write(
                async_states.model_requests, self.latest_model_requests_path
            )

    def _remove_unused_history_files(self, active_versions: list[int]) -> None:
        """
        Scans the checkpoint directory and removes untracked model history files.

        Identifies saved model files by matching the target glob pattern and unlinks
        any file whose version identifier is not present in the active versions list.

        Args:
            active_versions (list[int]): A list of version identifiers that are actively
                tracked and should be retained on disk.
        """
        for file_path in self.ckpt_dir.glob(
            self._get_latest_history_version_filename("*").name
        ):
            version_str = file_path.stem.split("_")[-1]
            if int(version_str) not in active_versions:
                file_path.unlink()

    def save_latest(
        self,
        server_state: ServerState,
        client_states: ClientMemoryStates | None,
        async_states: AsyncStateManager | None,
        global_idx: int,
    ) -> None:
        """
        Saves the current global state of the simulation to the latest checkpoint.

        Saves metadata, server states, active client states, active model history
        versions, and model requests. Also automatically removes stale model
        history files that are no longer tracked.

        Args:
            server_state (ServerState): The current state of the central server.
            client_states (ClientMemoryStates | None): The memory states of all clients,
                if available.
            async_states (AsyncStateManager | None): The object tracking asynchronous
                simulation states (model history, model requests), if available.
            global_idx (int): The current global iteration or round index.
        """
        self._save_latest_metadata(
            server_state=server_state, async_states=async_states, global_idx=global_idx
        )
        self._save_latest_server_dicts(
            server_state=server_state,
            history_versions=async_states.model_history.version_list
            if async_states is not None
            else [],
        )
        self._save_latest_client_dicts(client_states=client_states)
        self._save_latest_async_states(async_states=async_states)

    def save_best(self, model_state_dict: TensorDict, current_acc: float) -> bool:
        """
        Saves the model weights and metadata if the test accuracy has improved.

        Args:
            model_state_dict (TensorDict): The state dictionary of the model weights.
            current_acc (float): The evaluation accuracy of the current model.

        Returns:
            bool: True if a new best performing model was saved.
        """
        if current_acc > self.best_acc:
            logger.info(
                f"New best accuracy: {self.best_acc:.2f}% -> {current_acc:.2f}%"
            )
            self.best_acc = current_acc

            best_metadata_dict = {"best_model_acc": current_acc}
            best_metadata_path = self.ckpt_dir / "best_metadata.json"

            with open(best_metadata_path, "w") as file:
                json.dump(best_metadata_dict, file, indent=4)

            self._atomic_write(model_state_dict, self.best_path)

            return True

        return False

    def load_latest_metadata(self) -> LatestMetadataSchema:
        """
        Loads and validates the simulation metadata from the latest checkpoint.

        Returns:
            LatestMetadataSchema: The validated Pydantic schema containing simulation metadata.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        try:
            with open(self.latest_metadata_path) as file:
                return LatestMetadataSchema.model_validate_json(file.read())

        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Critical Error: Metadata from previous "
                "run could not be found. Simulation aborted."
            ) from error

    def load_server_states(self) -> ServerState:
        """
        Loads the most recent server state from disk.

        Restores the server's model state, buffer, and internal tracking metrics
        including the best and current accuracy.

        Returns:
            ServerState: The reconstructed state object for the server.
        """
        metadata = self.load_latest_metadata()

        if metadata.current_version in metadata.history_version_list:
            model_state = self.load_history_version(metadata.current_version)
        else:
            model_state = self._load_tensorDict(self.latest_server_model_path)

        buffer = self._load_tensorDict(self.latest_server_buffer_path)

        self.best_acc = metadata.best_acc
        logger.success(f"Restored best accuracy tracking: {self.best_acc:.2f}%")

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
        Loads the memory state for a specific client.

        Args:
            cid (int): The unique client identifier.

        Returns:
            TensorDict: The saved tensor dictionary for the requested client.
        """
        return self._load_tensorDict(self._get_latest_client_state_filename(cid))

    def load_history_version(self, version: int) -> TensorDict:
        """
        Loads a specific historical version of the model weights.

        Args:
            version (int): The integer identifier of the model version to load.

        Returns:
            TensorDict: The tensor dictionary representing the requested model version.
        """
        return self._load_tensorDict(self._get_latest_history_version_filename(version))

    def load_model_requests(self, num_clients: int) -> AsyncClientModelRequests:
        """
        Loads and rebuilds the asynchronous model requests for all clients.

        Args:
            num_clients (int): The total number of clients expected in the simulation.

        Returns:
            AsyncClientModelRequests: The reconstructed client requests object.

        Raises:
            KeyError: If a client ID is missing from the saved requests file.
            FileNotFoundError: If the requests file does not exist.
            ValueError: If the requests JSON is malformed or cannot be decoded.
        """
        try:
            with open(self.latest_model_requests_path) as file:
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

    def _load_tensorDict(self, tensordict_path: Path) -> TensorDict:
        """
        Safely loads a PyTorch tensor dictionary from disk to the CPU.

        Utilizes weights_only=True for secure unpickling of PyTorch files.

        Args:
            tensordict_path (Path): The path to the PyTorch checkpoint file.

        Returns:
            TensorDict: The loaded PyTorch dictionary.

        Raises:
            FileNotFoundError: If the specified tensor file does not exist.
        """
        try:
            tensor_data = torch.load(
                tensordict_path, map_location="cpu", weights_only=True
            )

            return cast("TensorDict", tensor_data)

        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Critical Error: The file '{tensordict_path}' from previous "
                "run could not be found. Simulation aborted."
            ) from error
