import json
import shutil
from pathlib import Path
from typing import cast

import torch
from loguru import logger

from afl_sim.enums import CheckpointFile
from afl_sim.simulation.simulation_states import (
    AsyncStateManager,
    ClientMemoryStates,
)
from afl_sim.types import LatestMetadataSchema, ServerState, TensorDict

from .checkpoint_path_provider import CheckpointPathProvider


def atomic_tensor_dict_write(
    data: TensorDict,
    file_path: Path,
) -> None:
    """
    Safely writes tensor data to disk utilizing an atomic replacement operation.

    Writes the tensor payload to a temporary file first, then atomically replaces the target file
    to prevent data corruption in the event of an execution interruption.

    Args:
        data (TensorDict): The tensor dictionary payload to be saved.
        file_path (Path): The final destination file path for the saved tensor data.

    Raises:
        RuntimeError: If the write operation fails, logging the error and cleaning up the temporary file before halting execution.
    """
    tmp_path = file_path.parent / ("tmp_" + file_path.name)

    try:
        torch.save(data, tmp_path)
        tmp_path.replace(file_path)

    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()

        raise RuntimeError(f"Critical checkpoint failure at {file_path}") from e


def load_tensor_dict(tensordict_path: Path) -> TensorDict:
    """
    Safely loads a serialized PyTorch tensor dictionary from disk to the CPU memory space.

    Employs 'weights_only=True' to ensure the strictly secure unpickling of PyTorch binaries.

    Args:
        tensordict_path (Path): The designated path to the PyTorch checkpoint file.

    Returns:
        TensorDict: The safely instantiated PyTorch dictionary.

    Raises:
        FileNotFoundError: If the specified tensor file path is unreachable or non-existent.
    """
    try:
        tensor_data = torch.load(tensordict_path, map_location="cpu", weights_only=True)

        return cast("TensorDict", tensor_data)

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Critical Error: The file '{tensordict_path}' from previous "
            "run could not be found. Simulation aborted."
        ) from error


def atomic_write_latest(
    server_state: ServerState,
    client_states: ClientMemoryStates | None,
    async_states: AsyncStateManager | None,
    global_idx: int,
    path_provider: CheckpointPathProvider,
) -> None:
    """
    Executes an atomic replacement of the entire checkpoint directory.

    Populates a temporary directory with the latest simulation states, then safely replaces the primary checkpoint directory to guarantee consistency.

    Args:
        server_state (ServerState): The current state of the central server.
        client_states (ClientMemoryStates | None): An object containing the memory states of all clients, or None if unavailable.
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
        global_idx (int): The current global iteration or round index.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    path_provider.tmp_dir.mkdir(parents=True, exist_ok=True)

    _save_all_latest_states_to_tmp_dir(
        server_state=server_state,
        client_states=client_states,
        async_states=async_states,
        global_idx=global_idx,
        path_provider=path_provider,
    )

    try:
        if path_provider.ckpt_dir.exists():
            shutil.rmtree(path_provider.ckpt_dir)

        path_provider.tmp_dir.replace(path_provider.ckpt_dir)
    except PermissionError as error:
        logger.error(
            f"Permission denied when overwriting {path_provider.ckpt_dir}. Ensure no "
            "other programs (e.g., TensorBoard, file explorer) are accessing it."
        )
        raise RuntimeError(
            "Failed to atomically replace checkpoint directory due to OS lock."
        ) from error


def _save_all_latest_states_to_tmp_dir(
    server_state: ServerState,
    client_states: ClientMemoryStates | None,
    async_states: AsyncStateManager | None,
    global_idx: int,
    path_provider: CheckpointPathProvider,
) -> None:
    """
    Aggregates and saves all current global simulation states to the temporary directory.

    Internal method that delegates the saving of metadata, server states, active client
    states, active model history versions, and model requests.

    Args:
        server_state (ServerState): The current state of the central server.
        client_states (ClientMemoryStates | None): An object containing the memory states of all clients, or None if unavailable.
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
        global_idx (int): The current global iteration or round index.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    _save_latest_metadata_to_tmp_dir(
        server_state=server_state,
        async_states=async_states,
        global_idx=global_idx,
        path_provider=path_provider,
    )
    _save_latest_server_dicts_to_tmp_dir(
        server_state=server_state,
        history_versions=async_states.model_history.version_list
        if async_states is not None
        else [],
        path_provider=path_provider,
    )
    _save_latest_client_dicts_to_tmp_dir(
        client_states=client_states, path_provider=path_provider
    )
    _save_latest_async_states_to_tmp_dir(
        async_states=async_states, path_provider=path_provider
    )

    _copy_best_checkpoint_to_tmp_dir(path_provider=path_provider)


def _save_latest_server_dicts_to_tmp_dir(
    server_state: ServerState,
    history_versions: list[int],
    path_provider: CheckpointPathProvider,
) -> None:
    """
    Saves the server's model weights and buffer states to the temporary directory.

    The model state is only saved if its current version is not already
    tracked in the active history versions, avoiding duplicate file writes.

    Args:
        server_state (ServerState): The current state of the central server.
        history_versions (list[int]): A list of active model version identifiers currently tracked in the model history.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    if server_state.current_version not in history_versions:
        torch.save(
            server_state.model_state,
            path_provider.get_path(CheckpointFile.SERVER_STATE, tmp=True),
        )

    torch.save(
        server_state.buffer,
        path_provider.get_path(CheckpointFile.SERVER_BUFFER, tmp=True),
    )


def _save_latest_metadata_to_tmp_dir(
    server_state: ServerState,
    async_states: AsyncStateManager | None,
    global_idx: int,
    path_provider: CheckpointPathProvider,
) -> None:
    """
    Extracts and saves the latest simulation metadata to the temporary directory.

    Args:
        server_state (ServerState): The current state of the central server.
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
        global_idx (int): The current global iteration or round index.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    metadata_dict = _extract_metadata(
        server_state=server_state, async_states=async_states, global_idx=global_idx
    )
    f_name = path_provider.get_path(CheckpointFile.LATEST_METADATA, tmp=True)
    with open(f_name, "w") as file:
        json_string = metadata_dict.model_dump_json(indent=4)
        file.write(json_string)


def _extract_metadata(
    server_state: ServerState,
    async_states: AsyncStateManager | None,
    global_idx: int,
) -> LatestMetadataSchema:
    """
    Extracts the current simulation metrics and state indices into a structured schema.

    Args:
        server_state (ServerState): The current state of the central server.
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
        global_idx (int): The current global iteration or round index.

    Returns:
        LatestMetadataSchema: A validated schema object containing the extracted metadata parameters.
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


def _save_latest_client_dicts_to_tmp_dir(
    client_states: ClientMemoryStates | None, path_provider: CheckpointPathProvider
) -> None:
    """
    Saves the local memory states for all registered clients to the temporary directory.

    Args:
        client_states (ClientMemoryStates | None): An object containing the memory states
            of all clients, or None if a memory-less strategy is used.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    if client_states is None:
        return

    for cid in client_states.client_ids:
        torch.save(
            client_states.get_client_mem_state(cid),
            path_provider.get_client_state_path(cid=cid, tmp=True),
        )


def _save_latest_async_states_to_tmp_dir(
    async_states: AsyncStateManager | None, path_provider: CheckpointPathProvider
) -> None:
    """
    Saves the asynchronous model histories and current client model requests to the temporary directory.

    This method automatically writes all active historical model versions to disk
    and triggers the cleanup process to remove any stale versions no longer tracked.

    Args:
        async_states (AsyncStateManager | None): The object tracking asynchronous simulation states, or None if synchronous communication.
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    if async_states is None:
        return

    active_versions = async_states.model_history.version_list

    for version in active_versions:
        orig_version_path = path_provider.get_history_version_path(version, tmp=False)
        tmp_version_path = path_provider.get_history_version_path(version, tmp=True)

        if not orig_version_path.exists():  # avoid saving existing versions twice
            torch.save(
                async_states.model_history.get_version(version),
                tmp_version_path,
            )
        else:
            shutil.copy2(orig_version_path, tmp_version_path)

    req_file = path_provider.get_path(CheckpointFile.MODEL_REQUESTS, tmp=True)
    with open(req_file, "w") as file:
        json.dump(async_states.model_requests.state_dict, file, indent=4)


def _copy_best_checkpoint_to_tmp_dir(path_provider: CheckpointPathProvider) -> None:
    """
    Transfers the existing best-performing model weights and metadata to the temporary directory.

    Ensures that historical records of the best model are preserved during an atomic directory replacement.

    Args:
        path_provider (CheckpointPathProvider): The object providing resolved file paths.
    """
    for best_file in (CheckpointFile.BEST_MODEL, CheckpointFile.BEST_METADATA):
        orig_path = path_provider.get_path(best_file, tmp=False)
        if orig_path.exists():
            shutil.copy2(orig_path, path_provider.get_path(best_file, tmp=True))
