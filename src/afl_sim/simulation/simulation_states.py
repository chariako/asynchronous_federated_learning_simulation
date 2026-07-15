from afl_sim.types import TensorDict


class AsyncModelHistory:
    """
    A structural artifact designed to simulate asynchronous staleness within a serial environment.

    In a distributed asynchronous architecture, clients receive models and train concurrently.
    To emulate this behavior during a sequential simulation on a single accelerator, this class
    maintains a centralized repository of historical global models. It ensures that during a
    client's sequential execution phase, the client is provisioned with the exact historical
    model state it was assigned at its simulated dispatch time, decoupling the server's current
    state from the client's local training state.

    Ensures that stored PyTorch tensors are securely detached and moved to the CPU
    to prevent GPU memory leaks (Out-Of-Memory errors) during long asynchronous runs.
    """

    def __init__(self, initial_model_dict: TensorDict | None) -> None:
        """
        Initializes the model history tracking database.

        Args:
            initial_model_dict (TensorDict | None): The starting model weights to
                initialize version 0. If None, the history starts empty.
        """
        if initial_model_dict is not None:
            self._history: dict[int, TensorDict] = {
                0: {
                    key: param.detach().to(device="cpu", copy=True)
                    for key, param in initial_model_dict.items()
                }
            }
        else:
            self._history = {}

    @property
    def version_list(self) -> list[int]:
        """
        Retrieves a list of all model versions currently stored in memory.

        Returns:
            list[int]: The active model version integers.
        """
        return list(self._history)

    def add_version(self, version: int, model_dict: TensorDict) -> None:
        """
        Adds a new model version to the history database.

        Safely detaches tensors and copies them to the CPU before storing.

        Args:
            version (int): The integer identifier for this model version.
            model_dict (TensorDict): The PyTorch state dictionary to store.
        """
        if version not in self._history:
            self._history[version] = {
                key: param.detach().to(device="cpu", copy=True)
                for key, param in model_dict.items()
            }

    def get_version(self, version: int) -> TensorDict:
        """
        Retrieves the state dictionary for a specific historical model version.

        Args:
            version (int): The integer identifier of the requested version.

        Returns:
            TensorDict: The requested PyTorch state dictionary.

        Raises:
            KeyError: If the requested version integer is not in the history.
        """
        if version not in self._history:
            raise KeyError(
                f"Requested model version {version} is not present in the history database."
            )

        return self._history[version]

    def refresh(self, version_list: set[int]) -> None:
        """
        Purges unused model versions from memory.

        Iterates through the stored history and removes any version that is no longer
        required by any active client to simulate staleness.

        Args:
            version_list (set[int]): A set of model versions currently requested
                by active clients.
        """
        for version in list(self._history.keys()):
            if version not in version_list:
                self._history.pop(version)


class AsyncClientModelRequests:
    """
    Tracks the historical model version assigned to each client upon server communication.

    Due to the serial nature of the simulation environment, true concurrent training is not
    feasible. This class serves as a tracking ledger, recording the specific global model
    version designated for each client. During the sequential execution loop, the simulation
    accesses this ledger to retrieve the appropriate historical model from the AsyncModelHistory,
    thereby accurately replicating asynchronous temporal staleness.
    """

    def __init__(self, num_clients: int) -> None:
        """
        Initializes the client request tracker.

        Args:
            num_clients (int): The total number of clients in the simulation.
                Initializes all clients to request version 0.
        """
        self._model_requests = {
            cid: 0 for cid in range(num_clients)
        }  # match with client_ids

    @property
    def state_dict(self) -> dict[int, int]:
        """
        Retrieves the raw mapping of all client IDs to their requested versions.

        Returns:
            dict[int, int]: A dictionary where keys are client IDs and values are
                model version integers.
        """
        return self._model_requests

    @property
    def version_list(self) -> set[int]:
        """
        Retrieves a unique set of all model versions currently being requested.

        Returns:
            set[int]: A mathematical set of active requested versions in O(1) format
                for fast lookup.
        """
        return set(self._model_requests.values())

    def update_client(self, cid: int, version: int) -> None:
        """
        Updates the requested model version for a specific client.

        Args:
            cid (int): The unique client identifier.
            version (int): The model version integer the client is now requesting.
        """
        self._model_requests[cid] = version

    def get_client_request(self, cid: int) -> int:
        """
        Retrieves the specific model version requested by a client.

        Args:
            cid (int): The unique client identifier.

        Returns:
            int: The model version integer currently requested by this client.

        Raises:
            KeyError: If the provided client ID is not registered in the ledger.
        """
        if cid not in self._model_requests:
            raise KeyError(
                f"Client ID {cid} is not registered in the model requests ledger."
            )

        return self._model_requests[cid]


class ClientMemoryStates:
    """Bundles internal memory state references from all clients before checkpointing."""

    def __init__(self) -> None:
        """Initializes an empty database for client memory states."""
        self._states: dict[int, TensorDict] = {}

    @property
    def client_ids(self) -> list[int]:
        """
        Retrieves a list of all client IDs that currently have saved memory states.

        Returns:
            list[int]: A list of active client identifiers.
        """
        return list(self._states.keys())

    def add_client_mem_state(self, client_id: int, mem_state: TensorDict) -> None:
        """
        Adds a client's internal memory state reference to the bundle.

        Args:
            client_id (int): The unique client identifier.
            mem_state (TensorDict): The state dictionary representing the client's
                local memory.
        """
        self._states[client_id] = mem_state

    def get_client_mem_state(self, client_id: int) -> TensorDict:
        """
        Retrieves the saved memory state for a specific client.

        Args:
            client_id (int): The unique client identifier.

        Returns:
            TensorDict: The stored state dictionary for the requested client.

        Raises:
            KeyError: If no memory state exists for the provided client ID.
        """
        if client_id not in self._states:
            raise KeyError(
                f"No memory state found in the checkpoint bundle for client ID {client_id}."
            )

        return self._states[client_id]


class AsyncStateManager:
    """
    A centralized container that unifies asynchronous simulation states.

    Acts as a facade, coordinating interactions between the historical
    model repository and the client request tracking ledger. It ensures that when
    a client's requested version is updated, the underlying history is automatically
    refreshed to purge stale models no longer needed by any active client.
    """

    def __init__(
        self, model_history: AsyncModelHistory, model_requests: AsyncClientModelRequests
    ):
        """
        Initializes the unified state manager for asynchronous runs.

        Args:
            model_history (AsyncModelHistory): The database tracking historical global models.
            model_requests (AsyncClientModelRequests): The ledger tracking requested model versions per client.
        """
        self.model_history = model_history
        self.model_requests = model_requests

    def fetch_historical_version_requested_by_client(self, cid: int) -> TensorDict:
        """
        Retrieves the specific historical model version currently requested by a client.

        Args:
            cid (int): The unique identifier of the requesting client.

        Returns:
            TensorDict: The state dictionary of the requested historical model.
        """
        requested_version = self.model_requests.get_client_request(cid=cid)
        return self.model_history.get_version(version=requested_version)

    def update_version_requested_by_client(
        self, cid: int, requested_version: int
    ) -> None:
        """
        Updates the version requested by a client and refreshes the model history.

        Records the new version requested by the client and subsequently
        triggers a refresh on the history database to remove any model versions that
        are no longer required by any active clients.

        Args:
            cid (int): The unique identifier of the client.
            requested_version (int): The new model version integer requested by the client.
        """
        self.model_requests.update_client(
            cid=cid,
            version=requested_version,
        )

        # Refresh model history (remove no-longer requested models)
        self.model_history.refresh(version_list=self.model_requests.version_list)

    def add_new_global_model_to_history(
        self, version: int, model_dict: TensorDict
    ) -> None:
        """
        Registers a newly updated global model into the history database.

        Args:
            version (int): The integer identifier for the new model version.
            model_dict (TensorDict): The PyTorch state dictionary of the new model.
        """
        self.model_history.add_version(
            version=version,
            model_dict=model_dict,
        )
