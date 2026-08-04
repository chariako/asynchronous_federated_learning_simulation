import pytest
import torch

from afl_sim.simulation.simulation_states import (
    AsyncClientModelRequests,
    AsyncModelHistory,
    AsyncStateManager,
    ClientMemoryStates,
)

_TENSOR_DICT = {"weights": torch.Tensor([1, 2])}
_NUM_CLIENTS = 2


@pytest.mark.parametrize(
    ("initial_dict", "expected_initial_history", "expected_initial_versions"),
    [(None, {}, []), (_TENSOR_DICT, {0: _TENSOR_DICT}, [0])],
)
def test_async_model_history_initialization(
    initial_dict, expected_initial_history, expected_initial_versions
):
    history = AsyncModelHistory(initial_model_dict=initial_dict)
    torch.testing.assert_close(
        expected_initial_history, history._history, atol=0.0, rtol=0.0
    )
    assert history.version_list == expected_initial_versions


def test_async_model_history_detaches_and_copies_tensors_on_init():
    original_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    model_dict = {"weights": original_tensor}

    history = AsyncModelHistory(initial_model_dict=model_dict)
    stored_tensor = history.get_version(0)["weights"]

    assert not stored_tensor.requires_grad
    assert original_tensor.data_ptr() != stored_tensor.data_ptr()
    assert stored_tensor.device.type == "cpu"


def test_async_model_history_detaches_and_copies_tensors_on_add():
    history = AsyncModelHistory(initial_model_dict=_TENSOR_DICT)

    original_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    model_dict = {"weights": original_tensor}

    history.add_version(version=10, model_dict=model_dict)
    stored_tensor = history.get_version(10)["weights"]

    assert not stored_tensor.requires_grad
    assert original_tensor.data_ptr() != stored_tensor.data_ptr()
    assert stored_tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("initial_dict", "added_versions", "expected_versions"),
    [(None, [1, 5], [1, 5]), (_TENSOR_DICT, [1, 5], [0, 1, 5])],
)
def test_async_model_history_roundtrip(initial_dict, added_versions, expected_versions):
    history = AsyncModelHistory(initial_model_dict=initial_dict)

    for version in added_versions:
        random_dict = {"weights": torch.rand(size=(2,))}
        history.add_version(version=version, model_dict=random_dict)
        torch.testing.assert_close(
            random_dict, history.get_version(version), atol=0.0, rtol=0.0
        )

    assert history.version_list == expected_versions


def test_async_model_history_get_version_raises_key_error():
    history = AsyncModelHistory(initial_model_dict=_TENSOR_DICT)
    with pytest.raises(KeyError, match="model version 10 is not present"):
        history.get_version(10)


def test_async_model_history_refresh():
    history = AsyncModelHistory(initial_model_dict=_TENSOR_DICT)
    old_versions = [5, 10, 15]
    active_versions = {10, 15}

    for version in old_versions:
        history.add_version(version=version, model_dict=_TENSOR_DICT)

    history.refresh(active_versions)

    assert history.version_list == [10, 15]


def test_async_client_model_requests_initialization():
    requests = AsyncClientModelRequests(num_clients=_NUM_CLIENTS)
    assert requests._model_requests == {0: 0, 1: 0}
    assert requests.version_list == {0}


def test_async_client_model_requests_round_trip():
    requests = AsyncClientModelRequests(num_clients=_NUM_CLIENTS)

    new_requests = {0: 10, 1: 20}

    for cid in new_requests:
        requests.update_client(cid, new_requests[cid])

    assert requests.version_list == set(new_requests.values())
    assert requests.state_dict == new_requests

    for cid in new_requests:
        assert requests.get_client_request(cid) == new_requests[cid]


def test_async_client_model_requests_raises_key_error():
    requests = AsyncClientModelRequests(num_clients=_NUM_CLIENTS)
    with pytest.raises(KeyError, match="Client ID 10 is not registered"):
        requests.get_client_request(10)


def test_client_memory_states_init():
    states = ClientMemoryStates()
    assert states._states == {}


def test_client_memory_states_round_trip():
    states = ClientMemoryStates()

    for cid in range(_NUM_CLIENTS):
        random_dict = {"weights": torch.rand(size=(2,))}
        states.add_client_mem_state(cid, random_dict)
        torch.testing.assert_close(
            states.get_client_mem_state(cid), random_dict, atol=0.0, rtol=0.0
        )

    assert states.client_ids == [0, 1]


def test_client_memory_states_get_raises_key_error():
    states = ClientMemoryStates()
    with pytest.raises(KeyError, match="No memory state found"):
        states.get_client_mem_state(10)


@pytest.fixture
def async_state_manager_factory():
    def _factory(requests=None, history=None):
        requests = (
            AsyncClientModelRequests(_NUM_CLIENTS) if requests is None else requests
        )
        history = (
            AsyncModelHistory(initial_model_dict=_TENSOR_DICT)
            if history is None
            else history
        )
        return AsyncStateManager(model_history=history, model_requests=requests)

    return _factory


def test_async_state_manager_fetch(async_state_manager_factory):
    requests = AsyncClientModelRequests(_NUM_CLIENTS)
    requests.update_client(1, 10)

    history = AsyncModelHistory(initial_model_dict=_TENSOR_DICT)
    version_10 = {"weights": torch.Tensor([0.3, 0.4])}
    history.add_version(version=10, model_dict=version_10)

    state = async_state_manager_factory(requests=requests, history=history)

    client_0_version = state.fetch_historical_version_requested_by_client(0)
    client_1_version = state.fetch_historical_version_requested_by_client(1)

    torch.testing.assert_close(client_0_version, _TENSOR_DICT, atol=0.0, rtol=0.0)
    torch.testing.assert_close(client_1_version, version_10, atol=0.0, rtol=0.0)


def test_async_state_manager_version_round_trip(async_state_manager_factory):
    state = async_state_manager_factory()

    version_10 = {"weights": torch.Tensor([0.3, 0.4])}
    state.add_new_global_model_to_history(version=10, model_dict=version_10)
    state.update_version_requested_by_client(cid=0, requested_version=10)

    client_0_version = state.fetch_historical_version_requested_by_client(0)
    torch.testing.assert_close(client_0_version, version_10, atol=0.0, rtol=0.0)


def test_async_state_manager_client_update_refreshes_history(
    async_state_manager_factory,
):
    state = async_state_manager_factory()

    version_10 = {"weights": torch.Tensor([0.3, 0.4])}
    state.add_new_global_model_to_history(version=10, model_dict=version_10)

    state.update_version_requested_by_client(cid=0, requested_version=10)
    assert state.model_history.version_list == [0, 10]

    state.update_version_requested_by_client(cid=1, requested_version=10)
    assert state.model_history.version_list == [10]
