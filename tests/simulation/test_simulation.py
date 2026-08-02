import pytest
import torch

from afl_sim.enums import MemoryType
from afl_sim.simulation.simulation_states import ClientMemoryStates
from afl_sim.utils import MetricsLogger


@pytest.mark.parametrize("device_type", ["mps", "cpu", "cuda"])
def test_model_moved_to_device_at_init(
    device_type, simulation_factory, model_factory, mocker
):
    model = model_factory()
    model_class = type(model)

    def fake_to(self, *args, **kwargs):
        return self

    mock_model_to = mocker.patch.object(
        model_class, "to", autospec=True, side_effect=fake_to
    )

    simulation = simulation_factory(device_type=device_type)

    mock_model_to.assert_called_with(
        simulation.model_shell, device=torch.device(device_type)
    )


@pytest.mark.parametrize(
    ("comm_type", "global_first_idx", "event_num", "local_idx"),
    [
        ("sync", 0, 30, 0),
        ("sync", 100, 30, 10),
        ("async", 0, 30, 10),
        ("async", 100, 30, 0),
    ],
)
def test_simulation_clock_coupling(
    local_idx,
    comm_type,
    global_first_idx,
    event_num,
    simulation_clock_factory,
    simulation_factory,
    mocker,
):
    num_clients = 10
    client_requests = {i: i + 1 for i in range(num_clients)}

    simulation_clock = simulation_clock_factory(
        comm_type=comm_type,
        global_first_idx=global_first_idx,
        event_num=event_num,
        num_clients=num_clients,
    )
    simulation = simulation_factory(
        comm_type=comm_type,
        global_first_idx=global_first_idx,
        event_num=event_num,
        local_idx=local_idx,
        num_clients=num_clients,
        client_requests=client_requests,
    )

    expected_idx = simulation_clock.local_to_global_idx(local_idx)
    expected_clients = simulation_clock.local_idx_to_incoming_clients(local_idx)
    expected_sim_time = simulation_clock.local_idx_to_sim_time(local_idx)

    assert simulation.global_idx == expected_idx
    assert simulation.incoming_clients == expected_clients
    assert simulation.sim_time == expected_sim_time

    mocker.patch.object(simulation, "_process_incoming_clients")
    mocker.patch.object(simulation, "_process_global_update", return_value=False)
    mocker.patch.object(simulation, "_handle_external_files_post_global_update")

    expected_sim_length = event_num - local_idx

    for _ in range(expected_sim_length):
        assert simulation._step() is True

    assert simulation._step() is False


@pytest.mark.parametrize("memory_type", list(MemoryType))
def test_build_client_state_dicts(memory_type, simulation_factory, clients_factory):
    num_clients = 2
    clients = clients_factory(mem_type=memory_type, num_clients=num_clients)
    simulation = simulation_factory(mem_type=memory_type, num_clients=num_clients)

    client_states = simulation._build_client_state_dicts()

    if memory_type.has_memory:
        assert isinstance(client_states, ClientMemoryStates)

        for cid in range(num_clients):
            client_state = client_states.get_client_mem_state(cid)
            expected_state = clients[cid]._memory

            torch.testing.assert_close(client_state, expected_state, rtol=0.0, atol=0.0)
    else:
        assert client_states is None


@pytest.mark.parametrize(
    ("comm_type", "expected_extractor"),
    [
        ("sync", lambda sim, cid, reqs: sim.server.global_model_dict),
        (
            "async",
            lambda sim, cid, reqs: sim.async_states.model_history.get_version(
                reqs[cid]
            ),
        ),
    ],
)
def test_fetch_model_request(comm_type, expected_extractor, simulation_factory):
    cid = 0
    client_requests = {0: 9, 1: 10}

    simulation = simulation_factory(
        comm_type=comm_type,
        history_versions=(9, 10),
        num_clients=2,
        client_requests=client_requests,
    )

    fetched = simulation._fetch_requested_state_dict_to_client(client_id=cid)

    expected = expected_extractor(simulation, cid, client_requests)

    torch.testing.assert_close(fetched, expected, rtol=0.0, atol=0.0)


def test_async_local_book_keeping_sync(simulation_factory):
    cid = 0
    num_clients = 2
    current_version = 5
    client_requests = {0: 4, 1: 3}
    history_versions = (client_requests[0], client_requests[1])
    comm_type = "sync"

    simulation = simulation_factory(
        comm_type=comm_type,
        current_version=current_version,
        num_clients=num_clients,
        client_requests=client_requests,
        history_versions=history_versions,
    )

    assert simulation.async_states is None
    simulation._async_post_local_update_book_keeping(client_id=cid)
    assert simulation.async_states is None


def test_async_local_book_keeping_async(simulation_factory):
    cid = 0
    comm_type = "async"
    num_clients = 2
    current_version = 5
    client_requests = {0: 4, 1: 3}
    history_versions = (client_requests[0], client_requests[1])
    history_versions_list = list(history_versions)

    simulation = simulation_factory(
        comm_type=comm_type,
        current_version=current_version,
        num_clients=num_clients,
        client_requests=client_requests,
        history_versions=history_versions,
    )

    old_client_req = simulation.async_states.model_requests.get_client_request(cid)
    old_versions = simulation.async_states.model_history.version_list

    assert old_client_req == client_requests[cid]
    assert old_versions == history_versions_list

    simulation._async_post_local_update_book_keeping(client_id=cid)
    history_versions_list.pop(cid)

    new_client_req = simulation.async_states.model_requests.get_client_request(cid)
    new_versions = simulation.async_states.model_history.version_list

    assert new_client_req == current_version
    assert new_versions == history_versions_list


@pytest.mark.parametrize(
    (
        "comm_type",
        "history_versions",
        "current_version",
        "expected_versions",
        "expect_async_states_none",
    ),
    [
        ("sync", (8, 9), 10, [], True),
        ("sync", (8, 9, 10), 10, [], True),
        ("async", (8, 9), 10, [8, 9, 10], False),
        ("async", (8, 9, 10), 10, [8, 9, 10], False),
    ],
)
def test_async_global_book_keeping(
    comm_type,
    history_versions,
    current_version,
    expected_versions,
    expect_async_states_none,
    simulation_factory,
):
    num_clients = 2
    client_requests = {0: 9, 1: 10}

    simulation = simulation_factory(
        comm_type=comm_type,
        history_versions=history_versions,
        current_version=current_version,
        num_clients=num_clients,
        client_requests=client_requests,
    )

    simulation._async_post_global_update_book_keeping()

    assert (simulation.async_states is None) == expect_async_states_none
    if not expect_async_states_none:
        assert simulation.async_states.model_history.version_list == expected_versions


@pytest.mark.parametrize("global_update_performed", [True, False])
def test_handle_external_files(global_update_performed, simulation_factory, mocker):
    simulation = simulation_factory()

    mock_exec_log = mocker.patch.object(simulation, "_update_logger_post_global_update")
    mock_metrics_log = mocker.patch.object(simulation.metrics_logger, "log")
    mock_checkpoint = mocker.patch.object(simulation.checkpoint_manager, "save_best")

    simulation._handle_external_files_post_global_update(global_update_performed)

    expected_exec_calls = [mocker.call()] if global_update_performed else []

    expected_metrics_calls = (
        [
            mocker.call(
                global_idx=simulation.global_idx,
                loss=simulation.server.current_loss,
                accuracy=simulation.server.current_acc,
                sim_time=simulation.sim_time,
            )
        ]
        if global_update_performed
        else []
    )

    expected_checkpoint_calls = (
        [
            mocker.call(
                simulation.server.global_model_dict,
                current_acc=simulation.server.current_acc,
                best_acc=simulation.server.best_acc,
            )
        ]
        if global_update_performed
        else []
    )

    assert mock_exec_log.call_args_list == expected_exec_calls
    assert mock_metrics_log.call_args_list == expected_metrics_calls
    assert mock_checkpoint.call_args_list == expected_checkpoint_calls


@pytest.mark.parametrize("comm_type", ["sync", "async"])
def test_process_local_update(comm_type, simulation_factory, mocker):
    num_clients = 2
    simulation = simulation_factory(num_clients=num_clients, comm_type=comm_type)

    mock_fetch = mocker.patch.object(
        simulation,
        "_fetch_requested_state_dict_to_client",
        side_effect=lambda client_id: client_id,
    )

    mock_client_0_update = mocker.patch.object(
        simulation.clients[0], "compute_update", return_value=0
    )
    mock_client_1_update = mocker.patch.object(
        simulation.clients[1], "compute_update", return_value=1
    )

    mock_book_keeping = mocker.patch.object(
        simulation,
        "_async_post_local_update_book_keeping",
    )
    mock_server_agg = mocker.patch.object(
        simulation.server,
        "aggregate_update",
    )

    manager = mocker.Mock()
    manager.attach_mock(mock_fetch, "fetch")
    manager.attach_mock(mock_client_0_update, "update_0")
    manager.attach_mock(mock_client_1_update, "update_1")
    manager.attach_mock(mock_server_agg, "server_agg")
    manager.attach_mock(mock_book_keeping, "book_keeping")

    incoming_clients = simulation.incoming_clients
    expected_calls = []

    for cid in incoming_clients:
        expected_calls.append(mocker.call.fetch(cid))
        client_args = {
            "model_shell": mocker.ANY,
            "device": mocker.ANY,
            "global_idx": mocker.ANY,
            "requested_state_dict": cid,
        }
        client_call = (
            mocker.call.update_0(**client_args)
            if cid == 0
            else mocker.call.update_1(**client_args)
        )
        expected_calls.append(client_call)
        expected_calls.append(mocker.call.book_keeping(cid))
        expected_calls.append(mocker.call.server_agg(cid))

    simulation._process_incoming_clients()

    manager.assert_has_calls(expected_calls, any_order=False)


def test_process_global_update(simulation_factory, mocker):
    simulation = simulation_factory()

    mock_server_update = mocker.patch.object(
        simulation.server,
        "global_update",
        return_value=False,
    )
    mock_book_keeping = mocker.patch.object(
        simulation,
        "_async_post_global_update_book_keeping",
    )

    manager = mocker.Mock()
    manager.attach_mock(mock_server_update, "server_update")
    manager.attach_mock(mock_book_keeping, "book_keeping")
    expected_calls = [
        mocker.call.server_update(
            model_shell=mocker.ANY, device=mocker.ANY, global_idx=mocker.ANY
        ),
        mocker.call.book_keeping(),
    ]

    simulation._process_global_update()

    manager.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.parametrize(
    ("local_idx", "event_num", "expect_step"), [(3, 2, False), (2, 3, True)]
)
def test_simulation_step(local_idx, event_num, expect_step, simulation_factory, mocker):
    simulation = simulation_factory(local_idx=local_idx, event_num=event_num)

    mock_client_update = mocker.patch.object(simulation, "_process_incoming_clients")
    mock_server_update = mocker.patch.object(
        simulation, "_process_global_update", return_value=False
    )
    mock_handle_external = mocker.patch.object(
        simulation, "_handle_external_files_post_global_update"
    )

    manager = mocker.Mock()
    manager.attach_mock(mock_client_update, "client_update")
    manager.attach_mock(mock_server_update, "server_update")
    manager.attach_mock(mock_handle_external, "handle_external")

    expected_calls = (
        [
            mocker.call.client_update(),
            mocker.call.server_update(),
            mocker.call.handle_external(global_update_performed=False),
        ]
        if expect_step
        else []
    )

    expected_idx = local_idx + 1 if expect_step else local_idx

    step_result = simulation._step()

    assert step_result is expect_step
    assert simulation.local_idx == expected_idx

    assert manager.mock_calls == expected_calls


def test_update_logger_dynamic_formatting(simulation_factory, capture_logs):
    simulation = simulation_factory()

    simulation.server.current_loss = 0.345678
    simulation.server.current_acc = 95.123

    simulation._update_logger_post_global_update()

    assert "Loss: 0.3457" in capture_logs.text
    assert "Acc: 95.12%" in capture_logs.text


@pytest.mark.parametrize(
    ("timeout", "stop_requested", "sim_duration", "log_text"),
    [
        (100.0, False, 200.0, "Timeout exceeded"),
        (100.0, True, 90.0, "Simulation interrupted"),
    ],
)
def test_simulation_break(
    timeout,
    stop_requested,
    sim_duration,
    log_text,
    simulation_factory,
    mocker,
    capture_logs,
):
    simulation = simulation_factory(timeout=timeout, stop_requested=stop_requested)
    simulation.metrics_logger.run_dir.mkdir(parents=True, exist_ok=True)

    mocker.patch.object(simulation, "_step", return_value=True)
    mocker.patch(
        "afl_sim.simulation.simulation.time.time",
        side_effect=[0, 0, sim_duration, sim_duration],
    )

    mock_save_ckpt = mocker.patch.object(simulation.checkpoint_manager, "save_latest")
    mock_save_shutdown_ckpt = mocker.patch.object(
        simulation.checkpoint_manager, "save_shutdown"
    )

    simulation.run()

    assert not mock_save_ckpt.called
    assert mock_save_shutdown_ckpt.called
    assert log_text in capture_logs.text


def test_simulation_run(
    simulation_factory,
    mocker,
):
    simulation = simulation_factory(event_num=1)
    simulation.metrics_logger.run_dir.mkdir(parents=True, exist_ok=True)

    mock_save_ckpt = mocker.patch.object(simulation.checkpoint_manager, "save_latest")
    mock_save_shutdown_ckpt = mocker.patch.object(
        simulation.checkpoint_manager, "save_shutdown"
    )
    spy_metrics_logger_enter = mocker.spy(MetricsLogger, "__enter__")
    spy_metrics_logger_exit = mocker.spy(MetricsLogger, "__exit__")
    spy_step = mocker.spy(simulation, "_step")

    manager = mocker.Mock()
    manager.attach_mock(spy_metrics_logger_enter, "metrics_logger_enter")
    manager.attach_mock(spy_metrics_logger_exit, "metrics_logger_exit")
    manager.attach_mock(mock_save_ckpt, "save_latest")
    manager.attach_mock(spy_step, "step")
    manager.attach_mock(mock_save_shutdown_ckpt, "save_shutdown")

    simulation.run()
    expected_calls = [
        mocker.call.metrics_logger_enter(mocker.ANY),
        mocker.call.step(),
        mocker.call.save_latest(
            server_state=mocker.ANY,
            client_states=mocker.ANY,
            async_states=mocker.ANY,
            global_idx=mocker.ANY,
            sim_duration=mocker.ANY,
        ),
        mocker.call.step(),
        mocker.call.save_shutdown(
            server_state=mocker.ANY,
            client_states=mocker.ANY,
            async_states=mocker.ANY,
            global_idx=mocker.ANY,
        ),
        mocker.call.metrics_logger_exit(mocker.ANY, mocker.ANY, mocker.ANY, mocker.ANY),
    ]

    manager.assert_has_calls(expected_calls, any_order=False)
