from dataclasses import dataclass
from pathlib import Path

import pytest

from afl_sim.config import (
    AppConfig,
    AsyncStrategy,
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    MemStrategyConfig,
    ModelConfig,
    OptimizationConfig,
    SimulationConfig,
    SyncStrategy,
    VisualizationConfig,
)
from afl_sim.enums import DeviceType, MemoryType, ModelType
from afl_sim.simulation.simulation_builder import (
    _get_next_global_index,
    _initialize_async_states,
    _initialize_clients,
    _initialize_metrics_logger,
    _initialize_server,
    build_simulation,
)

MODULEPATH = "afl_sim.simulation.simulation_builder"


@pytest.fixture
def app_config_factory():
    def _factory(
        comm_strategy_type="sync", mem_strategy_type=MemoryType.DISABLED, num_clients=2
    ):
        comm_strategy = (
            SyncStrategy(sample_size=num_clients)
            if comm_strategy_type == "sync"
            else AsyncStrategy()
        )
        mem_strategy = MemStrategyConfig(type=mem_strategy_type)

        return AppConfig(
            comm_strategy=comm_strategy,
            mem_strategy=mem_strategy,
            data=DataConfig(),
            model=ModelConfig(model_name=ModelType.LOG_REG),
            simulation=SimulationConfig(device=DeviceType.CPU, num_clients=num_clients),
            evaluation=EvaluationConfig(num_workers=0),
            optimization=OptimizationConfig(),
            checkpoints=CheckpointConfig(),
            visualization=VisualizationConfig(),
        )

    return _factory


@dataclass
class SimulationDirs:
    data_dir: Path
    checkpoint_dir: Path
    run_dir: Path


@pytest.fixture
def get_simulations_dirs(tmp_path):
    return SimulationDirs(
        data_dir=tmp_path.joinpath("data"),
        checkpoint_dir=tmp_path.joinpath("checkpoints"),
        run_dir=tmp_path.joinpath("outputs"),
    )


@pytest.fixture
def get_mock_device(mocker):
    mock_device = mocker.MagicMock(type="MOCK_DEVICE_TYPE")
    mock_device.__str__.return_value = "MOCK_DEVICE"
    return mock_device


def test_build_simulation_operation_order(
    app_config_factory,
    get_simulations_dirs,
    mocker,
    capture_logs,
    get_mock_device,
):
    app_config: AppConfig = app_config_factory()
    dirs = get_simulations_dirs
    resume = False
    mock_device = get_mock_device

    mock_torch_manual_seed = mocker.patch("torch.manual_seed")
    mock_checkpoint_manager = mocker.patch(
        f"{MODULEPATH}.CheckpointManager", return_value=mocker.sentinel.CKPT_MNGR
    )
    mock_get_device = mocker.patch(f"{MODULEPATH}.get_device", return_value=mock_device)
    mock_get_model = mocker.patch(
        f"{MODULEPATH}.get_model", return_value=mocker.sentinel.MODEL
    )
    mock_data_manager = mocker.patch(
        f"{MODULEPATH}.DataManager", return_value=mocker.sentinel.DATA_MNGR
    )
    mock_server_init = mocker.patch(
        f"{MODULEPATH}._initialize_server", return_value=mocker.sentinel.SERVER
    )
    mock_client_init = mocker.patch(
        f"{MODULEPATH}._initialize_clients", return_value=mocker.sentinel.CLIENTS
    )
    mock_get_index = mocker.patch(
        f"{MODULEPATH}._get_next_global_index", return_value=mocker.sentinel.IDX
    )
    mock_async_init = mocker.patch(
        f"{MODULEPATH}._initialize_async_states", return_value=mocker.sentinel.ASYNC
    )
    mock_get_clock = mocker.patch(
        f"{MODULEPATH}.get_clock", return_value=mocker.sentinel.CLOCK
    )
    mock_logger_init = mocker.patch(
        f"{MODULEPATH}._initialize_metrics_logger", return_value=mocker.sentinel.LOGGER
    )
    mock_simulation = mocker.patch(
        f"{MODULEPATH}.Simulation", return_value=mocker.sentinel.SIM
    )

    manager = mocker.Mock()
    manager.attach_mock(mock_torch_manual_seed, "set_seed")
    manager.attach_mock(mock_checkpoint_manager, "ckpt_mngr")
    manager.attach_mock(mock_get_device, "get_device")
    manager.attach_mock(mock_get_model, "get_model")
    manager.attach_mock(mock_data_manager, "data_mngr")
    manager.attach_mock(mock_server_init, "init_server")
    manager.attach_mock(mock_client_init, "init_clients")
    manager.attach_mock(mock_get_index, "get_idx")
    manager.attach_mock(mock_async_init, "init_async")
    manager.attach_mock(mock_get_clock, "get_clock")
    manager.attach_mock(mock_logger_init, "init_log")
    manager.attach_mock(mock_simulation, "sim")

    expected_calls = [
        mocker.call.set_seed(seed=app_config.simulation.torch_seed),
        mocker.call.ckpt_mngr(
            checkpoint_dir=dirs.checkpoint_dir, checkpoint_config=app_config.checkpoints
        ),
        mocker.call.get_device(app_config.simulation.device),
        mocker.call.get_model(
            dataset=app_config.data.dataset, model_config=app_config.model
        ),
        mocker.call.data_mngr(
            num_clients=app_config.simulation.num_clients,
            data_config=app_config.data,
            eval_config=app_config.evaluation,
            optim_config=app_config.optimization,
            data_dir=dirs.data_dir,
            device_type=mock_device.type,
            visualize=app_config.visualization.visualize_data_split,
        ),
        mocker.call.init_server(
            model=mocker.sentinel.MODEL,
            config=app_config,
            data_manager=mocker.sentinel.DATA_MNGR,
            resume=resume,
            checkpoint_manager=mocker.sentinel.CKPT_MNGR,
        ),
        mocker.call.init_clients(
            model=mocker.sentinel.MODEL,
            config=app_config,
            data_manager=mocker.sentinel.DATA_MNGR,
            resume=resume,
            checkpoint_manager=mocker.sentinel.CKPT_MNGR,
        ),
        mocker.call.get_idx(
            resume=resume, checkpoint_manager=mocker.sentinel.CKPT_MNGR
        ),
        mocker.call.init_async(
            num_clients=app_config.simulation.num_clients,
            comm_strategy=app_config.comm_strategy,
            resume=resume,
            checkpoint_manager=mocker.sentinel.CKPT_MNGR,
            initial_model=mocker.sentinel.MODEL,
        ),
        mocker.call.get_clock(
            config=app_config,
            data_dir=dirs.data_dir,
            global_next_idx=mocker.sentinel.IDX,
        ),
        mocker.call.init_log(
            run_dir=dirs.run_dir,
            resume=resume,
            checkpoint_manager=mocker.sentinel.CKPT_MNGR,
        ),
        mocker.call.sim(
            mem_strategy=app_config.mem_strategy,
            timeout=app_config.simulation.timeout_seconds,
            metrics_logger=mocker.sentinel.LOGGER,
            checkpoint_manager=mocker.sentinel.CKPT_MNGR,
            device=mock_device,
            server=mocker.sentinel.SERVER,
            clients=mocker.sentinel.CLIENTS,
            clock=mocker.sentinel.CLOCK,
            model_shell=mocker.sentinel.MODEL,
            async_states=mocker.sentinel.ASYNC,
        ),
    ]

    build_simulation(
        config=app_config,
        data_dir=dirs.data_dir,
        checkpoint_dir=dirs.checkpoint_dir,
        output_dir=dirs.run_dir,
        resume=resume,
    )

    assert "running on device: MOCK_DEVICE" in capture_logs.text
    manager.assert_has_calls(expected_calls, any_order=False)


@pytest.fixture
def get_mock_data_manager(mocker):
    mock_data_manager = mocker.Mock()
    mock_data_manager.get_evaluation_dataloader.return_value = (
        mocker.sentinel.EVAL_DATALOADER
    )
    mock_data_manager.get_eval_transform.return_value = mocker.sentinel.EVAL_TS

    mock_data_manager.get_client_dataloader.side_effect = lambda client_id: getattr(
        mocker.sentinel, f"DATALOADER_{client_id}"
    )
    mock_data_manager.get_client_weight.side_effect = lambda client_id: (
        1 / (client_id + 1)
    )

    mock_data_manager.get_train_transform.return_value = mocker.sentinel.TRAIN_TS
    return mock_data_manager


@pytest.fixture
def get_mock_checkpoint_manager(mocker):
    mock_manager = mocker.Mock()
    mock_manager.load_server_states.return_value = mocker.sentinel.SERVER_STATES
    mock_manager.load_client_memory_state.side_effect = lambda cid: getattr(
        mocker.sentinel, f"STATE_{cid}"
    )

    mock_manager.load_latest_metadata.return_value = mocker.Mock(
        global_idx=mocker.sentinel.IDX, history_version_list=[42, 43]
    )

    mock_manager.load_model_requests.return_value = mocker.sentinel.REQ_FROM_CKPT
    mock_manager.load_history_version.side_effect = lambda version: getattr(
        mocker.sentinel, f"V_{version}"
    )

    return mock_manager


@pytest.mark.parametrize("resume", [True, False])
def test_init_server(
    resume,
    app_config_factory,
    mocker,
    get_mock_data_manager,
    get_mock_checkpoint_manager,
):
    app_config = app_config_factory()

    mock_server = mocker.patch(f"{MODULEPATH}.Server")
    mock_data_manager = get_mock_data_manager
    mock_checkpoint_manager = get_mock_checkpoint_manager

    _initialize_server(
        model=mocker.sentinel.MODEL,
        config=app_config,
        data_manager=mock_data_manager,
        resume=resume,
        checkpoint_manager=mock_checkpoint_manager,
    )

    mock_server.assert_called_once_with(
        model=mocker.sentinel.MODEL,
        test_loader=mocker.sentinel.EVAL_DATALOADER,
        test_transform=mocker.sentinel.EVAL_TS,
        aggregation_goal=app_config.comm_strategy.agg_target,
        num_clients=app_config.simulation.num_clients,
        reset_buffer=app_config.mem_strategy.type.requires_buffer_reset,
        base_seed=app_config.simulation.torch_seed,
    )

    expected_load_calls = [mocker.call(mocker.sentinel.SERVER_STATES)] if resume else []
    mock_server.return_value.load_state_dict.assert_has_calls(expected_load_calls)


@pytest.mark.parametrize("resume", [True, False])
@pytest.mark.parametrize("mem_type", list(MemoryType))
def test_init_clients(
    resume,
    mem_type,
    app_config_factory,
    mocker,
    get_mock_data_manager,
    get_mock_checkpoint_manager,
):
    app_config = app_config_factory(mem_strategy_type=mem_type)

    mock_client_instances = [
        mocker.Mock(client_id=i) for i in range(app_config.simulation.num_clients)
    ]
    mock_client_class = mocker.patch(
        f"{MODULEPATH}.Client", side_effect=mock_client_instances
    )

    mock_data_manager = get_mock_data_manager
    mock_checkpoint_manager = get_mock_checkpoint_manager

    _initialize_clients(
        model=mocker.sentinel.MODEL,
        config=app_config,
        data_manager=mock_data_manager,
        resume=resume,
        checkpoint_manager=mock_checkpoint_manager,
    )

    expected_client_calls = [
        mocker.call(
            client_id=i,
            initial_model=mocker.sentinel.MODEL,
            dataloader=getattr(mocker.sentinel, f"DATALOADER_{i}"),
            weight=1 / (i + 1),
            transform=mocker.sentinel.TRAIN_TS,
            optim_config=app_config.optimization,
            memory_strategy=app_config.mem_strategy,
            base_seed=app_config.simulation.torch_seed,
        )
        for i in range(app_config.simulation.num_clients)
    ]
    mock_client_class.assert_has_calls(expected_client_calls, any_order=True)

    if resume and mem_type.has_memory:
        for i in range(app_config.simulation.num_clients):
            mock_client_instances[i].load_mem_state_dict.assert_called_once_with(
                getattr(mocker.sentinel, f"STATE_{i}")
            )
    else:
        for i in range(app_config.simulation.num_clients):
            mock_client_instances[i].load_mem_state_dict.assert_not_called()


@pytest.mark.parametrize("resume", [True, False])
def test_get_global_idx(resume, get_mock_checkpoint_manager, mocker):
    checkpoint_manager = get_mock_checkpoint_manager

    idx = _get_next_global_index(resume=resume, checkpoint_manager=checkpoint_manager)

    expected_idx = mocker.sentinel.IDX if resume else 0
    assert idx == expected_idx


@pytest.mark.parametrize("resume", [True, False])
def test_init_metrics_logger(
    resume, get_mock_checkpoint_manager, mocker, get_simulations_dirs
):
    checkpoint_manager = get_mock_checkpoint_manager
    dirs = get_simulations_dirs

    mock_logger = mocker.patch(f"{MODULEPATH}.MetricsLogger")
    _initialize_metrics_logger(
        run_dir=dirs.run_dir, resume=resume, checkpoint_manager=checkpoint_manager
    )

    mock_logger.assert_called_once_with(run_dir=dirs.run_dir)
    expected_trim_call = (
        [mocker.call(next_global_idx=mocker.sentinel.IDX)] if resume else []
    )

    mock_logger.return_value.trim_history.assert_has_calls(expected_trim_call)


@pytest.mark.parametrize("resume", [True, False])
def test_async_state_init_sync(
    resume, app_config_factory, get_mock_checkpoint_manager, mocker
):
    app_config: AppConfig = app_config_factory(comm_strategy_type="sync")
    checkpoint_manager = get_mock_checkpoint_manager
    states = _initialize_async_states(
        num_clients=app_config.simulation.num_clients,
        comm_strategy=app_config.comm_strategy,
        resume=resume,
        checkpoint_manager=checkpoint_manager,
        initial_model=mocker.sentinel.MODEL,
    )
    assert states is None


@pytest.fixture
def get_mock_model(mocker):
    mock_model = mocker.Mock()
    mock_model.state_dict.return_value = mocker.sentinel.STATE_DICT
    return mock_model


@pytest.fixture
def get_mock_history(mocker):
    return mocker.Mock()


@pytest.mark.parametrize("resume", [True, False])
def test_async_state_init_async(
    resume,
    app_config_factory,
    get_mock_checkpoint_manager,
    get_mock_model,
    get_mock_history,
    mocker,
):
    app_config: AppConfig = app_config_factory(comm_strategy_type="async")
    checkpoint_manager = get_mock_checkpoint_manager
    model = get_mock_model
    history = get_mock_history

    mock_history_class = mocker.patch(
        f"{MODULEPATH}.AsyncModelHistory", return_value=history
    )
    mock_requests_class = mocker.patch(
        f"{MODULEPATH}.AsyncClientModelRequests", return_value=mocker.sentinel.REQ
    )
    mock_manager_class = mocker.patch(f"{MODULEPATH}.AsyncStateManager")

    _initialize_async_states(
        num_clients=app_config.simulation.num_clients,
        comm_strategy=app_config.comm_strategy,
        resume=resume,
        checkpoint_manager=checkpoint_manager,
        initial_model=model,
    )

    if resume:
        mock_history_class.assert_called_once_with(initial_model_dict=None)
        history.add_version.assert_has_calls(
            [
                mocker.call(42, mocker.sentinel.V_42),
                mocker.call(43, mocker.sentinel.V_43),
            ]
        )
        mock_requests_class.assert_not_called()
        mock_manager_class.assert_called_once_with(
            model_history=history,
            model_requests=mocker.sentinel.REQ_FROM_CKPT,
        )
    else:
        mock_history_class.assert_called_once_with(
            initial_model_dict=mocker.sentinel.STATE_DICT
        )
        history.add_version.assert_not_called()
        mock_requests_class.assert_called_once_with(app_config.simulation.num_clients)
        mock_manager_class.assert_called_once_with(
            model_history=history,
            model_requests=mocker.sentinel.REQ,
        )
