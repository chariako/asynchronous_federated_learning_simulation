import pytest
from loguru import logger
from pydantic import ValidationError

from afl_sim.config import (
    AppConfig,
    AsyncStrategy,
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizationConfig,
    SimulationConfig,
    SyncStrategy,
    VisualizationConfig,
)
from afl_sim.enums import DatasetType, ModelType


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture loguru logs using pytest's standard caplog.
    Loguru does not propagate to caplog by default.
    """
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


def test_config_stability_defaults():
    """
    Test that the config instantiates with safe defaults.
    """
    config = AppConfig()

    # Workers default to 0 to prevent multiprocessing crashes on Windows/macOS
    assert config.optimization.num_workers == 0
    assert config.evaluation.num_workers == 0

    # Default strategy should be Async (safer fallback)
    assert isinstance(config.comm_strategy, AsyncStrategy)


def test_sync_strategy_sample_size_logic():
    """Ensure client sample size (k) does not exceed total clients (N)."""
    with pytest.raises(ValidationError) as excinfo:
        AppConfig(
            simulation=SimulationConfig(num_clients=10),
            comm_strategy=SyncStrategy(sample_size=11),
        )
    assert "Sample size (11) cannot exceed total clients (10)" in str(excinfo.value)


def test_model_channel_compatibility():
    """Ensure dataset channels (e.g., 1 for MNIST) match model input requirements."""
    with pytest.raises(ValidationError) as excinfo:
        AppConfig(
            data=DataConfig(dataset=DatasetType.MNIST),
            model=ModelConfig(model_name=ModelType.RESNET18),
        )
    assert "requires 3 channel(s)" in str(excinfo.value)


def test_stress_test_incompatible_with_logreg():
    """Ensure stress test (removing norms) fails for models without norm layers."""
    with pytest.raises(ValidationError) as excinfo:
        AppConfig(model=ModelConfig(model_name=ModelType.LOG_REG, stress_test=True))
    assert "Stress test (removing norms) is not applicable" in str(excinfo.value)


def test_batch_size_exceeds_dataset_size():
    """Ensure batch size cannot be larger than the available dataset."""
    # Case 1: Training Batch Size > Train Set
    with pytest.raises(ValidationError) as excinfo:
        AppConfig(
            optimization=OptimizationConfig(batch_size=60001),
            data=DataConfig(dataset=DatasetType.MNIST),
        )
    assert "Batch size (60001) cannot be equal to or exceed dataset size" in str(
        excinfo.value
    )

    # Case 2: Eval Batch Size > Test Set
    with pytest.raises(ValidationError) as excinfo:
        AppConfig(
            evaluation=EvaluationConfig(batch_size=10001),
            data=DataConfig(dataset=DatasetType.MNIST),
        )
    assert "Evaluation batch size" in str(excinfo.value)


def test_checkpoint_interval_warning(capture_logs):
    """Ensure a warning is logged if checkpoint interval >= simulation timeout."""
    AppConfig(
        simulation=SimulationConfig(timeout_seconds=100),
        checkpoints=CheckpointConfig(interval_seconds=200),
    )

    assert "Checkpoint interval 200.0 is equal or greater" in capture_logs.text


def test_visualization_auto_disabled_for_large_scale(capture_logs):
    """Ensure visualization is automatically disabled for large client counts."""
    config = AppConfig(
        simulation=SimulationConfig(num_clients=151),
        visualization=VisualizationConfig(
            visualize_client_arrivals=True, visualize_data_split=True
        ),
    )

    assert not config.visualization.visualize_client_arrivals
    assert not config.visualization.visualize_data_split

    assert "Too many clients" in capture_logs.text


def test_strategy_parsing_from_dict():
    """Test that raw dictionaries are correctly converted to specific Strategy classes."""
    raw_sync = {"comm_strategy": {"type": "sync", "sample_size": 5}}
    config = AppConfig.model_validate(raw_sync)
    assert isinstance(config.comm_strategy, SyncStrategy)

    raw_async = {"comm_strategy": {"type": "async", "buffer_size": 10}}
    config = AppConfig.model_validate(raw_async)
    assert isinstance(config.comm_strategy, AsyncStrategy)

    raw_ambiguous = {"comm_strategy": {"sample_size": 5}}
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw_ambiguous)
