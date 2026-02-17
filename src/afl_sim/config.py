from typing import Annotated, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from afl_sim.enums import DatasetType, DeviceType, MemoryType, ModelType


class BaseImmutableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyncStrategy(BaseImmutableConfig):
    type: Literal["sync"] = "sync"
    sample_size: int = Field(default=3, gt=0, description="Clients sampled per round.")

    @property
    def agg_target(self) -> int:
        return self.sample_size


class AsyncStrategy(BaseImmutableConfig):
    type: Literal["async"] = "async"
    buffer_size: int = Field(default=3, gt=0, description="Buffer size trigger.")

    @property
    def agg_target(self) -> int:
        return self.buffer_size


CommStrategyConfig = Annotated[
    SyncStrategy | AsyncStrategy, Field(discriminator="type")
]


class MemStrategyConfig(BaseImmutableConfig):
    type: MemoryType = Field(
        default=MemoryType.DISABLED, description="Type of memory-based correction."
    )


class ModelConfig(BaseImmutableConfig):
    model_name: ModelType = Field(
        default=ModelType.CNN, description="Model architecture to use."
    )
    stress_test: bool = Field(
        default=False,
        description=(
            "If True, removes architectural stabilizers (GroupNorm/BatchNorm) "
            "to evaluate optimizer stability on ill-conditioned landscapes."
        ),
    )


class VisualizationConfig(BaseImmutableConfig):
    """
    Configuration for creating and saving data split
    and client arrival visualizations.

    Requires matplotlib if True.
    """

    visualize_data_split: bool = Field(
        default=False,
        description="Saves a visualization of the data split in .png format.",
    )

    visualize_client_arrivals: bool = Field(
        default=False,
        description="Saves a visualization of client arrivals in .png format.",
    )


class CheckpointConfig(BaseImmutableConfig):
    """
    Configuration for saving simulation state and model artifacts.
    Controls both resume-capability (heavy) and inference-ready (light) checkpoints.
    """

    interval_seconds: float = Field(
        default=400.0,
        gt=0,
        description="Wall-clock time interval between heavy checkpoints.",
    )
    keep_best: bool = Field(
        default=False, description="Save global model with highest accuracy."
    )


class OptimizationConfig(BaseImmutableConfig):
    learning_rate: float = Field(
        default=0.1, gt=0.0, description="Client-side learning rate for local SGD."
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight-decay parameter.",
    )
    num_local_steps: int = Field(
        default=100,
        gt=0,
        description="Number of local SGD steps performed by each client.",
    )
    num_workers: int = Field(
        default=0, ge=0, description="Subprocesses for data loading (0=main process)."
    )
    batch_size: int = Field(
        default=32, gt=0, description="Local batch size for client training."
    )


class EvaluationConfig(BaseImmutableConfig):
    batch_size: int = Field(
        default=32, gt=0, description="Batch size for server-side evaluation."
    )
    num_workers: int = Field(
        default=0, ge=0, description="Subprocesses for evaluation data loading."
    )


class DataConfig(BaseImmutableConfig):
    dataset: DatasetType = Field(
        default=DatasetType.MNIST,
        description="Target dataset for training and evaluation.",
    )
    dirichlet_alpha: float = Field(
        default=0.1,
        gt=0.0,
        description="Concentration parameter for Dirichlet non-IID partitioning.",
    )
    split_seed: int = Field(default=42, ge=0, description="Seed for random data split.")


class SimulationConfig(BaseImmutableConfig):
    device: DeviceType = Field(
        default=DeviceType.AUTO, description="Device for training and evaluation."
    )
    """
    Hardware device to use (cpu, cuda, mps).
    Defaults to AUTO, which picks the fastest available hardware.
    """
    num_clients: int = Field(
        default=10, gt=1, description="Total number of clients in the federated pool."
    )
    duration_sim_units: float = Field(
        default=100.0,
        gt=0,
        description="Target simulation duration (in arbitrary simulated time units).",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Hard timeout for the experiment (wall-clock seconds).",
    )
    client_rate_std: float = Field(
        default=1.0,
        ge=0.0,
        description="Std dev of zero-mean lognormal distribution for client update rates.",
    )
    rate_seed: int = Field(
        default=42, ge=0, description="Seed for random clock generation."
    )


class AppConfig(BaseImmutableConfig):
    comm_strategy: CommStrategyConfig = Field(default_factory=AsyncStrategy)
    mem_strategy: MemStrategyConfig = Field(default_factory=MemStrategyConfig)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @model_validator(mode="after")
    def check_logical_consistency(self) -> "AppConfig":
        if self.comm_strategy.type == "sync":
            if self.comm_strategy.sample_size > self.simulation.num_clients:
                raise ValueError(
                    f"Config Error: Sample size ({self.comm_strategy.sample_size}) cannot "
                    f"exceed total clients ({self.simulation.num_clients})."
                )
        return self

    @model_validator(mode="after")
    def check_model_compatibility(self) -> "AppConfig":
        dataset = self.data.dataset
        model = self.model.model_name

        if model.required_channels and dataset.num_channels != model.required_channels:
            raise ValueError(
                f"Config Error: '{model}' requires {model.required_channels} channel(s), "
                f"but '{dataset}' has {dataset.num_channels}. Choose a different model."
            )

        return self

    @model_validator(mode="after")
    def check_stress_test_compatibility(self) -> "AppConfig":
        model = self.model.model_name
        stress_test = self.model.stress_test

        if stress_test and not model.has_norm_layers:
            raise ValueError(
                f"Config Error: Stress test (removing norms) is not applicable to "
                f"'{model}' because it does not utilize normalization layers by default."
            )

        return self

    @model_validator(mode="after")
    def check_checkpoint_interval(self) -> "AppConfig":
        interval = self.checkpoints.interval_seconds
        timeout = self.simulation.timeout_seconds

        if interval >= timeout:
            logger.warning(
                f"Config Warning: Checkpoint interval {interval} is equal or greater than "
                f"simulation timeout {timeout}. The simulation will save a final checkpoint "
                f"upon termination, and no intermediate checkpoints."
            )

        return self

    @model_validator(mode="after")
    def sanitize_visualization_config(self) -> "AppConfig":
        limit = 150  # Threshold for readable plots

        disable_split = (
            self.visualization.visualize_data_split
            and self.simulation.num_clients > limit
        )
        disable_arrivals = (
            self.visualization.visualize_client_arrivals
            and self.simulation.num_clients > limit
        )

        if not (disable_split or disable_arrivals):
            return self

        if disable_split:
            logger.warning(
                f"Config Warning: Too many clients ({self.simulation.num_clients}) for "
                "data split visualization. Disabling to prevent unreadable plot."
            )
        if disable_arrivals:
            logger.warning(
                f"Config Warning: Too many clients ({self.simulation.num_clients}) for "
                "arrival visualization. Disabling to prevent unreadable plot."
            )

        new_viz_config = self.visualization.model_copy(
            update={
                "visualize_data_split": False
                if disable_split
                else self.visualization.visualize_data_split,
                "visualize_client_arrivals": False
                if disable_arrivals
                else self.visualization.visualize_client_arrivals,
            }
        )

        object.__setattr__(self, "visualization", new_viz_config)

        return self

    @model_validator(mode="after")
    def check_batch_size_validity(self) -> "AppConfig":
        train_size = self.data.dataset.train_size
        test_size = self.data.dataset.test_size
        batch_size = self.optimization.batch_size
        batch_size_eval = self.evaluation.batch_size

        if batch_size >= train_size:
            raise ValueError(
                f"Config Error: Batch size ({batch_size}) cannot be equal to or exceed "
                f"dataset size ({train_size}) for {self.data.dataset.name}."
            )

        if batch_size_eval >= test_size:
            raise ValueError(
                f"Config Error: Evaluation batch size ({batch_size_eval}) cannot be equal to or exceed "
                f"test dataset size ({test_size}) for {self.data.dataset.name}."
            )
        return self
