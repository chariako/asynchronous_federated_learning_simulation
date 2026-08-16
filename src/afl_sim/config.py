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
    """
    Base Pydantic model enforcing strict, immutable configuration schemas.
    Forbids extra attributes and freezes instances upon creation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SyncStrategy(BaseImmutableConfig):
    """Configuration for synchronous federated learning strategies."""

    type: Literal["sync"] = "sync"
    sample_size: int = Field(
        default=3,
        gt=0,
        description="Number of clients sampled by the server at each round.",
    )

    @property
    def agg_target(self) -> int:
        """
        Retrieves the target number of client updates required for a global aggregation.

        Returns:
            int: The sample size per synchronous round.
        """
        return self.sample_size


class AsyncStrategy(BaseImmutableConfig):
    """Configuration for asynchronous federated learning strategies."""

    type: Literal["async"] = "async"
    buffer_size: int = Field(
        default=3,
        gt=0,
        description="Number of client updates that triggers a global model update.",
    )

    @property
    def agg_target(self) -> int:
        """
        Retrieves the target number of client updates required for a global aggregation.

        Returns:
            int: The configured buffer size limit.
        """
        return self.buffer_size


CommStrategyConfig = Annotated[
    SyncStrategy | AsyncStrategy, Field(discriminator="type")
]


class MemStrategyConfig(BaseImmutableConfig):
    """Configuration defining the memory tracking behavior of clients."""

    type: MemoryType = Field(
        default=MemoryType.DISABLED, description="Type of client memory augmentation."
    )


class ModelConfig(BaseImmutableConfig):
    """Configuration detailing the target neural network architecture."""

    model_name: ModelType = Field(
        default=ModelType.CNN, description="Model architecture to use."
    )


class VisualizationConfig(BaseImmutableConfig):
    """
    Configuration for creating and saving data split and client arrival visualizations.

    Note: Requires `matplotlib` to be installed if enabled.
    """

    visualize_data_split: bool = Field(
        default=False,
        description="Generates and saves a chart in .png format illustrating the distribution of dataset samples across the clients.",
    )

    visualize_client_arrivals: bool = Field(
        default=False,
        description="Generates and saves a timeline plot in .png format depicting the simulated arrival times and latencies of the clients.",
    )


class CheckpointConfig(BaseImmutableConfig):
    """
    Configuration for managing state serialization and disk I/O.
    Controls both interval-based heavy checkpoints and best-model artifacts.
    """

    interval_seconds: float = Field(
        default=400.0,
        gt=0,
        description="The interval (in wall-clock seconds) at which the simulator saves a resumable checkpoint.",
    )
    keep_best: bool = Field(
        default=False,
        description="If set to `True`, the simulator continuously saves a separate copy of the global model that achieved the highest accuracy on the test set.",
    )


class OptimizationConfig(BaseImmutableConfig):
    """Configuration for the local client-side optimization process."""

    learning_rate: float = Field(
        default=0.1,
        gt=0.0,
        description="The step size applied during local client training.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="The L2 penalty (weight decay) applied by the PyTorch optimizer to prevent overfitting.",
    )
    num_local_steps: int = Field(
        default=100,
        gt=0,
        description="The exact number of local SGD steps (batches) a client performs before communicating with the server.",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="The number of samples processed per local training step.",
    )


class EvaluationConfig(BaseImmutableConfig):
    """Configuration for the server-side global evaluation process."""

    batch_size: int = Field(
        default=32,
        gt=0,
        description="The number of test dataset samples processed per batch during global model evaluation (for metric generation).",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="The number of subprocesses used for data loading, corresponding to the PyTorch DataLoader parameter.",
    )


class DataConfig(BaseImmutableConfig):
    """Configuration for dataset selection and distributed partitioning."""

    dataset: DatasetType = Field(
        default=DatasetType.MNIST,
        description="The target dataset for the simulation.",
    )
    dirichlet_alpha: float = Field(
        default=0.1,
        gt=0.0,
        description="Dirichlet distribution parameter.",
    )
    split_seed: int = Field(
        default=42,
        ge=0,
        description="The random seed ensuring reproducibility during the dataset partitioning process.",
    )


class SimulationConfig(BaseImmutableConfig):
    """Configuration for the top-level simulation environment and hardware settings."""

    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="The hardware accelerator used for the simulation.",
    )
    num_clients: int = Field(default=10, gt=1, description="Total number of clients.")
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Simulation duration in wall-clock seconds.",
    )
    client_rate_std: float = Field(
        default=1.0,
        ge=0.0,
        description="Standard deviation of client latency.",
    )
    rate_seed: int = Field(
        default=42,
        ge=0,
        description="The random seed used to generate client arrival times and latency distributions.",
    )
    torch_seed: int = Field(
        default=42, ge=0, description="The random seed for all PyTorch operations."
    )


class AppConfig(BaseImmutableConfig):
    """
    The master configuration schema for the entire simulation application.
    Aggregates all sub-configurations and validates cross-domain logical consistency.
    """

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
        """
        Validates that the communication strategy does not request more clients
        than are available in the simulation pool.

        Returns:
            AppConfig: The validated configuration object.

        Raises:
            ValueError: If synchronous sample size exceeds total clients.
        """
        if (
            self.comm_strategy.type == "sync"
            and self.comm_strategy.sample_size > self.simulation.num_clients
        ):
            raise ValueError(
                f"Config Error: Sample size ({self.comm_strategy.sample_size}) cannot "
                f"exceed total clients ({self.simulation.num_clients})."
            )
        return self

    @model_validator(mode="after")
    def check_model_compatibility(self) -> "AppConfig":
        """
        Validates that the selected model architecture can accept the number
        of input channels provided by the selected dataset.

        Returns:
            AppConfig: The validated configuration object.

        Raises:
            ValueError: If a strict channel mismatch occurs.
        """
        dataset = self.data.dataset
        model = self.model.model_name

        if model.required_channels and dataset.num_channels != model.required_channels:
            raise ValueError(
                f"Config Error: '{model}' requires {model.required_channels} channel(s), "
                f"but '{dataset}' has {dataset.num_channels}. Choose a different model."
            )

        return self

    @model_validator(mode="after")
    def check_checkpoint_interval(self) -> "AppConfig":
        """
        Checks if the checkpoint interval exceeds or equals the simulation timeout,
        warning the user that intermediate checkpoints will not be saved.

        Returns:
            AppConfig: The validated configuration object.
        """
        interval = self.checkpoints.interval_seconds
        timeout = self.simulation.timeout_seconds

        if interval >= timeout:  # pragma: no branch
            logger.warning(
                f"Config Warning: Checkpoint interval '{interval}' is equal or greater than the "
                f"simulation timeout '{timeout}'. The simulation will save a final checkpoint "
                f"upon termination or interruption, and no intermediate checkpoints."
            )

        return self

    @model_validator(mode="after")
    def sanitize_visualization_config(self) -> "AppConfig":
        """
        Disables visualizations automatically if the client count exceeds a readability
        threshold (150 clients) to prevent resource exhaustion and unreadable plots.

        Returns:
            AppConfig: The sanitized configuration object with updated visualization flags.
        """
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

        if disable_split:  # pragma: no branch
            logger.warning(
                f"Config Warning: Too many clients ({self.simulation.num_clients}) for "
                "data split visualization. Disabling to prevent unreadable plot."
            )
        if disable_arrivals:  # pragma: no branch
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
        """
        Validates that training and evaluation batch sizes do not exceed
        their respective total dataset sizes.

        Returns:
            AppConfig: The validated configuration object.

        Raises:
            ValueError: If a configured batch size exceeds the available dataset size.
        """
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
