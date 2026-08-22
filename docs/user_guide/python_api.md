# Using the Python API

In addition to the CLI functionality, AFL-Sim supports launching new simulations or resuming existing ones from disk using Python scripts. To do this, import AFL-Sim as a Python library using the command `import afl_sim`.

For the full list of modules that can be imported from AFL-Sim and their descriptions, check the [API Reference](../reference/api.md).

!!! tip

    If you installed AFL-Sim with `uv`, prepend `uv run` when running a Python script that imports AFL-Sim:

    ``` bash
    uv run python launch_new_from_app_config.py
    ```

---

## Launching a New Simulation

Launching simulations from a Python script provides greater flexibility than using the CLI. In addition to providing a path to a YAML configuration file, users can manually configure an `AppConfig` object whose attributes correspond directly to the parameters in a YAML configuration.

To launch a new simulation with the Python API, use the `run_simulation` function from the `afl_sim` library. Configuration parameters can be supplied to `run_simulation` using either `AppConfig` objects or paths to YAML configuration files.

!!! tip

    - For a detailed description of `afl_sim.AppConfig`, `afl_sim.run_simulation` and their attributes, see the dedicated sections in the [API Reference](../reference/api.md).
    - Check the [Parameter Reference](configuration.md#parameter-reference) section in [Setting Up a YAML Configuration](configuration.md) for a deep dive into the configuration parameters.

### Using `AppConfig` Objects

An annotated example of a Python script that launches a new simulation by constructing and supplying an `AppConfig` object is provided below.

``` python title="launch_new_from_app_config.py" linenums="1"
# Import necessary modules from afl_sim
from afl_sim import (
    AppConfig,
    AsyncStrategy,  # or SyncStrategy for synchronous mode
    CheckpointConfig,
    DataConfig,
    DatasetType,
    DeviceType,
    EvaluationConfig,
    MemoryType,
    MemStrategyConfig,
    ModelConfig,
    ModelType,
    OptimizationConfig,
    SimulationConfig,
    VisualizationConfig,
    run_simulation,
)

# ------------------------------------------------------------------------------
# 1. Create Attributes of AppConfig
# ------------------------------------------------------------------------------
# --- Communication Strategy ---
# TIP: To configure a synchronous communication strategy, import
# the module SyncStrategy from afl_sim. Then configure as follows:
# comm_strategy = SyncStrategy(
#     type="sync",                        # Synchronous FL mode.
#     sample_size=3                       # Clients sampled per round.
# )
comm_strategy = AsyncStrategy(
    type="async",  # Asynchronous FL mode.
    buffer_size=3,  # Server buffer size trigger.
)

# --- Memory Strategy ---
mem_strategy = MemStrategyConfig(
    type=MemoryType.MODELS  # Type of memory-based correction.
)

# --- Dataset ---
data_config = DataConfig(
    dataset=DatasetType.MNIST,  # Dataset name.
    dirichlet_alpha=0.1,  # Dirichlet distribution parameter.
    split_seed=42,  # Seed for data partitioning.
)

# --- Model ---
model_config = ModelConfig(
    model_name=ModelType.LOG_REG  # Model architecture to use.
)

# --- Simulation Parameters ---
sim_config = SimulationConfig(
    device=DeviceType.AUTO,  # Simulation device.
    num_clients=10,  # Number of clients.
    timeout_seconds=1000.0,  # Simulation duration in seconds.
    client_rate_std=1.0,  # Standard deviation of client latency.
    rate_seed=42,  # Seed for client arrival generation.
    torch_seed=42,  # Seed for PyTorch operations.
)

# --- Evaluation (for metric generation) ---
eval_config = EvaluationConfig(
    batch_size=64,  # Evaluation batch size.
    num_workers=0,  # PyTorch Dataloader num_workers param.
)

# --- Client-side Optimization ---
optim_config = OptimizationConfig(
    learning_rate=1.0,  # Client learning rate.
    weight_decay=0.1,  # PyTorch Optimizer weight_decay param.
    num_local_steps=50,  # Number of local SGD steps.
    batch_size=32,  # Training batch size.
)

# --- Checkpoints ---
checkpoint_config = CheckpointConfig(
    interval_seconds=500.0,  # Seconds between resumable checkpoints.
    keep_best=False,  # Save global model with highest accuracy.
)

# --- Visualization ---
vis_config = VisualizationConfig(
    visualize_client_arrivals=False,  # Saves a visualization of client arrivals.
    visualize_data_split=False,  # Saves a visualization of the data split.
)

# ------------------------------------------------------------------------------
# 2. Create AppConfig Object
# ------------------------------------------------------------------------------
config = AppConfig(
    comm_strategy=comm_strategy,
    mem_strategy=mem_strategy,
    data=data_config,
    model=model_config,
    simulation=sim_config,
    evaluation=eval_config,
    optimization=optim_config,
    checkpoints=checkpoint_config,
    visualization=vis_config,
)

# ------------------------------------------------------------------------------
# 3. Launch the Simulation
# ------------------------------------------------------------------------------
# ATTENTION: Learning rate overrides are only allowed when the configuration
# parameters are provided with a YAML file. Attempting to override the
# learning rate of an AppConfig object will trigger an error. To modify the
# learning rate, edit the optimization config input of AppConfig directly.
run_simulation(
    config=config,
    output_dir="outputs",  # Directory for storing logs and metadata.
    data_dir="data",  # Directory for storing input artifacts.
    checkpoint_dir="checkpoints",  # Directory for storing checkpoints.
)
```

!!! info

    - The inputs `output_dir`, `data_dir` and `checkpoint_dir` to `run_simulation` are optional. If they are not provided, AFL-Sim will default to `./outputs`, `./data`, and `./checkpoints`. AFL-Sim will automatically create non-existing directories.
    - For a complete list of supported options for `ModelType`, `DeviceType`, `DatasetType` and `MemoryType`, check the corresponding sections in the [API Reference](../reference/api.md).

!!! danger

    The `learning_rate` argument of `run_simulation` permits overriding the  `learning_rate` parameter of a YAML configuration (see the [next section](#using-yaml-configurations) for launching new simulations using YAML files). This behavior is disabled when `AppConfig` objects are provided to `run_simulation`, and attempting to override the `learning_rate` of an `AppConfig` object (`config.optimization.learning_rate` in the script above) will raise a `RuntimeError`. To modify the learning rate, edit the `AppConfig` object directly.

### Using YAML Configurations

An example script launching a new simulation from a YAML configuration file can be found below. Note that `learning_rate` overrides are allowed in this instance.

``` python title="launch_new_from_yaml.py" linenums="1"
from afl_sim import run_simulation  # import run_simulation

run_simulation(
    config="configs/base_config.yaml",  # str or pathlib.Path accepted
    output_dir="outputs",
    data_dir="data",
    checkpoint_dir="checkpoints",
    learning_rate=0.001,  # Overriding the learning rate allowed
)
```

!!! info

    Check [Setting Up a YAML Configuration](configuration.md) for guidance on creating YAML configurations.

---

## Resuming a Simulation

AFL-Sim assigns a unique identifier to every simulation (e.g., `2026-08-08_12-31-42_936aed`) and creates a folder with that name in the _output_ directory specified when the simulation was launched (e.g., `outputs/`). To resume an existing simulation from a checkpoint saved to disk, import the `resume_simulation` function from the `afl_sim` library and provide it with the path to this output folder:

``` python title="resume_existing.py" linenums="1"
from afl_sim import resume_simulation  # import resume_simulation

# output_path can be str or pathlib.Path
resume_simulation(
    output_path="outputs/2026-08-08_12-31-42_936aed",
    timeout=1000.0,  # optional timeout (simulation duration in seconds) override
)
```

!!! tip

    - For a description of `afl_sim.resume_simulation` and its arguments, check the corresponding section in the [API Reference](../reference/api.md).
    - The section [Launching a New Simulation](execution.md#launching-a-new-simulation) in the [Execution Guide](execution.md) details AFL-Sim directory organization, including output directories and simulation artifact storage.
