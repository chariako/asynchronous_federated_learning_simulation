# Setting Up a YAML Configuration

AFL-Sim supports launching simulations over the CLI or in a Python script using YAML configuration files. This guide details the structure of a configuration file and its parameters. It also explains how to recover standard algorithmic benchmarks by tuning the configuration parameters.

## Overview

A configuration file defines all aspects of a fresh AFL-Sim run, including hardware/reproducibility settings, federated learning communication modes, dataset partitioning, simulated system latency, and checkpoint management.

A configuration file can be passed to the simulator using the CLI or inside a Python script:

=== "CLI"

    Use the CLI command `afl-sim run` to launch a new simulation from a YAML configuration:

    ``` bash
    afl-sim run path/to/config.yaml
    ```

    !!! info

        Check the [`run`](../reference/cli.md#afl-sim-run) command section in the [CLI Reference](../reference/cli.md) for the full list of arguments.

=== "Python"

    To launch a new simulation from a YAML configuration in a Python script, import the `run_simulation` function from the `afl_sim` library:

    ``` python title="run_new_from_yaml.py" linenums="1"
    from afl_sim import run_simulation  # import the `run_simulation` function

    # run with the YAML path as input (str or pathlib.Path accepted)
    run_simulation(config="path/to/config.yaml")
    ```

    !!! info

        For the full list of `run_simulation` arguments and defaults, check its documentation in the [API Reference](../reference/api.md).

---

## Configuration Template

A complete YAML configuration template is provided below. You can copy this block directly to create your own configuration file.

``` yaml title="config.yaml" linenums="1"
# ------------------------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------------------------
data:
  dataset: mnist                    # Dataset name.
  dirichlet_alpha: 0.1              # Dirichlet distribution parameter.
  split_seed: 42                    # Seed for data partitioning.

# ------------------------------------------------------------------------------
# 2. Simulation Parameters
# ------------------------------------------------------------------------------
simulation:
  device: auto                      # Simulation device.
  num_clients: 10                   # Number of clients.
  timeout_seconds: 300.0            # Simulation duration in seconds.
  client_rate_std: 1.0              # Standard deviation of client latency.
  rate_seed: 42                     # Seed for client arrival generation.
  torch_seed: 42                    # Seed for PyTorch operations.

# ------------------------------------------------------------------------------
# 3. Model
# ------------------------------------------------------------------------------
model:
  model_name: cnn                   # Model architecture to use.

# ------------------------------------------------------------------------------
# 4. Client Memory Augmentation
# ------------------------------------------------------------------------------
mem_strategy:
  type: disabled                    # Type of memory-based correction.

# ------------------------------------------------------------------------------
# 5. Communication Strategy
# ------------------------------------------------------------------------------
# ATTENTION: Only one communication strategy block (async or sync) is permitted.
# To use the synchronous strategy, comment out the async block and uncomment
# the sync block.

# 5a. Asynchronous Mode
comm_strategy:
  type: async                       # Asynchronous FL mode.
  buffer_size: 3                    # Server buffer size trigger.

# 5b. Synchronous Mode
# comm_strategy:
#   type: sync                      # Synchronous FL mode.
#   sample_size: 3                  # Clients sampled per round.

# ------------------------------------------------------------------------------
# 6. Client Optimizer Settings
# ------------------------------------------------------------------------------
optimization:
  learning_rate: 0.1                # Client learning rate.
  weight_decay: 0.0                 # PyTorch Optimizer weight_decay param.
  num_local_steps: 100              # Number of local SGD steps.
  batch_size: 32                    # Training batch size.

# ------------------------------------------------------------------------------
# 7. Evaluation Parameters (for metric generation)
# ------------------------------------------------------------------------------
evaluation:
  batch_size: 32                    # Evaluation batch size.
  num_workers: 0                    # PyTorch Dataloader num_workers param.

# ------------------------------------------------------------------------------
# 8. Checkpoints
# ------------------------------------------------------------------------------
checkpoints:
  keep_best: False                  # Save global model with highest accuracy.
  interval_seconds: 400.0           # Seconds between resumable checkpoints.

# ------------------------------------------------------------------------------
# 9. Visualization
# ------------------------------------------------------------------------------
visualization:
  visualize_data_split: False       # Saves a visualization of the data split.
  visualize_client_arrivals: False  # Saves a visualization of client arrivals.
```

!!! tip "Parameter Overrides And Dry Runs"

    AFL-Sim allows overriding the YAML `learning_rate` parameter for easy sweeps. Moreover, you can quickly check the correctness of a YAML configuration without running a full simulation by performing a dry run.

    === "CLI"

        Use the `--lr` and `--dry-run` flags to override the YAML `learning_rate` and perform a dry run, respectively:

        ``` bash
        afl-sim run path/to/config.yaml --lr 0.01 --dry-run
        ```

    === "Python"

        Use the `learning_rate` argument of `run_simulation` to override the YAML value. To perform a dry run, set the `dry_run` input argument to `True`:

        ``` python
        from afl_sim import run_simulation

        run_simulation(config="path/to/config.yaml", learning_rate=0.01, dry_run=True)
        ```

---

## Parameter Reference

### Dataset

| Parameter         | Type    | Default | Description                                                                                                  |
| :---------------- | :------ | :------ | :----------------------------------------------------------------------------------------------------------- |
| `dataset`         | String  | `mnist` | The target dataset for the simulation. Supported options: (`mnist`, `fashion_mnist`, `cifar10`, `cifar100`). |
| `dirichlet_alpha` | Float   | `0.1`   | Dirichlet distribution parameter.                                                                            |
| `split_seed`      | Integer | `42`    | The random seed ensuring reproducibility during the dataset partitioning process.                            |

!!! tip

    Higher values of `dirichlet_alpha` produce more homogeneous data distributions across clients, moving closer to an Independent and Identically Distributed (IID) setting.

!!! warning

    AFL-Sim requires each client to have a minimum number of samples equal to the training batch size. Because the training dataset is finite, this constraint becomes harder to satisfy if you increase the number of clients or decrease the Dirichlet parameter. If the simulator exceeds its maximum attempts to generate a valid data split, it will abort and prompt you to increase `dirichlet_alpha`.

### Simulation Parameters

| Parameter         | Type    | Default | Description                                                                                |
| :---------------- | :------ | :------ | :----------------------------------------------------------------------------------------- |
| `device`          | String  | `auto`  | The hardware accelerator used for the simulation. Options: (`auto`, `cpu`, `cuda`, `mps`). |
| `num_clients`     | Integer | `10`    | Total number of clients.                                                                   |
| `timeout_seconds` | Float   | `300.0` | Simulation duration in wall-clock seconds.                                                 |
| `client_rate_std` | Float   | `1.0`   | Standard deviation of client latency.                                                      |
| `rate_seed`       | Integer | `42`    | The random seed used to generate client arrival times and latency distributions.           |
| `torch_seed`      | Integer | `42`    | The random seed for all PyTorch operations.                                                |

!!! tip "Clock Tips"

    - Decreasing `client_rate_std` reduces the variance in client latency, resulting in more consistent response times across the network (i.e., reducing the "straggler effect").
    - The median client latency is hardcoded to `1` simulation time unit.

!!! info

    Setting the `device` parameter to `auto` selects the fastest available hardware accelerator. For example, AFL-Sim will prioritize `cuda` (NVIDIA GPUs) or `mps` (Apple Silicon) over standard `cpu` execution if those accelerators are detected.

### Model

| Parameter    | Type   | Default | Description                                                        |
| :----------- | :----- | :------ | :----------------------------------------------------------------- |
| `model_name` | String | `cnn`   | Model architecture to use. Options: (`logreg`, `cnn`, `resnet18`). |

!!! warning

    AFL-Sim enforces strict dataset-model compatibility rules. While `logreg` and `cnn` can be paired with any supported dataset, `resnet18` can only be used with `cifar10` and `cifar100`. Attempting to pair `resnet18` with `mnist` or `fashion_mnist` will cause the configuration validation to fail.

### Client Memory Augmentation

| Parameter | Type   | Default    | Description                                                             |
| :-------- | :----- | :--------- | :---------------------------------------------------------------------- |
| `type`    | String | `disabled` | Type of client memory augmentation (`disabled`, `models`, `gradients`). |

The `type` parameter defines how clients compute the updates they send to the server:

- **`disabled`**: Standard Federated Learning. Clients send the local delta (i.e., the difference between their newly trained local model and the initial global model received before training).
- **`models`**: Clients maintain a local history, updating the server with the difference between their _most recent_ trained local model and their _second most recent_ trained local model.
- **`gradients`**: Clients maintain a gradient (delta) history, updating the server with the difference between their _most recent_ local deltas and their _second most recent_ local deltas.

!!! note

    AFL-Sim implements memory augmentation strictly on the client side. However, this approach is mathematically equivalent to a server-side memory architecture where the server maintains a dedicated memory buffer for every individual client. In that alternative setup, clients would simply send their freshest updates (models or gradients), and the server would track the stale versions to compute the differences itself.

### Communication Strategy

!!! failure

    You must define exactly one communication strategy block (`async` OR `sync`) per configuration file. Attempting to define both will cause the configuration validation to fail and abort the simulation.

=== "Asynchronous mode"

    | Parameter     | Type    | Default | Description                                                   |
    | :------------ | :------ | :------ | :------------------------------------------------------------ |
    | `type`        | String  | `async` | Asynchronous FL mode.                                         |
    | `buffer_size` | Integer | `3`     | Number of client updates that triggers a global model update. |

=== "Synchronous mode"

    | Parameter     | Type    | Default | Description                                            |
    | :------------ | :------ | :------ | :----------------------------------------------------- |
    | `type`        | String  | `sync`  | Synchronous FL mode.                                   |
    | `sample_size` | Integer | `3`     | Number of clients sampled by the server at each round. |

In synchronous mode the server selects the `sample_size` subset of clients uniformly at random (i.e., with equal probability). In asynchronous mode, there is no central sampling; the server processes updates as they arrive in its buffer, and updates the global model after receiving a number of updates equal to `buffer_size`.

### Client Optimizer Settings

| Parameter         | Type    | Default | Description                                                                                           |
| :---------------- | :------ | :------ | :---------------------------------------------------------------------------------------------------- |
| `learning_rate`   | Float   | `0.1`   | The step size applied during local client training.                                                   |
| `weight_decay`    | Float   | `0.0`   | The L2 penalty (weight decay) applied by the PyTorch optimizer to prevent overfitting.                |
| `num_local_steps` | Integer | `100`   | The exact number of local SGD steps (batches) a client performs before communicating with the server. |
| `batch_size`      | Integer | `32`    | The number of samples processed per local training step.                                              |

!!! note

    The local optimization algorithm is pre-set to Stochastic Gradient Descent (SGD). Adaptive algorithms such as Adam and its variants are not currently supported for local client training. In non-IID federated settings, naively applying adaptive optimizers locally causes [objective inconsistency](https://arxiv.org/abs/2106.02305), arbitrarily distorting the geometry of the global objective function.

### Evaluation Parameters

| Parameter     | Type    | Default | Description                                                                                                    |
| :------------ | :------ | :------ | :------------------------------------------------------------------------------------------------------------- |
| `batch_size`  | Integer | `32`    | The number of test dataset samples processed per batch during global model evaluation (for metric generation). |
| `num_workers` | Integer | `0`     | The number of subprocesses used for data loading, corresponding to the PyTorch DataLoader parameter.           |

!!! tip

    Leaving `num_workers` at `0` means data loading occurs sequentially on the main process. If you notice that evaluating the global model is creating a bottleneck and slowing down your simulation speed, try increasing this value (e.g., to `2` or `4`) to enable multi-process data loading.

### Checkpoints

| Parameter          | Type    | Default | Description                                                                                                                                |
| :----------------- | :------ | :------ | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `keep_best`        | Boolean | `False` | If set to `True`, the simulator continuously saves a separate copy of the global model that achieved the highest accuracy on the test set. |
| `interval_seconds` | Float   | `400.0` | The interval (in wall-clock seconds) at which the simulator saves a resumable checkpoint.                                                  |

!!! tip

    To effectively disable periodic checkpoints, set `interval_seconds` to a value greater than `timeout_seconds` (defined in the simulation parameters). Regardless of this interval, AFL-Sim will always attempt to save a resumable checkpoint once the simulation timeout is reached, or in the event of a user/system interruption (e.g., `Ctrl+C`).

### Visualization

| Parameter                   | Type    | Default | Description                                                                                                            |
| :-------------------------- | :------ | :------ | :--------------------------------------------------------------------------------------------------------------------- |
| `visualize_data_split`      | Boolean | `False` | Generates and saves a chart in .png format illustrating the distribution of dataset samples across the clients.        |
| `visualize_client_arrivals` | Boolean | `False` | Generates and saves a timeline plot in .png format depicting the simulated arrival times and latencies of the clients. |

!!! note

    Visualizations are created with [Matplotlib](https://matplotlib.org/).

---

## Implemented Algorithms

AFL-Sim provides a unified framework capable of recovering equivalent versions of several standard Federated Learning and distributed/parallel SGD algorithmic benchmarks. This is achieved by tuning three key configuration parameters: communication strategy, memory strategy, and the number of local steps.

Before mapping configurations to specific algorithms, it is important to understand how AFL-Sim's architecture manages communication and memory:

| Functionality           | Implementation In AFL-Sim                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Asynchrony**          | AFL-Sim's asynchronous mode is strictly _client-driven_. At no point does the server actively sample clients to send them the global model. Instead, the server broadcasts the global model to all clients once at initialization, and then accepts incoming client updates as they arrive. Whenever a client returns an update to the server, it receives the latest global model in return.                                                                                                                                                                                                                                                                                                  |
| **Memory Augmentation** | With the exception of [AREA](https://arxiv.org/abs/2405.10123), the algorithms listed in the table below natively implement memory augmentation on the server side, allocating dedicated buffers for each client. AFL-Sim, conversely, implements memory augmentation _strictly on the client side_. As noted in the [Client Memory Augmentation](#client-memory-augmentation) section, the two architectures are mathematically equivalent. However, in real-world systems, client-side augmentation could drastically reduce memory requirements at the server and enhance scalability as the number of clients grows, provided clients have enough disk space to store a copy of the model. |

!!! info

    For a deeper dive into the specific order of operations in asynchronous mode, refer to the [Simulated Federated Learning Workflows](../implementation/workflows.md) section in the implementation notes of AFL-Sim.

### Algorithm Recovery Matrix

Although the underlying memory-augmentation (or lack thereof) remains equivalent up to constants, due to the architectural deviations in the communication protocol and in the location of memory-augmentation described above, AFL-Sim can be said to return methods that are "like" the algorithms below (e.g., [FedBuff](https://arxiv.org/abs/2106.06639)-like), rather than exact replicas. The exception is [AREA](https://arxiv.org/abs/2405.10123), which utilizes the communication and memory architecture native to AFL-Sim.

| `comm_strategy.type` | `mem_strategy.type` | `optimization.num_local_steps` | Recovered Algorithm                                                                                       |
| :------------------- | :------------------ | :----------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `sync`               | `disabled`          | `1`                            | [Synchronous SGD](https://arxiv.org/abs/1106.5730)                                                        |
| `async`              | `disabled`          | `1`                            | [Asynchronous SGD](https://arxiv.org/abs/1710.06952)                                                      |
| `sync`               | `disabled`          | `> 1`                          | [FedAvg](https://arxiv.org/abs/1602.05629)                                                                |
| `async`              | `disabled`          | `> 1`                          | [FedBuff](https://arxiv.org/abs/2106.06639)-like, [AFA-CD](https://arxiv.org/abs/2108.09875)-like         |
| `sync`               | `gradients`         | `> 1`                          | [MIFA](https://arxiv.org/abs/2106.04159)-like, [FedVARP](https://arxiv.org/abs/2207.14130)-like           |
| `async`              | `gradients`         | `> 1`                          | [AFA-CS](https://arxiv.org/abs/2108.09875)-like, [CA2FL](https://openreview.net/forum?id=4aywmeb97I)-like |
| `async`              | `models`            | `> 1`                          | [AREA](https://arxiv.org/abs/2405.10123)                                                                  |

!!! note "Note on MIFA-like Recovery"

    [MIFA](https://arxiv.org/abs/2106.04159) was proposed to tackle client unavailability in the synchronous setting, hence this recovery assumes that the sampled clients are the available clients at each round. Because AFL-Sim hardcodes uniform client sampling under synchronous communication, this yields a particularly benign version of client unavailability where clients are equally likely to be unavailable.
