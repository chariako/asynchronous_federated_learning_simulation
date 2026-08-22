# AFL-Sim: Asynchronous Federated Learning Simulator

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-98%25-brightgreen.svg)
![CI Status](https://github.com/chariako/asynchronous_federated_learning_simulation/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Summary

AFL-Sim is a PyTorch-based simulation framework for benchmarking Federated Learning (FL) algorithms and systems. It addresses the challenges of benchmarking FL under constrained computational resources (e.g., a single GPU), especially when local training is executed asynchronously and as the number of clients increases. It achieves this by converting concurrent, parallel client training sessions into serial simulation events, allowing at most one client to use the accelerator at a time. Provided there are sufficient training samples and an appropriate configuration (e.g., small batch sizes and homogeneous data distributions) to support the requested split, AFL-Sim can successfully simulate up to 5,000-6,000 clients for standard datasets.

AFL-Sim is geared towards asynchronous FL, but supports both asynchronous and synchronous communication protocols for completeness. Moreover, it allows augmenting each client with an optional memory buffer for algorithms requiring additional client-side storage (e.g., algorithms utilizing previous states for error-correction purposes).

**📚 [Read the full AFL-Sim documentation here](https://chariako.github.io/asynchronous_federated_learning_simulation/)**

## Main Features

AFL-Sim provides the following features out of the box:

### Core FL Capabilities

- **Communication Modes**: Full support for both synchronous and asynchronous training.

- **Extensible Framework**: Developers can easily integrate custom algorithms, datasets and models into AFL-Sim.

- **Client Memory Augmentation**: Optional memory functionality for algorithms requiring additional storage.

- **Simulated Custom Client Latency**: Modeling of varying client latency times based on a user-supplied standard deviation parameter and a hardcoded mean value.

- **Standard Benchmark Implementation**: Equivalent versions under AFL-Sim's architecture can be recovered for several standard FL and distributed SGD algorithms by appropriately choosing the configuration parameters.

### Data & Models

- **Benchmarking Tasks**: Composed of select TorchVision datasets and curated vision models.

- **Custom Data Splits**: Dirichlet partitioning with user-provided parameters to simulate varying degrees of dataset heterogeneity among clients.

- **Visualizations**: Optional visual representation of data splits and simulated client arrivals.

### Engineering & Reliability

- **Hardware & OS Agnostic**: AFL-Sim is OS-independent. Hardware interfacing is handled natively by PyTorch, with full support for CUDA, MPS (Apple Silicon), and CPU devices.

- **Flexible Execution**: AFL-Sim can be run via the CLI using configuration YAML files with optional parameter overrides, or by importing individual modules like a standard Python library.

- **Strict Reproducibility**: Deterministic simulations conditional on three random seeds controlling data splitting, client arrivals, and PyTorch operations, respectively.

- **Dual Checkpointing System**: Handles interruptions gracefully by saving resumable simulation checkpoints at shutdown, in addition to periodic resumable checkpoints and best model checkpoints. All checkpointing occurs atomically to prevent corruption.

- **Dual Logging System**: Outputs both execution log artifacts and JSONL metrics for the global model to facilitate downstream processing.

- **Resource Efficiency**: Data splits and simulation clocks are saved to disk, preventing the expensive regeneration of reusable simulation artifacts.

## Quick Start

AFL-Sim uses [`uv`](https://docs.astral.sh/uv/) for dependency management; installation with `uv` is highly recommended, but `pip` is also supported.

First, clone the repository:

```bash
git clone https://github.com/chariako/asynchronous_federated_learning_simulation.git
cd asynchronous_federated_learning_simulation
```

Then, install the package based on your needs:

**Option 1:** Using `uv`

- For users (Run only):

```bash
uv sync
```

- For Developers (Edit & Test):

```bash
uv sync --all-groups
uv run pre-commit install
```

**Option 2:** Using `pip`

It is highly recommended to create and activate a virtual environment first (e.g., `python -m venv .venv && source .venv/bin/activate`).

- For users (Run only):

```bash
pip install .
```

- For Developers (Edit & Test):

```bash
pip install -e ".[dev]"
```

## Running Simulations

AFL-Sim provides two main functionalities over the CLI and through its Python API:

- Launching a new simulation.
- Resuming an existing simulation from disk.

> **Note for `uv` users**: If you installed the package using uv, prepend `uv run` to all `afl-sim` CLI commands below to perform the execution in the isolated environment.

### Launching a New Simulation

The fastest way to launch a new simulation is to create a configuration file with your desired parameters (e.g., `config.yaml`), and pass it to AFL-Sim over the CLI or inside a Python script.

You can optionally specify an output directory `output_dir` (default: `./outputs`), where the simulation logs and metadata will be stored under a unique identifier assigned to the simulation by AFL-Sim (e.g., `2026-08-08_12-31-42_936aed`).

- To launch a simulation from a YAML configuration over the CLI, use the `afl-sim run` command with the optional `--output-dir` flag:

    ```bash
    afl-sim run path/to/config.yaml --output-dir path/to/output_dir
    ```

- To launch a new simulation from a YAML configuration inside a Python script, import the `run_simulation` function from the `afl_sim` library and use the optional `output_dir` argument:

    ```python
    from afl_sim import run_simulation

    run_simulation(config="path/to/config.yaml", output_dir="path/to/output_dir")
    ```

An example YAML configuration file is provided in [configs/base_config.yaml](https://github.com/chariako/asynchronous_federated_learning_simulation/blob/main/configs/base_config.yaml).

### Resuming an Existing Simulation

You can resume a previous simulation using its unique identifier (e.g., `2026-08-08_12-31-42_936aed`) by providing the path to the corresponding folder inside the output directory specified when the simulation was first launched (e.g., `output_dir`). Resumes can be performed over the CLI or in a Python script.

- Resume a simulation from its unique output folder using the `afl-sim resume` CLI command:

    ```bash
    afl-sim resume path/to/output_dir/2026-08-08_12-31-42_936aed
    ```

- Import the `resume_simulation` command from the `afl_sim` library to resume a simulation from its unique output folder inside a Python script:

    ```python
    from afl_sim import resume_simulation

    resume_simulation(output_path="path/to/output_dir/2026-08-08_12-31-42_936aed")
    ```

## Documentation

The complete documentation for AFL-Sim is available on the project's [website](https://chariako.github.io/asynchronous_federated_learning_simulation/).

To learn more about using and extending AFL-Sim, visit the following pages:

- **[Setting Up a YAML Configuration](https://chariako.github.io/asynchronous_federated_learning_simulation/user_guide/configuration/)**: Read up on creating YAML configuration files and tuning their parameters, and review the supported options for hardware acceleration, models, datasets and implemented algorithms.
- **[Using the Python API](https://chariako.github.io/asynchronous_federated_learning_simulation/user_guide/python_api/)**: Learn how to construct custom configuration objects in lieu of YAML files, and use them to launch new simulations inside a Python script.
- **[Execution Guide](https://chariako.github.io/asynchronous_federated_learning_simulation/user_guide/execution/)**: Dive deeper into the run and resume functionalities of AFL-Sim, the types of artifacts AFL-Sim produces, and how it manages directories and storage.
- **[Implementation Notes](https://chariako.github.io/asynchronous_federated_learning_simulation/implementation/)**: Learn more about how AFL-Sim works under the hood.
- **[CLI Reference](https://chariako.github.io/asynchronous_federated_learning_simulation/reference/cli/)**: Read through the documentation for AFL-Sim's CLI commands, including input arguments, options and defaults.
- **[API Reference](https://chariako.github.io/asynchronous_federated_learning_simulation/reference/api/)**: Review the documentation for AFL-Sim's Python API, including all importable modules and their arguments.

## Contact & Support

For bugs or help with troubleshooting, open an [issue](https://github.com/chariako/asynchronous_federated_learning_simulation/issues).

For other questions or feedback, feel free to reach out directly at: chariako \[at\] u \[dot\] northwestern \[dot\] edu.
