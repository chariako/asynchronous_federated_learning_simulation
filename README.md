# AFL-Sim: Asynchronous Federated Learning Simulator

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-98%25-brightgreen.svg)
![CI Status](https://github.com/chariako/asynchronous_federated_learning_simulation/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Summary

AFL-Sim is a PyTorch-based simulation framework for benchmarking Federated Learning (FL) algorithms and systems. It addresses the challenges of benchmarking FL under constrained computational resources (e.g., a single GPU), especially when local training is executed asynchronously and as the number of clients increases. It achieves this by converting concurrent, parallel client training sessions into serial simulation events, allowing at most one client to use the accelerator at a time. Provided there are sufficient training samples and an appropriate configuration (e.g., small batch sizes and homogeneous data distributions) to support the requested split, AFL-Sim can successfully simulate up to 5,000-6,000 clients for standard datasets.

AFL-Sim is geared towards asynchronous FL, but supports both asynchronous and synchronous communication protocols for completeness. Moreover, it allows augmenting each client with an optional memory buffer for algorithms requiring additional client-side storage (e.g., algorithms utilizing previous states for error-correction purposes).

## Main Features

AFL-Sim provides the following features out of the box:

### Core FL Capabilities

- **Communication Modes**: Full support for both synchronous and asynchronous training.

- **Extensible Framework**: Developers can easily integrate custom algorithms, datasets and models into AFL-Sim.

- **Client Memory Augmentation**: Optional memory functionality for algorithms requiring additional storage.

- **Simulated Custom Client Latency**: Simulated varying client latency times based on a user-supplied standard deviation parameter.

- **Standard Benchmark Implementation**: Equivalent versions under AFL-Sim's architecture can be recovered for several standard FL and distributed SGD algorithms by appropriately choosing the configuration parameters.

### Data & Models

- **Benchmarking Tasks**: Composed of select TorchVision datasets and curated vision models.

- **Custom Data Splits**: Dirichlet partitioning with user-provided parameters to simulate varying degrees of dataset heterogeneity among clients.

- **Visualizations**: Optional visual representation of data splits and simulated client arrivals.

### Engineering & Reliability

- **Hardware & OS Agnostic**: AFL-Sim is OS-independent. Hardware interfacing is handled natively by PyTorch, with full support for CUDA, MPS (Apple Silicon), and CPU devices.

- **Flexible Execution**: AFL-Sim can be run via the CLI using configuration YAML files with optional parameter overrides, or by importing individual modules like a standard Python package.

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
uv sync --no-dev
```

- For Developers (Edit & Test):

```bash
uv sync
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

> **Note for uv users**: If you installed the package using uv, prepend `uv run --no-dev` to all `afl-sim` commands below to execute them in the isolated environment.

To launch a simulation via the CLI, create a configuration file with your desired parameters, e.g., `configs/config.yaml`, and run:

```bash
afl-sim run configs/config.yaml
```

An example YAML configuration file is provided in the [configs/](https://github.com/chariako/asynchronous_federated_learning_simulation/blob/main/configs/base_config.yaml) directory of this project.

AFL-Sim creates a unique ID for every simulation and a corresponding folder with the same name in a user-specified output directory (e.g., `outputs/`). To resume a previous simulation using its unique ID (e.g., 2026-08-08_12-31-42_936aed), run:

```bash
afl-sim resume outputs/2026-08-08_12-31-42_936aed
```

## Documentation

The project documentation is currently being finalized. An official release is coming soon.

## Contact & Support

For bugs or help with troubleshooting, open an [issue](https://github.com/chariako/asynchronous_federated_learning_simulation/issues).

For other questions or feedback, feel free to reach out directly at: chariako \[at\] u \[dot\] northwestern \[dot\] edu.
