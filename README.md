# AFL-Sim: Asynchronous Federated Learning Simulator

> ⚠️ **Work in Progress**: This repository is currently under active construction. A formal v1.0.0 release is coming soon.

## Installation (From Source)

### Prerequisites
* Python 3.12+
* [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chariako/afl-sim.git
    cd afl-sim
    ```

2.  **Install the package:**

    **Option A: For Users (Run only)**
    ```bash
    uv sync
    # OR with pip
    pip install .
    ```

    **Option B: For Developers (Edit & Test)**
    ```bash
    uv sync --extra dev
    # OR with pip
    pip install -e ".[dev]"
    ```

## User Guide

**Usage**:

```console
$ afl-sim [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `run`: Start a new federated learning simulation.
* `resume`: Resume an existing simulation from folder.

### `afl-sim run`

Start a new federated learning simulation.

This command loads a YAML configuration, creates a timestamped results directory,
and initializes the simulation.

**Usage**:

```console
$ afl-sim run [OPTIONS] CONFIG_PATH
```

**Arguments**:

* `CONFIG_PATH`: Path to YAML config.  [required]

**Options**:

* `--output-dir PATH`: Base output directory.  [default: outputs]
* `--data-dir PATH`: Directory for saving input data, including datasets, data splits and simulated clocks.  [default: data]
* `--checkpoint-dir PATH`: Directory for saving and loading checkpoints.  [default: checkpoints]
* `--lr FLOAT`: Override client learning rate.
* `--tag TEXT`: Optional label for this run (e.g. &#x27;baseline&#x27;)
* `--dry-run`: Validate config and exit without running.
* `--help`: Show this message and exit.

### `afl-sim resume`

Resume an existing simulation from folder.

**Usage**:

```console
$ afl-sim resume [OPTIONS] OUTPUT_PATH
```

**Arguments**:

* `OUTPUT_PATH`: Path to the output directory (e.g. &#x27;outputs/2026...&#x27;) containing config.yaml.  [required]

**Options**:

* `--timeout FLOAT`: Override the wall-clock timeout (in seconds) for this specific resume session.
* `--sim-duration FLOAT`: Set a new experiment duration in simulated time units.
* `--help`: Show this message and exit.
