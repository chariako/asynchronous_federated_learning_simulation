# `afl-sim`

Federated Learning Simulation CLI.

Provides commands to configure, run, and resume discrete-event
federated learning simulations.

**Usage**:

```console
$ afl-sim [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `run`: Starts a new federated learning simulation.
* `resume`: Resumes an existing simulation from a previously saved output directory.

## `afl-sim run`

Starts a new federated learning simulation.

This command loads a YAML configuration, creates a timestamped results directory,
initializes the data partitions and simulation environment, and begins the run.

Args:
    config_path (Path): Path to the YAML configuration file.
    output_dir (Path): Base directory for all output runs.
    data_dir (Path): Directory for saving/loading datasets, splits, and clocks.
    checkpoint_dir (Path): Base directory for saving checkpoints.
    learning_rate (float | None): Optional override for the client learning rate.
    tag (str | None): Optional label appended to the run directory name.
    dry_run (bool): If True, validates the config and exits without starting.

Raises:
    typer.Exit: Exits with code 1 if configuration validation, filesystem operations,
        or the simulation run fails. Exits with code 0 on a successful dry run.

**Usage**:

```console
$ afl-sim run [OPTIONS] {config_path}
```

**Arguments**:

* `config_path`: Path to YAML config.  \[required\]

**Options**:

* `--output-dir <path>`: Base output directory.  \[default: outputs\]
* `--data-dir <path>`: Directory for saving input data, including datasets, data splits and simulated clocks.  \[default: data\]
* `--checkpoint-dir <path>`: Directory for saving and loading checkpoints.  \[default: checkpoints\]
* `--lr <float>`: Override client learning rate.
* `--tag <str>`: Optional label for this run (e.g. &#x27;baseline&#x27;)
* `--dry-run`: Validate config and exit without running.
* `--help`: Show this message and exit.

## `afl-sim resume`

Resumes an existing simulation from a previously saved output directory.

Restores the configuration, locates the appropriate datasets and checkpoints
from the runtime metadata, and continues the simulation loop from the exact
global index where it last stopped.

Args:
    output_path (Path): Path to the existing run directory containing `config.yaml`.
    timeout (float | None): Optional override for the wall-clock timeout in seconds
        for this specific session.

Raises:
    typer.Exit: Exits with code 1 if configuration/metadata validation, filesystem operations,
        or the simulation run fails.

**Usage**:

```console
$ afl-sim resume [OPTIONS] {output_path}
```

**Arguments**:

* `output_path`: Path to the output directory (e.g. &#x27;outputs/2026...&#x27;) containing config.yaml.  \[required\]

**Options**:

* `--timeout <float>`: Override the wall-clock timeout (in seconds) for this specific resume session.
* `--help`: Show this message and exit.
