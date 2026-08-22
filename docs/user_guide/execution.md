# Execution Guide

This guide explores AFL-Sim execution, including directory management and the creation and storage of simulation artifacts. It details directory organization and disk space demands, and outlines the format of simulation outputs, logs and checkpoints.

## Overview

AFL-Sim provides two main functionalities:

- Run a new simulation.
- Resume an existing simulation from a system checkpoint saved to disk.

These functionalities are provided over the CLI or by importing AFL-Sim like a standard Python library.

!!! info "Executing AFL-Sim"

    - **Over the CLI:** For a full list of CLI commands and their arguments, check the dedicated [CLI Reference](../reference/cli.md).
    - **In a Python script:** Check the [API Reference](../reference/api.md) for the full list of importable modules, and [Using the Python API](python_api.md) for example scripts.

---

## Launching a New Simulation

AFL-Sim generates and stores various types of artifacts and metadata. To ensure proper storage of these items, AFL-Sim allows the user to optionally specify the following directories when launching a new simulation:

| Directory                                 | Description                                                                                                                                                                                                                                                                                 |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Directory (data_dir)**             | Stores reusable simulation inputs such as raw datasets, data partitions, and simulated client arrivals (organized in a custom "clock" structure). While partitions and clocks are lightweight `.npz` objects, raw datasets may have non-negligible storage demands depending on their size. |
| **Output Directory (output_dir)**         | Stores simulation logs, a copy of the effective YAML configuration, and metadata under a unique simulation identifier. This directory generally contains only lightweight objects.                                                                                                          |
| **Checkpoint Directory (checkpoint_dir)** | Stores periodic (or shutdown) resumable checkpoints, as well as optional best-model checkpoints. This directory may have significant space requirements depending on model size, communication mode (async vs. sync), number of clients, and client memory augmentation settings.           |

=== "CLI"

    To run a new simulation with custom paths over the CLI, use the `afl-sim run` command with the respective path flags:

    ``` bash
    afl-sim run path/to/config.yaml --data-dir path/to/data_dir --output-dir path/to/output_dir --checkpoint-dir path/to/checkpoint_dir
    ```

    !!! tip

        Check the [`run`](../reference/cli.md#afl-sim-run) command documentation in the [CLI Reference](../reference/cli.md) for the full list of arguments, including parameter overrides.

=== "Python"

    To run a new simulation with custom paths in a Python script, import the `run_simulation` function from the `afl_sim` library and use the respective arguments:

    ``` python
    from afl_sim import run_simulation  # import run_simulation

    # Both str and pathlib.Path formats accepted for input paths
    run_simulation(
        config="path/to/config.yaml",
        output_dir="path/to/output_dir",
        data_dir="path/to/data_dir",
        checkpoint_dir="path/to/checkpoint_dir",
    )
    ```

    !!! tip

        Check the `run_simulation` function documentation in the [API Reference](python_api.md) for the full list of arguments, including parameter overrides.

!!! info

    If directories are omitted, AFL-Sim defaults to `./data`, `./outputs`, and `./checkpoints`. AFL-Sim will automatically create non-existing directories.

---

### Run Artifacts

Below is the full list of artifacts generated when a new simulation is launched.

#### Input Data Artifacts

The following artifacts are generated and stored in the user-specified `data_dir`:

| Artifact              | Description                                                                                                                                                                                                                                                                                                                                                   |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Raw Datasets**      | Stored in the predefined TorchVision format.                                                                                                                                                                                                                                                                                                                  |
| **Data Partitions**   | Saved in a subdirectory `data_dir/partitions/` under a unique parameter hash (e.g., `data_dir/partitions/1c78ef3d7fa50f8d`). This folder contains the `.npz` partition data, a `.json` metadata file for inspection, and an optional `.png` visualization of the split.                                                                                       |
| **Simulation Clocks** | Pre-generated client arrivals stored in `data_dir/clocks/` under a unique hash (e.g., `fa104d8089fba76b`). The folder includes `.json` metadata, an optional `.png` visualization, and the first `3,000` simulation events saved as `chunk0.npz`. AFL-Sim will automatically generate additional clock chunks (e.g., `chunk1.npz`) if more events are needed. |

!!! info

    For more information on simulation clock generation, check the [Client Latency Distributions](../implementation/modeling.md#client-latency-distributions) section in the implementation notes of AFL-Sim.

!!! tip

    AFL-Sim pre-generates partitions and client arrivals before execution begins. Although these objects are reproducible based on random seeds, they are saved to disk so they can be quickly reused by other simulations with partial configuration overlaps. Note that simulations should share the same `data_dir` to enable artifact reuse between them.

#### Output Artifacts

AFL-Sim creates a unique identifier for each simulation (e.g., `2026-08-08_12-31-42_936aed`). The following items are saved in `output_dir/2026-08-08_12-31-42_936aed`:

| File            | Description                                                                                                                        |
| :-------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`   | A copy of the effective configuration file, accurately displaying any parameter overrides passed via the CLI.                      |
| `run.log`       | The simulation's execution log containing all info, warnings, and errors (also displayed in the output console).                   |
| `metrics.jsonl` | The test loss function value and test set accuracy evaluated on the global model, logged by event counts and simulated time units. |
| `runtime.yaml`  | Essential simulation metadata, including the resolved locations of `data_dir` and `checkpoint_dir`.                                |

!!! danger

    If `runtime.yaml` is accidentally deleted, the simulation cannot be resumed. The `data_dir` and `checkpoint_dir` paths are essential for recovering the partition data, simulation clocks, and resumable checkpoints.

#### Checkpoint Artifacts

The following files are generated in `checkpoint_dir/2026-08-08_12-31-42_936aed` to enable resuming a simulation:

| Object                                  | Checkpoint Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Server States**                       | The files `latest_server_state.pt` (global model) and `latest_server_buffer.pt` (server buffer) are always saved regardless of configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Client States**                       | The file `latest_client_{client_id}_state.pt` is saved only if client memory augmentation is enabled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Simulation Metadata**                 | The file `latest_metadata.json` saves crucial metadata like the current event index and server update counts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Historical Models (Async Mode Only)** | To simulate asynchronous training in a serial manner, AFL-Sim records the global model version a client receives from the server upon returning an update and saves it to a historical database. Next time it's the client's turn to update the server, it retrieves this potentially stale version from the database to perform its local training. These historical model dictionaries are saved as `latest_history_version_{version_id}.pt`, and a lookup table tracking client-version requests is saved in `latest_model_requests.json`. These artifacts are not generated in synchronous mode. |
| **Best Model (Optional)**               | The files `model_best.pt` and `best_metadata.json` are saved upon user request to track the global model with the highest test accuracy.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

!!! info

    For more details on how AFL-Sim uses historical models and look-up tables of client requests to simulate asynchronous training in a serial manner, check the [Serializing Concurrent Execution](../implementation/serialization.md) section in the implementation notes of AFL-Sim.

## Resuming an Existing Simulation

Resuming simulations over the CLI is straightforward. Simply provide the path to the simulation's uniquely identified folder (e.g., `2026-08-08_12-31-42_936aed`) in the output directory you specified when you launched the simulation for the first time (or in the default `./outputs` if you did not explicitly specify an output directory):

=== "CLI"

    To resume a simulation over the CLI, use the `afl-sim resume` command:

    ``` bash
    afl-sim resume outputs/2026-08-08_12-31-42_936aed
    ```

    !!! tip

        Check the [`resume`](../reference/cli.md#afl-sim-resume) section in the [CLI Reference](../reference/cli.md) for the full list of arguments, including parameter overrides.

=== "Python"

    To resume a simulation in a script, import the `resume_simulation` function from the `afl_sim` library:

    ``` python
    from afl_sim import resume_simulation  # import resume_simulation

    # Both str and pathlib.Path inputs are accepted
    resume_simulation(output_path="outputs/2026-08-08_12-31-42_936aed")
    ```

    !!! tip

        Check the `resume_simulation` function documentation in the [API Reference](../reference/api.md) for the full list of arguments, including parameter overrides.

!!! warning

    To resume an existing simulation, AFL-Sim expects the relative path to the simulation _output_ directory, i.e., `output_dir`, not the checkpoint or data directories (`checkpoint_dir` and `data_dir`, respectively). The section [Launching a New Simulation](#launching-a-new-simulation) provides a detailed guide to AFL-Sim directory management.

!!! tip

    A copy of the simulation's effective YAML configuration file and the simulation execution log can be found in the simulation's unique output folder for easy inspection. Check the [Output Artifacts](#output-artifacts) section for more details.

Resuming a simulation does not create any new directories or artifacts. The logging artifacts `run.log` and `metrics.jsonl` are appended to, and resumable/best checkpoints are overwritten with their updated versions.

## A Note on Checkpoint Functionality

Checkpoint functionality is identical between fresh runs and existing simulation resumes:

- **Checkpoint Interval:** The interval (in seconds) between periodic resumable checkpoints is defined by the `interval_seconds` configuration parameter (see [Setting Up a YAML Configuration](configuration.md) for more details).

- **Shutdown Checkpoints:** AFL-Sim will always attempt to save a resumable shutdown checkpoint when the simulation/resume timeout is reached (specified by the `timeout_seconds` configuration parameter), or when the simulation is interrupted by the user (SIGINT) or the system (SIGTERM).
