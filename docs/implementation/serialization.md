# Serializing Concurrent Execution

To enable simulating large client cohorts on a single hardware accelerator (or the CPU), AFL-Sim relies on an internal mechanism that converts concurrent client training sessions to serial events that can be processed sequentially without overwhelming the computing resource.

At most one client is allowed to use the accelerator for its local training at a time; once the training is complete, the client removes its states from the accelerator, which becomes available for the next client in the queue.

## Overview

In the course of a simulation, AFL-Sim repeatedly performs a sequence of processing blocks until a user-specified timeout[^1] is reached. This sequence is almost identical between the synchronous and asynchronous training modes of AFL-Sim, with some additional mechanics in the asynchronous mode to handle the system's dynamic nature.

## Asynchronous State Manager (ASM)

In asynchronous mode, clients may concurrently train different (and most likely, outdated) versions of the global model. To enable serial execution where a single client trains at a time, each client needs to "remember" which global model version they are supposed to train on, and store the corresponding weights while waiting in the queue for the accelerator until their turn arrives.

Excluding optional buffers when memory augmentation is enabled for algorithmic purposes, AFL-Sim clients are stateless and do not store model weights. To keep track of the global model versions that will be processed by the clients, AFL-Sim employs a special structure called **Asynchronous State Manager (ASM)**. ASM maintains the following internal objects:

- A lookup table, which keeps a tally of the global model version ID each client is supposed to train on when its turn to use the accelerator arrives.
- A historical database of stale global model versions, storing the weights of the version IDs listed in the lookup table.

Using a centralized historical database reduces memory requirements compared to endowing clients with model states. Assuming there are `N` clients in total, at most `N` models will need to be saved, whereas client-side storage requires saving strictly `N` models.

AFL-Sim automatically purges global models from the database once they are no longer needed by clients, and does not allow duplicate entries.

!!! tip

    For a refresher on client memory augmentation, check the respective [section](../user_guide/configuration.md#client-memory-augmentation) in [Setting Up a YAML Configuration](../user_guide/configuration.md).

## AFL-Sim Step Breakdown

An AFL-Sim simulation step has two components: a client processing block, followed by a server processing block. While the fundamental structure of these blocks remains consistent across synchronous and asynchronous training modes, some additional operations are introduced in asynchronous mode to allow for Asynchronous State Manager (ASM) updates.

!!! info "Notation"

    In all diagrams below, the term "global model" will be denoted by **GB**.

### Base Client Processing Block

A base client processing block is depicted below:

1. The client being processed receives the global model, i.e., the current version for synchronous mode, or a historical version for asynchronous mode.
2. The client trains the global model on its local data, and updates its local states if applicable (e.g., for memory-augmented methods).
3. The client pushes its update to the server's buffer.
4. In asynchronous mode, the client would at this point receive from the server the current version of the global model to begin a new training session. Instead, AFL-Sim updates the client's target global model version ID in ASM's lookup table to the current global model ID. At the client's next turn, AFL-Sim will fetch this version to the client for training.

``` mermaid
---
title: Base Client Processing Block
---

stateDiagram-v2
    direction LR
    state if_state <<choice>>
    [*] --> Fetch
    Fetch: Receive GB
    Fetch --> TrainUpdate
    TrainUpdate: Train on Local Data<br/>Update Local States
    TrainUpdate --> UpdateServer
    UpdateServer: Push Update<br/>to Server
    UpdateServer --> if_state: Async Mode?
    if_state --> [*]: No
    if_state --> Bookkeeping: Yes
    Bookkeeping: Sync Target<br/>Version ID
    Bookkeeping --> [*]
```

!!! tip

    The real-world Federated Learning workflows AFL-Sim simulates in its synchronous and asynchronous training modes are detailed in the [Simulated Federated Learning Workflows](workflows.md) section.

### Base Server Processing Block

The diagram below depicts a base server processing block, which always directly follows a client processing block in an AFL-Sim simulation.

1. A client update arrives and is stored in the server's buffer.
2. If the aggregation goal has been reached, i.e., all sampled clients have returned their updates in synchronous mode, or the buffer is full in asynchronous mode, the server aggregates the buffer's contents to update the global model. The server is also responsible for calculating the new model's performance metrics by evaluating it on the test set.
3. In asynchronous mode, the simulation will append the newly generated global model to ASM's historical database under the corresponding version ID.

``` mermaid
---
title: Base Server Processing Block
---

stateDiagram-v2
    direction LR
    state if_buffer <<choice>>
    state if_async <<choice>>
    [*] --> Receive
    Receive: Receive Client<br/>Update
    Receive --> if_buffer: Aggregation<br/>Goal Reached?
    if_buffer --> [*]: No
    if_buffer --> GlobalUpdate: Yes
    GlobalUpdate: Aggregate Updates<br/>Evaluate New GB
    GlobalUpdate --> if_async: Async Mode?
    if_async --> [*]: No
    if_async --> Bookkeeping: Yes
    Bookkeeping: Add New GB<br/>to History
    Bookkeeping --> [*]
```

!!! tip

    The model's performance metrics are saved in a JSONL file inside the simulation's unique output directory. Check the [Execution Guide](../user_guide/execution.md) for a detailed description of the artifacts AFL-Sim generates and their storage.

## Workflow Diagram

The following diagram summarizes AFL-Sim's entire workflow for the computation of a global update, including hardware management. Both synchronous and asynchronous training modes are depicted in the diagram, making their overlap and divergence easier to track.

### Hardware Management

The server, clients, and the Asynchronous State Manager (ASM) in asynchronous mode are based on the CPU and conduct the bulk of their operations there to avoid exhausting the accelerator.

Accelerator usage is allowed only in two instances:

- A client trains the global model version it received on its local dataset.
- The server evaluates a newly generated global model to calculate its performance metrics.

To perform these operations, the clients and server pull a dedicated, centralized model shell maintained by the simulation, which resides on the accelerator[^2] and load their local model dictionaries. Once the training or evaluation is completed, all local states are removed from the accelerator to ensure ample memory for the next user.

!!! note

    The client processing phase in the diagram below is a loop over all clients participating in the global update, i.e., the sampled clients in synchronous mode, and the clients corresponding to the buffered updates in asynchronous mode. The loop has been abstracted away to a single client `I`.

``` mermaid
sequenceDiagram
    autonumber
    participant Man as ASM
    participant Sim as Simulation<br/>(Orchestrator)
    participant S as Server
    participant C as Client I

    rect
    Note over Man, C: PHASE 1: Client Processing
    loop

    %% Fetch
    alt Synchronous Mode
        Sim->>S: Request Current GB Weights
        S-->>Sim: Return Current GB Weights
    else Asynchronous Mode
        Sim->>Man: Request Target GB for Client I
        Note over Man: Query<br/>Lookup & History
        Man-->>Sim: Return Target GB Weights
    end

    Sim->>C: Send Target GB Weights

    %% Client Update
    C->>Sim: Request Shared Model Shell
    Sim-->>C: Return Shared Model Shell

    Note over C: Load Weights to Shell<br/>Train on Local Data<br/>(Accelerator)
    Note over C: Update Local States<br/>(CPU)

    C->>S: Push Model Update to Buffer (CPU)

    %% Bookkeeping
    opt Asynchronous Mode Only (Lookup Update)
        Sim->>S: Request Current Version ID
        S-->>Sim: Return Current Version ID
        Sim->>Man: Update Lookup Table<br/>(Client I -> Current ID)
    end
    end
    end

    rect
    Note over Man, C: PHASE 2: Server Processing

    Note over S: Aggregation Goal Reached<br/>Aggregate Updates into New GB Weights<br/>Reset Buffer (CPU)

    S->>Sim: Request Shared Model Shell
    Sim-->>S: Return Shared Model Shell
    Note over S: Load New GB to Shell<br/>Evaluate on Test Set (Accelerator)


    opt Asynchronous Mode Only (History Update)
        Sim->>S: Request New Version ID & Weights
        S-->>Sim: Return New Version ID & Weights
        Sim->>Man: Archive ID & Weights<br/>to GB History
    end
    end
```

[^1]: In wall-clock seconds.
[^2]: Or the CPU in CPU-only mode.
