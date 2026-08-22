# Simulated Federated Learning Workflows

A standard Federated Learning (FL) loop typically comprises the following steps:

- **Local Training**: A subset of clients receive the current version of the global model from the server, and use it as the starting point for local training sessions on their respective, private datasets. When the training is complete, clients construct their updates (e.g., the difference between the trained model and the initial global model, sometimes referred to as a "model delta") and share them with the server.
- **Global Model Updates**: The server receives the updates from the clients and aggregates them to update the global model (e.g., by taking a gradient step from current global model using the combined local model deltas in the place of a global gradient).

``` mermaid
graph LR
  A["Global Model Updates (Server)"] -->|"Global Model"| B["Local Training (Clients)"];
  B -->|"Local Updates"| A;
```

FL workflows can be _server-driven_, where the server initiates FL loops by sending the global model to select clients, or _client-driven_, where clients independently query the server for the current global model to train on at their own discretion. This section specifies the structure of the FL workflows AFL-Sim simulates in its two supported communication modes.

## Synchronous Mode

The synchronous mode of AFL-Sim is _server-driven_. The following steps are repeated until some stopping criterion is satisfied (e.g., the maximum allowable number of global updates has been reached)[^1]:

1. At the beginning of each round, the server **uniformly** samples a subset of clients and sends them the global model. The sample size is specified in the simulation configuration.
2. The clients receive the global model, train it on their local datasets, and send the server their local updates.
3. The server calculates a weighted average of the received updates, and adds the result to the global model to produce the new global model version.

A toy example with three clients and sample size equal to two is depicted below:

``` mermaid
sequenceDiagram
  autonumber

  participant S as Server
  participant C1 as Client 1
  participant C2 as Client 2
  participant C3 as Client 3

  Note over S: Initialize Global Model v0
  Note over S: Sample Clients 1 & 2
  S->>C1: Send Global Model v0
  S->>C2: Send Global Model v0
  Note over C1, C2: Local Training on v0
  C1-->>S: Return Update u01
  C2-->>S: Return Update u02
  Note over S: Aggregate u01, u02<br/>Generate Global Model v1
  Note over S: Sample Clients 2 & 3
  S->>C2: Send Global Model v1
  S->>C3: Send Global Model v1
```

!!! note

    AFL-Sim hardcodes uniform client sampling, but non-uniform sampling may often work better in practice, e.g., [importance sampling](https://arxiv.org/abs/2012.07383). Custom sampling schemes can be implemented in AFL-Sim by modifying the generation of simulated clocks (see the [Client Latency Distributions](modeling.md#client-latency-distributions) section in the [Modeling](modeling.md) implementation notes for more details on these constructions).

## Asynchronous Mode

The asynchronous mode of AFL-Sim is _client-driven_. The server broadcasts the global model to all participating clients during the initialization phase. The following steps are then repeated until some stopping criterion is satisfied (e.g., the maximum allowable number of global updates has been reached)[^1]:

1. Each client trains the version of the global model it received on its local data independently and at its own pace. Then, it returns its update to the server.
2. The server accepts client updates as they arrive, and stores them into a buffer (i.e., [buffered aggregation](https://arxiv.org/abs/2106.06639)).
3. The server immediately shares its current version of the global model with the client who just returned an update, so that the client can begin a new local training session. Only then does it proceed to the next step.
4. If the server has received a number of client updates equal to its buffer size, it uses the buffer contents (e.g., their weighted average) to update the global model and optionally resets the buffer.

It is possible to implement a server-driven asynchronous communication protocol, where the server periodically samples clients, sends them the global model, and then again accepts updates as they arrive. The client-driven protocol described above was chosen instead to achieve the following goals:

- Limit server orchestration, making the system simpler to implement and more scalable as the number of clients grows ("cross-device" setting).
- Increase robustness to client unavailability and unresponsiveness.
- Avoid keeping clients idle when they can be computing.
- Decouple communication from computation, enabling a non-blocking architecture where the server instantly returns the current model to the client while computing the global update in parallel.

A toy example with three clients and buffer size equal to two is shown below:

``` mermaid
sequenceDiagram
  autonumber

  participant S as Server
  participant C1 as Client 1
  participant C2 as Client 2
  participant C3 as Client 3

  Note over S: Initialize Global Model v0
  S->>C1: Broadcast Global Model v0
  S->>C2: Broadcast Global Model v0
  S->>C3: Broadcast Global Model v0
  Note over C3: Local Training on v0
  C3-->>S: Return Update u03
  S->>C3: Send Global Model v0
  Note over S: Buffer: 1/2
  Note over C2: Local Training on v0
  C2-->>S: Return Update u02
  S->>C2: Send Global Model v0
  Note over S: Buffer: 2/2 (Full)<br/>Aggregate u03, u02<br/>Generate Global Model v1
  Note over C1: Local Training on v0
  C1-->>S: Return Update u01
  S->>C1: Send Global Model v1
  Note over S: Buffer: 1/2
  Note over C1: Local Training on v1
  C1-->>S: Return Update u11
  S->>C1: Send Global Model v1
  Note over S: Buffer: 2/2 (Full)<br/>Aggregate u01, u11<br/>Generate Global Model v2
```

!!! note

    - As shown in the diagram above, over the course of learning clients may receive model versions that they have seen before. This is true in particular before the first global update, or if clients return updates at a rate faster than the global update rate. AFL-Sim allows this additional communication to preserve client latency statistics and avoid breaking the simulation flow, but such redundancy can easily be resolved in practical systems (e.g., by keeping a lookup table of the latest version each client has received and avoid sending the same version more than once).
    - AFL-Sim **will not discard** multiple updates by the same client in the same buffer window (e.g., Client 1 in the diagram above). This may create bias in favor of fast clients, but can also potentially [speed up learning](https://arxiv.org/abs/2206.07638). Memory augmentation can [offset this bias](https://arxiv.org/abs/2405.10123) by implicitly balancing heterogeneous response rates.
    - AFL-Sim does not currently support staleness-dependent discounting of straggler updates. As explained in the previous bullet, AFL-Sim's architecture already penalizes slow clients by having the server indiscriminately accept updates as they arrive. Further penalization of stragglers may result in losing important information about their local distributions when local datasets are highly non-IID.

!!! warning

    In both synchronous and asynchronous modes, the aggregation weights for local updates are hardcoded for all clients to `1 / (total participating clients)` (e.g., `1/3` in the toy examples above). It is possible to implement custom weights by modifying the aggregation functionality of the [Server](https://github.com/chariako/asynchronous_federated_learning_simulation/blob/main/src/afl_sim/server/server.py) module. It is important to note, however, that AFL-Sim natively scales the local stochastic gradients at each client according to relative dataset size (see the [Local Training and Aggregation Weights](local_training.md) documentation page).

[^1]: AFL-Sim simulations will halt once the timeout configuration parameter (in wall-clock seconds) has been reached. This design is intentional to accommodate time-constrained runs, e.g., on HPC. The logical stopping criterion may or may not have been satisfied until that point; if not, the simulation may be resumed at a later time using AFL-Sim's resume functionality.
