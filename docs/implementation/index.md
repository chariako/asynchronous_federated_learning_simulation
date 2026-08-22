# Implementation Notes

This section covers various aspects of AFL-Sim's implementation, for interested readers and developers who are considering extending the framework.

## Table of Contents

**[Simulated Federated Learning Workflows](workflows.md):** A detailed description of the real-world Federated Learning implementations simulated by AFL-Sim, including examples and diagrams.

**[Local Training and Aggregation Weights](local_training.md):** How AFL-Sim internally handles these two crucial components of Federated Learning, while maintaining theoretical consistency.

**[Serializing Concurrent Execution](serialization.md):** How AFL-Sim's architecture converts concurrent client training to serial simulation events, for both synchronous and asynchronous training modes.

**[Modeling System Heterogeneity](modeling.md)** A short guide on how AFL-Sim synthesizes realistic inputs for its benchmarking tasks, including data splits and client arrivals.
