# Simulate the Asynchronous Federated Training of NN-based Classifiers

## Overview
This project provides a serial framework for simulating the asynchronous federated training of neural network (NN)-based classifiers on various standard datasets. Currently, the following datasets and models are supported:

- MNIST with a simple CNN
- CIFAR-10 with ResNet-18

## Asynchronous Federated Learning
The framework simulates the following implementation of asynchronous federated learning, best suited for cross-silo settings:
1. The server initializes the global model and broadcasts it to all participating clients.
2. Clients independently train the global model using their local data. Once a client completes its training, it sends its update to the server, requests, and receives the latest version of the global model.
3. Upon receiving the global model, clients repeat step 2.
4. The server periodically updates the global model after receiving a predefined number of local updates (buffered asynchronous aggregation, e.g., FedBuff ([https://arxiv.org/abs/2106.06639](https://arxiv.org/abs/2106.06639))).

## Supported Training Modes
The following federated training modes are supported:
- Asynchronous modes:
  - Clients asynchronously update the server with local pseudo-gradients on the global model.
  - Clients asynchronously update the server with updates corrected using the scheme described in [https://arxiv.org/abs/2405.10123](https://arxiv.org/abs/2405.10123) to balance heterogeneous client update frequencies.
- Synchronous modes:
  - FedAvg ([https://arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)): At each global update, the server uniformly samples a subset of clients and sends them the global model. Sampled clients synchronously update the server with their local pseudo-gradients on the global model.
 
## Client update model
The interval $t_i$ between consecutive updates from client $i \in \\{1,...,n\\}$ is modeled as an exponential random variable $t_i \sim \text{Exp}(\lambda_i)$. Given a user-specified standard deviation parameter $\sigma>0$, client rates are generated as samples from a log-normal distribution with mean $\mu=0$, i.e., $\lambda_i \sim \text{Log-normal}(0, \sigma^2)$.

## Usage

To run the `main.py` script, use the following command format:

```bash
python main.py --args <args>
```
### Arguments
- **`--num_clients`**: Specifies the number of clients participating in federated training. (Type: `int`, Default: `10`)
- **`--dataset`**: Indicates the dataset the be used for training. Options are `MNIST` or `CIFAR-10`. (Type: `str`, Default: `mnist`)
- **`--train_batch_size`**: Sets the batch size for local training at each client. (Type: `int`, Default: `64`)
- **`--test_batch_size`**: Defines the batch size for evaluating loss and accuracy on the test data. (Type: `int`, Default: `32`)
- **`--Delta`**: Determines the number of local updates required for a global aggregation, or the number of (uniformly) sampled clients for FedAvg. (Type: `int`, Default: `3`)
- **`--lr`**: Specifies the learning rate for local training. (Type: `float`, Default: `0.01`)
- **`--num_local_steps`**: Sets the number of local stochastic gradient descent (SGD) steps for training at each client. (Type: `int`, Default: `100`)
- **`--dirichlet_alpha`**: Controls the heterogeneity among client datasets using a Dirichlet distribution sample. Smaller values yield more heterogeneous datasets. (Type: `float`, Default: `1.0`)
- **`--mode`**: Chooses the communication mode for training. Options are `sync` for FedAvg or `async` for asynchronous training. (Type: `str`, Default: `async`)
- **`--correction`**: Enables the correction scheme described in [https://arxiv.org/abs/2405.10123](https://arxiv.org/abs/2405.10123) to balance heterogeneous client update rates. Set `True` to activate. (Type: `bool`, Default: default=`False`)
- **`--client_rate_std`**: Specifies the standard deviation used for generating client update rates. (Type: `float`, Default: `0.1`)
- **`--T_train`**: Sets the total training time in time units. (Type: `float`, Default: `5.0`)
