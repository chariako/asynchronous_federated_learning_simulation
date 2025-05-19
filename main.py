import argparse
import torch
import torch.nn as nn
import numpy as np
from src import *
from pathlib import Path
import torchvision.models as models

parser = argparse.ArgumentParser(
    description='Train a NN-based classifier using asynchronous FL or FedAvg')
parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
parser.add_argument('--train_batch_size', type=int, default=64, help='Client training batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='Testing batch size')
parser.add_argument('--Delta', type=int, default=3,
                    help='Number of clients for aggregation')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--num_local_steps', type=int, default=100, help='Local SGD steps')
parser.add_argument('--dirichlet_alpha', type=float, default=1.0,
                    help='Dirichlet data split parameter')
parser.add_argument('--T_train', type=float, default=5.0, help='Total experiment time')
parser.add_argument('--mode', type=str, default='async', help='Aggregation mode (sync/async)')
parser.add_argument('--client_rate_std', type=float, default=0.1,
                    help='STD for client Poisson rate (mean=1, lognormal distribution)')
parser.add_argument('--correction', type=str, default='False', help='AREA correction')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')

args = parser.parse_args()
num_clients = args.num_clients
Delta = args.Delta
alpha = args.lr
num_local_steps = args.num_local_steps
dirichlet_alpha = args.dirichlet_alpha
T_train = args.T_train
mode = args.mode
sigma_rate = args.client_rate_std
correction_str = args.correction
correction = False
if correction_str.lower() in ('yes', 'true', 't', 'y', '1'):
    correction = True
batch_size = args.train_batch_size
test_batch = args.test_batch_size
dataset_name = args.dataset 

Path('./results').mkdir(parents=True, exist_ok=True)
    
config = './results/' + dataset_name.lower() + '_num_clients_' + str(num_clients) + '_Delta_' + str(Delta) + '_alpha_' + str(alpha) + '_num_local_steps_' \
    + str(num_local_steps) + '_dirichlet_alpha_' + str(dirichlet_alpha) + '_T_train_' + str(T_train) \
    + '_mode_' + mode + '_sigma_rate_' + str(sigma_rate) + '_correction_' + str(correction) \
    + '_batch_size_' + str(batch_size)


if __name__ == '__main__':
    # Load dataset
    full_train_dataset, test_dataset = load_dataset(dataset_name)

    # Partition the dataset among clients using Dirichlet distribution
    client_datasets = partition_data_by_dirichlet(
        full_train_dataset, num_clients, dirichlet_alpha)
    client_weights = np.array(
        [len(i) for i in client_datasets]) / len(full_train_dataset)

    # Specify batch sizes for each client
    client_batch_sizes = [batch_size] * num_clients

    # Initialize model, loss function, and optimizer
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN().to(device)
    elif dataset_name.lower() in ('cifar','cifar10','cifar-10'):
        model = models.resnet18(weights=None).to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize clients and server
    clients = [Client(i, model, client_datasets[i], device,
                      client_batch_sizes[i], client_weights[i], correction) for i in range(num_clients)]
    server = Server(model, num_clients, Delta, mode, device, test_dataset)

    # Generate asynchronous clock
    rates = np.random.lognormal(0, sigma_rate, num_clients)
    global_clock = generate_global_clock(num_clients, rates, T_train, Delta, mode)
    Iters = len(global_clock)

    with open(config + '.txt', 'w') as f:
        f.write("Timestamp, Loss, Accuracy\n")

    # Main loop
    for k in range(Iters):
        time_stamp, i_k = global_clock[k]  # active client

        # Client performs local update with multiple steps and returns the difference
        for i in i_k:
            client_update = clients[i].client_update(alpha, criterion, num_local_steps)
            
            # Server aggregates updates
            server.aggregate_updates(client_update)

        if mode == 'async':
            # Client receives the latest global model
            for i in i_k:
                clients[i].receive_global_model(
                    server.global_model.state_dict())

        server.global_update(criterion, test_batch, time_stamp, config)

        if mode == 'sync':

            if k < Iters-1:
                _, i_k_next = global_clock[k+1]

                for i in i_k_next:
                    clients[i].receive_global_model(
                        server.global_model.state_dict())
