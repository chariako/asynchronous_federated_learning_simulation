from torch.utils.data import Subset
import numpy as np

def partition_data_by_dirichlet(full_dataset, num_clients, alpha):
    # Get the targets of the dataset
    targets = np.array(full_dataset.targets)
    num_classes = np.max(targets) + 1

    # Create a list to hold the indices for each client
    client_indices = [[] for _ in range(num_clients)]

    # For each class, partition the data among clients using a Dirichlet distribution
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) *
                       len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, proportions)

        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices)

    # Create datasets for each client
    client_datasets = [Subset(full_dataset, indices)
                       for indices in client_indices]

    return client_datasets
