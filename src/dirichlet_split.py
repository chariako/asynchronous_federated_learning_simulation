from torch.utils.data import Subset
import numpy as np

def partition_data_by_dirichlet(full_dataset, num_clients, alpha, batch_size):
    # Get the targets of the dataset
    targets = np.array(full_dataset.targets)
    num_classes = len(np.unique(targets))
    min_indices = 0

    while min_indices < batch_size:

        # Create a list to hold the indices for each client
        client_indices = [[] for _ in range(num_clients)]

        # For each class, partition the data among clients using a Dirichlet distribution
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            proportions = (proportions * len(class_indices)).astype(int)
            ## Correct leftover/extra samples
            leftover = len(class_indices) - np.sum(proportions)
            if leftover != 0:
                if leftover < 0: # assigned indices exceed samples
                    client_index = np.where(proportions > leftover)[0]
                    proportions[client_index] -= leftover
                else: # leftover samples
                    proportions[-1] += leftover
            proportions = (np.cumsum(proportions)).astype(int)[:-1]
            split_indices = np.split(class_indices, proportions)

            for client_id, indices in enumerate(split_indices):
                client_indices[client_id].extend(indices)
        
        number_of_indices = [len(indices) for indices in client_indices]
        min_indices =  np.min(number_of_indices)

    # Create datasets for each client
    client_datasets = [Subset(full_dataset, indices)
                       for indices in client_indices]

    return client_datasets
