import torch
from torch.utils.data import DataLoader
from copy import deepcopy

class Server:
    def __init__(self, model, num_clients, Delta, mode, device, test_dataset):
        self.global_model = deepcopy(model).to(device)
        state_dict = self.global_model.state_dict()
        self.aggregator = {key: torch.zeros_like(val) for key, val in state_dict.items()}
        self.num_clients = num_clients
        self.count = 0
        self.Delta = Delta
        self.mode = mode
        self.device = device
        self.test_dataset = test_dataset

    def aggregate_updates(self, client_update):
        self.count += 1
        # Aggregate updates from clients
        with torch.no_grad():
            # Aggregate updates from clients
            for key, agg_val in self.aggregator.items():
                if agg_val.dtype == torch.float32:
                    agg_val.add_(client_update[key], alpha=1.0 / self.num_clients)
                else:
                    val_type = agg_val.dtype
                    agg_val = agg_val.to(torch.float32) 
                    agg_val.add_(client_update[key].to(torch.float32), alpha=1.0 / self.num_clients)
                    agg_val = agg_val.to(val_type)

    def global_update(self, criterion, test_batch, time_stamp, config):
        # Update the global model
        if self.count == self.Delta or self.mode == 'sync':
            state_dict = self.global_model.state_dict()
            self.count = 0
            for key in self.aggregator.keys():
                state_dict[key] += self.aggregator[key]
                self.aggregator[key] = torch.zeros_like(self.aggregator[key])

            self.global_model.load_state_dict(deepcopy(state_dict))
            self.evaluate_metrics(criterion, test_batch, time_stamp, config)

    def evaluate_metrics(self, criterion, test_batch, time_stamp, config):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in DataLoader(self.test_dataset, batch_size=test_batch, shuffle=False, num_workers=0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss_k = total_loss / \
            len(DataLoader(self.test_dataset, batch_size=test_batch))
        acc_k = 100 * correct / total

        with open(config + '.txt', 'a') as f:
            f.write(f"{time_stamp:.4f}, {loss_k:.4f}, {acc_k:.2f}\n")
