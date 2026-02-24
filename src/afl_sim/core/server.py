from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from afl_sim.utils import compute_seed_from_dict, recursive_to_cpu

StateDict = dict[str, torch.Tensor]


class Server:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        aggregation_goal: int,
        num_clients: int,
        reset_buffer: bool,
        base_seed: int,
    ):
        self.device = device
        self.test_loader = test_loader
        self.agg_goal = aggregation_goal
        self.num_clients = num_clients
        self.reset_buffer = reset_buffer
        self.current_acc = -1.0
        self.current_loss = -1.0
        self.best_acc = -1.0
        self.base_seed = base_seed
        self.just_updated = False

        self.model = model.to(self.device)

        # Shell Model
        self.shell_model = deepcopy(self.model)

        self.buffer: StateDict = {
            name: torch.zeros_like(param).to(self.device)
            for name, param in self.model.named_parameters()
        }

        self.current_count = 0

    def get_shell_model(self) -> nn.Module:
        """Returns the reusable model shell for clients."""
        return self.shell_model

    def get_current_loss(self) -> float:
        """Returns the current test loss."""
        return self.current_loss

    def get_current_accuracy(self) -> float:
        """Returns the current test accuracy."""
        return self.current_acc

    def aggregate_updates(self, client_update: StateDict) -> None:
        """Ingests a client update into the buffer."""
        with torch.no_grad():
            for name, delta in client_update.items():
                if name in self.buffer:
                    self.buffer[name].add_(delta.to(self.device))

        self.current_count += 1

    def global_update(self, event_idx: int) -> None:
        """
        Checks if buffer is full. If so, updates model, evaluates, and resets.
        """
        if self.current_count >= self.agg_goal:
            seed_dict = {"base_seed": self.base_seed, "event_idx": event_idx}
            torch.manual_seed(compute_seed_from_dict(seed_dict))
            self._apply_buffer_update(divisor=self.num_clients)
        else:
            if self.just_updated:
                self.just_updated = False

    def _apply_buffer_update(self, divisor: int) -> None:
        """Updates the Master Model using the aggregated buffer."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.buffer:
                    update_step = self.buffer[name] / divisor
                    param.add_(update_step)

            if self.reset_buffer:
                self._reset_buffer()
            self.current_count = 0
            self.just_updated = True

    def _reset_buffer(self) -> None:
        with torch.no_grad():
            for tensor in self.buffer.values():
                tensor.zero_()

    def evaluate(self) -> None:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        self.current_loss = avg_loss
        self.current_acc = accuracy
        if accuracy >= self.best_acc:
            self.best_acc = accuracy

    def get_state_dict(self) -> dict[str, Any]:
        """Returns a serializable dictionary of the server's state."""
        state = {
            "model_state": self.model.state_dict(),
            "buffer": self.buffer,
            "current_count": self.current_count,
            "best_acc": self.best_acc,
            "current_acc": self.current_acc,
        }
        return recursive_to_cpu(state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Resumes server state."""
        self.model.load_state_dict(state_dict["model_state"])

        loaded_buffer = state_dict["buffer"]
        with torch.no_grad():
            for name, tensor in loaded_buffer.items():
                if name in self.buffer:
                    self.buffer[name].copy_(tensor.to(self.device))

        self.current_count = state_dict["current_count"]
        self.best_acc = state_dict["best_acc"]
        self.current_acc = state_dict["current_acc"]

    def get_global_state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def just_performed_global_update(self) -> bool:
        return self.just_updated
