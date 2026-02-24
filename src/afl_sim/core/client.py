from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from afl_sim.config import MemStrategyConfig, OptimizationConfig
from afl_sim.enums import MemoryType
from afl_sim.utils import compute_seed_from_dict, recursive_to_cpu

StateDict = dict[str, torch.Tensor]


class Client:
    """
    Builds a Federated Learning client.
    """

    def __init__(
        self,
        client_id: int,
        initial_model: nn.Module,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        weight: float,
        optim_config: OptimizationConfig,
        memory_strategy: MemStrategyConfig,
        base_seed: int,
    ):
        self.client_id = client_id
        self.data_loader = dataloader
        self.weight = weight

        # config
        self.base_lr = optim_config.learning_rate
        self.weight_decay = optim_config.weight_decay
        self.memory_type = memory_strategy.type
        self.base_seed = base_seed

        # Initialize Stale State on CPU
        self.stale_state: StateDict = {
            k: v.detach().clone().to("cpu")
            for k, v in initial_model.state_dict().items()
        }

        self.memory: StateDict = {}
        if self.memory_type.has_memory:
            self._init_memory(initial_model)

    def _init_memory(self, model: nn.Module) -> None:
        """Initialize memory tensors on CPU."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self.memory_type == MemoryType.MODELS:
                    self.memory[name] = param.detach().clone().to("cpu")
                elif self.memory_type == MemoryType.GRADS:
                    self.memory[name] = torch.zeros_like(param, device="cpu")

    def receive_global_model(self, global_model_state: StateDict) -> None:
        """
        Updates the client's local view (stale state) with the latest global model.
        """
        for k, v in global_model_state.items():
            self.stale_state[k] = v.detach().clone().to("cpu")

    def compute_update(
        self, shell_model: nn.Module, device: torch.device, event_idx: int
    ) -> StateDict:
        """
        Performs local training using the provided shared model shell.
        """
        seed_dict = {
            "base_seed": self.base_seed,
            "event_idx": event_idx,
            "client_id": self.client_id,
        }
        torch.manual_seed(compute_seed_from_dict(seed_dict))
        shell_model.load_state_dict(self.stale_state, strict=True)
        shell_model.train()

        self._train_local(shell_model, device)

        return self._derive_update(shell_model)

    def _train_local(self, model: nn.Module, device: torch.device) -> None:
        """Standard local training loop."""
        # Scale learning rate by client weight
        effective_lr = self.base_lr * self.weight

        optimizer = optim.SGD(
            model.parameters(), lr=effective_lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for inputs, labels in self.data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def _derive_update(self, trained_model: nn.Module) -> StateDict:
        """
        Calculates the update vector based on MemoryType.
        """
        delta: StateDict = {}

        for name, new_param in trained_model.named_parameters():
            if name not in self.stale_state:
                continue

            if self.memory_type == MemoryType.DISABLED:
                stale_tensor = self.stale_state[name].to(new_param.device)
                delta[name] = new_param - stale_tensor

            elif self.memory_type == MemoryType.MODELS:
                if name in self.memory:
                    mem_tensor = self.memory[name].to(new_param.device)
                    delta[name] = new_param - mem_tensor
                    self.memory[name] = new_param.detach().cpu().clone()

            elif self.memory_type == MemoryType.GRADS:
                stale_tensor = self.stale_state[name].to(new_param.device)
                current_grad = new_param - stale_tensor

                if name in self.memory:
                    mem_tensor = self.memory[name].to(new_param.device)
                    delta[name] = current_grad - mem_tensor
                    self.memory[name] = current_grad.detach().cpu().clone()

        return delta

    def get_stale_state_dict(self) -> StateDict | None:
        """Returns stale state or None."""
        return recursive_to_cpu(self.stale_state)

    def get_memory_dict(self) -> StateDict | None:
        """Returns memory or None."""
        if not self.memory:
            return None
        return recursive_to_cpu(self.memory)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restores client state from a dictionary (Resuming).
        """
        if "stale_state" in state_dict and state_dict["stale_state"] is not None:
            self.stale_state = state_dict["stale_state"]

        if "memory" in state_dict and state_dict["memory"] is not None:
            self.memory = state_dict["memory"]
