from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from afl_sim.config import MemStrategyConfig, OptimizationConfig
from afl_sim.enums import MemoryType
from afl_sim.types import SimulationDevice, SimulationModel, TensorDict
from afl_sim.utils import compute_seed_from_dict


class Client:
    """
    Represents a participating node in the federated learning simulation.

    Manages local training execution, asynchronous memory states, and the computation
    of federated update vectors. Designed to minimize system memory allocations and
    safely transition weights between the CPU and GPU.

    Attributes:
        client_id (int): Unique identifier for the client.
        memory (TensorDict): Provides access to the client's internal memory state dictionary.
        _data_loader (DataLoader): Local data partition for training.
        _transform (Callable[..., Any] | None): The GPU-bound transformation pipeline applied during training.
        _weight (float): The client's scalar statistical weight.
        _base_lr (float): Base learning rate for local optimization.
        _weight_decay (float): Weight decay rate for local optimization.
        _memory_type (MemoryType): The configured historical memory tracking strategy.
        _base_seed (int): Base random seed used alongside global indexing for reproducibility.
        _memory (TensorDict): Internal dictionary mapping parameter names to CPU-bound memory tensors.
    """

    def __init__(
        self,
        client_id: int,
        initial_model: SimulationModel,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        weight: float,
        transform: Callable[..., Any] | None,
        optim_config: OptimizationConfig,
        memory_strategy: MemStrategyConfig,
        base_seed: int,
    ):
        """
        Initializes the federated client with its local data and hyperparameter configurations.

        Args:
            client_id (int): Unique identifier for the client.
            initial_model (SimulationModel): The PyTorch model used to initialize the client's memory shapes.
            dataloader (DataLoader[tuple[torch.Tensor, torch.Tensor]]): Local data partition for training.
            weight (float): The client's scalar weight (usually based on data partition size) applied to the learning rate.
            transform (Callable[..., Any] | None): The GPU-bound transformation pipeline applied during training.
            optim_config (OptimizationConfig): Configuration containing learning rate and weight decay.
            memory_strategy (MemStrategyConfig): Configuration defining how the client tracks historical state.
            base_seed (int): Base random seed used alongside global indexing for reproducible training sequences.
        """
        self.client_id = client_id

        self._data_loader = dataloader
        self._transform = transform
        self._weight = weight

        # config
        self._base_lr = optim_config.learning_rate
        self._weight_decay = optim_config.weight_decay
        self._memory_type = memory_strategy.type
        self._base_seed = base_seed

        self._memory: TensorDict = {}
        self._init_memory(initial_model)

    @property
    def memory(self) -> TensorDict:
        """
        Provides access to the client's internal memory state dictionary.

        Returns:
            TensorDict: A dictionary mapping parameter names to CPU-bound memory tensors.
        """
        return self._memory

    def _init_memory(self, model: SimulationModel) -> None:
        """
        Initializes the client's memory tensors directly into system RAM (CPU).

        Allocates required memory based on the configured strategy (MODELS or GRADS)
        without building a computation graph.

        Args:
            model (SimulationModel): The model used as a structural template for memory allocation.
        """
        if not self._memory_type.has_memory:
            return

        with torch.no_grad():
            if self._memory_type == MemoryType.MODELS:
                self._memory = {
                    name: param.detach().to(device="cpu", copy=True)
                    for name, param in model.named_parameters()
                }
            else:  # MemoryType.GRADS
                self._memory = {
                    name: torch.zeros_like(param, device="cpu", requires_grad=False)
                    for name, param in model.named_parameters()
                }

    def compute_update(
        self,
        model_shell: SimulationModel,
        device: SimulationDevice,
        global_idx: int,
        requested_state_dict: TensorDict,
    ) -> TensorDict:
        """
        Executes the full local training pipeline and computes the federated delta.

        Sets the deterministic seed, loads the requested global state into the GPU shell,
        executes local training, and derives the final update vector based on the memory strategy.

        Args:
            model_shell (SimulationModel): The shared GPU-bound model shell used for training.
            device (SimulationDevice): The target hardware device (e.g., GPU).
            global_idx (int): The current global event index, used for seed generation.
            requested_state_dict (TensorDict): The specific global model version requested by the client.

        Returns:
            TensorDict: The final, CPU-bound update vector (delta) to be sent to the server.
        """
        seed_dict = {
            "base_seed": self._base_seed,
            "global_idx": global_idx,
            "client_id": self.client_id,
        }
        torch.manual_seed(compute_seed_from_dict(seed_dict))

        model_shell.load_state_dict(requested_state_dict, strict=True)
        self._train_local(model_shell, device)

        return self._derive_update(model_shell, requested_state_dict)

    def _train_local(self, model: SimulationModel, device: SimulationDevice) -> None:
        """
        Executes the standard PyTorch local training loop.

        Performs SGD optimization on the client's local dataset. Utilizes asynchronous
        PCIe data transfers if running on compatible hardware to prevent CPU blocking.

        Args:
            model (SimulationModel): The GPU-bound model shell loaded with the requested global weights.
            device (SimulationDevice): The target computational device.
        """
        model.train()
        is_cuda = device.type == "cuda"

        # Scale learning rate by client weight
        effective_lr = self._base_lr * self._weight

        optimizer = optim.SGD(
            params=model.parameters(), lr=effective_lr, weight_decay=self._weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for inputs, labels in self._data_loader:
            inputs, labels = (
                inputs.to(device, non_blocking=is_cuda),
                labels.to(device, non_blocking=is_cuda),
            )

            if self._transform is not None:
                inputs = self._transform(inputs)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def _derive_update(
        self, trained_model: SimulationModel, initial_model_dict: TensorDict
    ) -> TensorDict:
        """
        Computes the federated update vector (delta) based on the active memory strategy.

        Transfers the trained parameters to the CPU and applies in-place mathematical
        mutations using existing memory blocks to prevent massive temporary allocations
        and garbage collection overhead.

        Args:
            trained_model (SimulationModel): The model after local training has completed.
            initial_model_dict (TensorDict): The original global model weights before local training.

        Returns:
            TensorDict: The derived update vector (delta) stored entirely in CPU RAM.
        """
        trained_model.zero_grad(set_to_none=True)
        delta: TensorDict = {}

        with torch.no_grad():
            for name, new_param in trained_model.named_parameters():
                new_param_cpu = new_param.detach().to(device="cpu", copy=True)

                if self._memory_type == MemoryType.DISABLED:
                    delta[name] = new_param_cpu.sub_(initial_model_dict[name])

                elif self._memory_type == MemoryType.MODELS:
                    if name in self._memory:
                        delta[name] = self._memory[name].sub_(new_param_cpu).neg_()
                        self._memory[name] = new_param_cpu

                elif self._memory_type == MemoryType.GRADS:  # pragma: no branch
                    new_param_cpu.sub_(initial_model_dict[name])
                    if name in self._memory:
                        delta[name] = self._memory[name].sub_(new_param_cpu).neg_()
                        self._memory[name] = new_param_cpu

        return delta

    def load_mem_state_dict(self, mem_dict: TensorDict) -> None:
        """
        Restores the client's memory state from a serialized dictionary.

        Safely copies the incoming tensors directly into the client's pre-allocated
        memory blocks to prevent structural corruption from external references.

        Args:
            mem_dict (TensorDict): The state dictionary containing the historical memory weights.
        """
        if not self._memory_type.has_memory:
            return

        with torch.no_grad():
            for name, tensor in mem_dict.items():
                if name in self._memory:
                    self._memory[name].copy_(tensor)
