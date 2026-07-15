import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from afl_sim.types import ServerState, SimulationDevice, SimulationModel, TensorDict
from afl_sim.utils import compute_seed_from_dict


class Server:
    """
    Orchestrates the global federated learning model and update aggregations.

    Manages the central model state, client update buffering, and global evaluation
    metrics while ensuring state tensors remain on the CPU to prevent VRAM exhaustion.
    """

    def __init__(
        self,
        model: SimulationModel,
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        aggregation_goal: int,
        num_clients: int,
        reset_buffer: bool,
        base_seed: int,
    ):
        """
        Initializes the server with the starting model and evaluation parameters.

        Args:
            model (SimulationModel): The PyTorch model used to initialize the global state.
            test_loader (DataLoader[tuple[torch.Tensor, torch.Tensor]]): DataLoader for evaluation.
            aggregation_goal (int): Number of client updates required to trigger a global update.
            num_clients (int): Total number of participating clients (used as the update divisor).
            reset_buffer (bool): Whether to zero out the aggregation buffer after a global update.
            base_seed (int): Base random seed used to ensure reproducible evaluations.
        """

        self._test_loader = test_loader
        self._agg_goal = aggregation_goal
        self._num_clients = num_clients
        self._reset_buffer_required = reset_buffer
        self._base_seed = base_seed

        self._best_acc = -1.0
        self._current_count = 0

        # Public attributes
        self.current_acc = -1.0
        self.current_loss = -1.0
        self.current_version = 0

        # State dicts
        self._global_model_dict: TensorDict = {
            name: param.detach().to(device="cpu", copy=True)
            for name, param in model.named_parameters()
        }

        self._buffer: TensorDict = {
            name: torch.zeros_like(param, device="cpu", requires_grad=False)
            for name, param in model.named_parameters()
        }

    @property
    def global_model_dict(self) -> TensorDict:
        """
        Provides read-only access to the global model's state dictionary.

        Returns:
            TensorDict: A dictionary mapping parameter names to CPU-bound tensors.
        """

        return self._global_model_dict

    @property
    def state(self) -> ServerState:
        """
        Packages the server's current state into a dataclass for serialization.

        Returns:
            ServerState: A dataclass instance containing the model state, buffer, and metrics.
        """
        return ServerState(
            model_state=self._global_model_dict,
            buffer=self._buffer,
            current_count=self._current_count,
            best_acc=self._best_acc,
            current_acc=self.current_acc,
            current_version=self.current_version,
        )

    def aggregate_updates(self, client_update: TensorDict) -> None:
        """
        Ingests a single client update into the server's aggregation buffer.

        Safely adds the client's parameter updates to the CPU buffer without
        building a computation graph.

        Args:
            client_update (TensorDict): Dictionary mapping parameter names to update tensors.
        """

        with torch.no_grad():
            for name, param in client_update.items():
                if name in self._buffer:
                    self._buffer[name].add_(param)

        self._current_count += 1

    def global_update(
        self,
        model_shell: SimulationModel,
        device: SimulationDevice,
        global_idx: int,
        sim_time: float,
    ) -> bool:
        """
        Triggers a global model update and evaluation if the aggregation goal is met.

        Args:
            model_shell (SimulationModel): The shared GPU-bound model shell used for evaluation.
            device (SimulationDevice): The hardware device (e.g., GPU) to execute the evaluation on.
            global_idx (int): The current global event index.
            sim_time (float): The current simulated time in seconds.

        Returns:
            bool: True if a global update and evaluation were performed, False otherwise.
        """
        if self._current_count >= self._agg_goal:
            seed_dict = {"base_seed": self._base_seed, "global_idx": global_idx}
            torch.manual_seed(compute_seed_from_dict(seed_dict))
            self._apply_buffer_update(divisor=self._num_clients)
            self.current_version += 1
            self._evaluate(model_shell, device, global_idx, sim_time)

            return True
        return False

    def _apply_buffer_update(self, divisor: int) -> None:
        """
        Updates the global master model using the aggregated buffer.

        Divides the accumulated buffer by the specified divisor and adds it to the global weights.
        Automatically resets the buffer if required by the configuration.

        Args:
            divisor (int): The scalar value to divide the accumulated buffer by.
        """
        with torch.no_grad():
            for name, param in self._global_model_dict.items():
                param.add_(self._buffer[name], alpha=1.0 / divisor)

            if self._reset_buffer_required:
                self._reset_buffer()
            self._current_count = 0

    def _reset_buffer(self) -> None:
        """Zeros out the aggregation buffer without tracking gradients."""
        with torch.no_grad():
            for tensor in self._buffer.values():
                tensor.zero_()

    def _evaluate(
        self,
        model_shell: SimulationModel,
        device: SimulationDevice,
        global_idx: int,
        sim_time: float,
    ) -> None:
        """
        Evaluates the updated global model against the test dataset.

        Temporarily loads the global CPU state into the GPU-bound model shell to compute
        loss and accuracy, logs the results, and updates tracking metrics.

        Args:
            model_shell (SimulationModel): The shared GPU-bound model shell.
            device (SimulationDevice): The target device for evaluation math.
            global_idx (int): The current global event index.
            sim_time (float): The current simulated time in seconds.
        """
        model_shell.zero_grad(set_to_none=True)
        model_shell.load_state_dict(self._global_model_dict)
        model_shell.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        is_cuda = device.type == "cuda"

        with torch.no_grad():
            for inputs, labels in self._test_loader:
                inputs, labels = (
                    inputs.to(device, non_blocking=is_cuda),
                    labels.to(device, non_blocking=is_cuda),
                )
                outputs = model_shell(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0

        # Update metric states
        self.current_loss = avg_loss
        self.current_acc = accuracy

        # Update best accuracy
        if accuracy >= self._best_acc:
            self._best_acc = accuracy

        # Update logger
        logger.info(
            f"Global Update | Event: {global_idx:6d} | Time: {sim_time:5.2f} | "
            f"Loss: {avg_loss:2.4f} | Acc: {accuracy:3.2f}%"
        )

    def load_state_dict(self, state_dict: ServerState) -> None:
        """
        Restores the server's internal state from a serialized state object.

        Safely copies the serialized buffer and global model weights back into the
        server's CPU memory without altering computation graph properties.

        Args:
            state_dict (ServerState): The dataclass state object to restore from.
        """

        loaded_buffer = state_dict.buffer
        with torch.no_grad():
            for name, tensor in loaded_buffer.items():
                if name in self._buffer:
                    self._buffer[name].copy_(tensor)

        loaded_model_dict = state_dict.model_state
        with torch.no_grad():
            for name, tensor in loaded_model_dict.items():
                if name in self._global_model_dict:
                    self._global_model_dict[name].copy_(tensor)

        self._current_count = state_dict.current_count
        self._best_acc = state_dict.best_acc
        self.current_acc = state_dict.current_acc
        self.current_version = state_dict.current_version
