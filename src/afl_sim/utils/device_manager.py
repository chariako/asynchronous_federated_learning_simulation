import torch

from afl_sim.enums import DeviceType
from afl_sim.types import SimulationDevice


def get_device(device_type: DeviceType) -> SimulationDevice:
    """
    Resolves and initializes the requested PyTorch hardware device.

    Args:
        device_type (DeviceType): The requested hardware backend from the configuration.

    Returns:
        SimulationDevice: The initialized PyTorch device object (CPU, CUDA, or MPS).

    Raises:
        ValueError: If a specific hardware accelerator (CUDA or MPS) is explicitly
            requested but is not available on the host machine.
    """
    match device_type:
        case DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        case DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise ValueError("cuda requested but not available.")
            return torch.device("cuda")

        case DeviceType.MPS:
            if not torch.backends.mps.is_available():
                raise ValueError("mps requested but not available.")
            return torch.device("mps")

        case DeviceType.CPU:  # pragma: no branch
            return torch.device("cpu")
