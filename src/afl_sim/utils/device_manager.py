import torch

from afl_sim.enums import DeviceType


def get_device(device_type: DeviceType) -> torch.device:
    if device_type == DeviceType.AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    elif device_type == DeviceType.CUDA:
        if not torch.cuda.is_available():
            raise ValueError("cuda requested but not available.")
        return torch.device("cuda")

    elif device_type == DeviceType.MPS:
        if not torch.backends.mps.is_available():
            raise ValueError("mps requested but not available.")
        return torch.device("mps")

    elif device_type == DeviceType.CPU:
        return torch.device("cpu")
