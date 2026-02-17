import hashlib
import json
from typing import Any, overload

import torch


@overload
def recursive_to_cpu[K](data: dict[K, Any], detach: bool = True) -> dict[K, Any]: ...


@overload
def recursive_to_cpu[T](data: T, detach: bool = True) -> T: ...


def recursive_to_cpu(data: Any, detach: bool = True) -> Any:
    """
    Recursively moves tensors to CPU.
    """
    if isinstance(data, torch.Tensor):
        if detach:
            return data.detach().cpu()
        return data.cpu()

    elif isinstance(data, dict):
        return {k: recursive_to_cpu(v, detach) for k, v in data.items()}

    elif isinstance(data, list):
        return [recursive_to_cpu(v, detach) for v in data]

    elif isinstance(data, tuple):
        return tuple(recursive_to_cpu(v, detach) for v in data)

    else:
        return data


def compute_hash_from_dict(config_dict: dict[str, Any]) -> str:
    """
    Generates a deterministic hash given a dictionary.
    """
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
