import hashlib
import json
from collections.abc import Mapping
from typing import Any


def compute_hash_from_dict(config_dict: Mapping[str, Any]) -> str:
    """
    Generates a deterministic 16-character SHA-256 hash from a dictionary.

    Useful for creating unique, reproducible identifiers for configuration states.

    Args:
        config_dict (Mapping[str, Any]): The dictionary containing configuration parameters.

    Returns:
        str: A truncated 16-character hexadecimal hash string.
    """
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def compute_seed_from_dict(seed_dict: Mapping[str, Any]) -> int:
    """
    Generates a deterministic 32-bit integer seed from a dictionary.

    Transforms execution context variables into a stable seed for random number generators.

    Args:
        seed_dict (Mapping[str, Any]): The dictionary containing context variables.

    Returns:
        int: A 32-bit unsigned integer suitable for seeding pseudo-random number generators.
    """
    seed_str = json.dumps(seed_dict, sort_keys=True).encode("utf-8")
    hash_obj = hashlib.sha256(seed_str)
    return int(hash_obj.hexdigest(), 16) % (2**32)
