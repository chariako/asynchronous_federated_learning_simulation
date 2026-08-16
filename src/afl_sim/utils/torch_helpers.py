import torch

from afl_sim.types import TensorDict


def safe_tensor_dict_overwrite(
    orig_dict: TensorDict, new_dict: TensorDict, context: str
) -> None:
    """
    Safely overwrites tensors in an original dictionary with tensors from a new dictionary in-place.

    Enforces strict structural and shape matching between the two dictionaries. It safely
    copies the incoming tensors directly into the pre-allocated memory blocks of the
    original dictionary. This prevents structural corruption, preserves external tensor
    references, and avoids unnecessary memory allocation overhead.

    Args:
        orig_dict (TensorDict): The original state dictionary whose tensors will be overwritten.
        new_dict (TensorDict): The new state dictionary containing the incoming tensor values.
        context (str): A description of the object or context (e.g., 'Client 5') used to
            provide clear context in error messages.

    Raises:
        RuntimeError: If a required parameter key from `orig_dict` is missing in `new_dict`,
            or if a tensor's shape does not exactly match the pre-allocated memory.
    """
    with torch.no_grad():
        for name in orig_dict:
            if name not in new_dict:
                raise RuntimeError(
                    f"State dict corruption detected in {context}. "
                    f"Missing expected memory key: '{name}'"
                )

            if orig_dict[name].shape != new_dict[name].shape:
                raise RuntimeError(
                    f"Shape mismatch for '{name}': Expected {orig_dict[name].shape}, "
                    f"got {new_dict[name].shape}"
                )

            orig_dict[name].copy_(new_dict[name])
