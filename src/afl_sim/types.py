from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from torch import Tensor, device
    from torch.nn import Module
    from torchvision.models import ResNet


type TensorDict = dict[str, Tensor]
type SimulationModel = Module | ResNet
type SimulationDevice = device


@dataclass(frozen=True, slots=True)
class ServerState:
    """
    Immutable data structure representing the current state of the central server.

    Attributes:
        model_state (TensorDict): The state dictionary of the current global model.
        buffer (TensorDict): The aggregated buffer containing incoming client updates.
        current_count (int): The number of client updates currently stored in the buffer.
        best_acc (float): The highest validation/test accuracy achieved in the simulation so far.
        current_acc (float): The validation/test accuracy of the most recent model iteration.
        current_version (int): The current integer version of the global model.
    """

    model_state: TensorDict
    buffer: TensorDict
    current_count: int
    best_acc: float
    current_acc: float
    current_version: int


class LatestMetadataSchema(BaseModel):
    """
    Pydantic schema for validating and serializing core simulation metadata.

    Attributes:
        global_idx (int): The current discrete event index of the simulation clock.
        best_acc (float): The highest validation/test accuracy achieved so far.
        current_acc (float): The validation/test accuracy of the most recent model.
        current_version (int): The current integer version of the global model.
        current_server_count (int): The number of client updates currently in the server buffer.
        history_version_list (list[int]): A list of model versions actively maintained in the asynchronous history.
    """

    global_idx: int
    best_acc: float
    current_acc: float
    current_version: int
    current_server_count: int
    history_version_list: list[int]
