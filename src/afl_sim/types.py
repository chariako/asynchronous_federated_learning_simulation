from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class SimulationState(TypedDict):
    server: ServerState
    clients: dict[str, ClientState]


class ClientState(TypedDict):
    memory: dict[str, Any] | None
    stale_state: dict[str, Any] | None


class ServerState(TypedDict):
    model_state: dict[str, Any]
    buffer: dict[str, Any]
    current_count: int
    best_acc: float
    current_acc: float


@dataclass(frozen=True)
class PathCollection:
    """Collection of paths for saved input data."""

    data_path: Path
    meta_path: Path
    plot_path: Path

    @classmethod
    def from_hash(cls, data_dir: Path, hash_str: str) -> PathCollection:
        return cls(
            data_path=data_dir / f"{hash_str}.npz",
            meta_path=data_dir / f"{hash_str}.json",
            plot_path=data_dir / f"{hash_str}.png",
        )

    @classmethod
    def from_clock_specs(cls, data_dir: Path, hash_str: str) -> PathCollection:
        return cls(
            data_path=data_dir,
            meta_path=data_dir / f"{hash_str}.json",
            plot_path=data_dir / f"{hash_str}.png",
        )

    def get_clock_chunk_path(self, chunk_num: int) -> Path:
        return self.data_path / f"chunk{chunk_num}.npz"


class LatestCheckpoint(TypedDict):
    simulation_state: SimulationState
    global_next_event: int


class BestCheckpoint(TypedDict):
    model_state_dict: dict[str, Any]
    accuracy: float
