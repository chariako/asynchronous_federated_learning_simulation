from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathCollection:
    """Collection of paths for saved input data."""

    data_path: Path
    meta_path: Path
    plot_path: Path

    @classmethod
    def from_hash(cls, data_dir: Path, hash: str) -> PathCollection:
        return cls(
            data_path=data_dir / f"{hash}.npz",
            meta_path=data_dir / f"{hash}.json",
            plot_path=data_dir / f"{hash}.png",
        )
