from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SimulationDirectories:
    """
    A container for the essential directory paths used during a simulation run.

    Attributes:
        output_dir (Path): The primary output directory for the current simulation run.
        checkpoint_dir (Path): The directory where simulation checkpoints are saved.
        data_dir (Path): The directory for saving or loading input data, datasets, and splits.
    """

    output_dir: Path
    checkpoint_dir: Path
    data_dir: Path


@dataclass(frozen=True)
class PartitionPathCollection:
    """
    A unified collection of file paths representing a saved data partition state.

    Attributes:
        data_path (Path): Path to the serialized `.npz` data file containing split indices.
        meta_path (Path): Path to the `.json` metadata file describing the partition.
        plot_path (Path): Path to the `.png` visual representation of the split distribution.
    """

    data_path: Path
    meta_path: Path
    plot_path: Path

    @classmethod
    def from_hash(cls, data_dir: Path, hash_str: str) -> PartitionPathCollection:
        """
        Constructs a path collection based on the base directory and a unique configuration hash.

        Args:
            data_dir (Path): The root directory where data artifacts are stored.
            hash_str (str): The unique identifier hash for the current partition configuration.

        Returns:
            PartitionPathCollection: The initialized collection of resolved paths.
        """
        return cls(
            data_path=data_dir / f"{hash_str}.npz",
            meta_path=data_dir / f"{hash_str}.json",
            plot_path=data_dir / f"{hash_str}.png",
        )


@dataclass(frozen=True)
class ClockPathCollection:
    """
    A unified collection of file paths representing a simulated execution clock.

    Attributes:
        data_path (Path): Path to the base directory containing partitioned clock chunks.
        meta_path (Path): Path to the `.json` metadata file describing the clock generation parameters.
        plot_path (Path): Path to the `.png` visual representation of client arrival distributions.
    """

    data_path: Path
    meta_path: Path
    plot_path: Path

    @classmethod
    def from_clock_specs(cls, data_dir: Path, hash_str: str) -> ClockPathCollection:
        """
        Constructs a path collection based on the base directory and a unique clock configuration hash.

        Args:
            data_dir (Path): The root directory where clock artifacts are stored.
            hash_str (str): The unique identifier hash for the current clock configuration.

        Returns:
            ClockPathCollection: The initialized collection of resolved paths.
        """
        return cls(
            data_path=data_dir,
            meta_path=data_dir / f"{hash_str}.json",
            plot_path=data_dir / f"{hash_str}.png",
        )

    def get_clock_chunk_path(self, chunk_num: int) -> Path:
        """
        Resolves the specific file path for a numbered clock chunk `.npz` file.

        Args:
            chunk_num (int): The integer index of the requested chunk.

        Returns:
            Path: The resolved file path for the data chunk.
        """
        return self.data_path / f"chunk{chunk_num}.npz"
