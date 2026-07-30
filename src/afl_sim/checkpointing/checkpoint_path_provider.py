from pathlib import Path
from typing import Literal

from afl_sim.enums import CheckpointFile


class CheckpointPathProvider:
    """
    Provides resolved checkpoint paths for I/O file operations.

    Attributes:
        ckpt_dir (Path): The root directory where all checkpoint files are stored.
        tmp_dir (Path): The temporary file directory utilized for atomic write operations.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initializes the CheckpointPathProvider with the requisite file paths.

        Args:
            checkpoint_dir (Path): The root directory where all checkpoint files are stored.
            checkpoint_config (CheckpointConfig): Configuration parameters governing checkpointing behavior and intervals.
        """
        self.ckpt_dir = checkpoint_dir
        self.tmp_dir = checkpoint_dir.parent / f"tmp_{checkpoint_dir.name}"

    def get_path(self, file_type: CheckpointFile | str, tmp: bool = False) -> Path:
        """Resolves the path for a standard file enum or custom string."""
        parent = self.tmp_dir if tmp else self.ckpt_dir
        return parent / file_type

    def get_client_state_path(self, cid: int, tmp: bool = False) -> Path:
        """
        Generates the file path for the specific memory state of a registered client.

        Args:
            cid (int): The unique client identifier.
            tmp (bool, optional): Indicates whether to return the path within the temporary directory for atomic operations. Defaults to False.

        Returns:
            Path: The target file path for the client's state.
        """
        return self.get_path(f"latest_client_{cid}_state.pt", tmp)

    def get_history_version_path(
        self, version: int | Literal["*"], tmp: bool = False
    ) -> Path:
        """
        Generates the file path or glob pattern for a specific model history version.

        Args:
            version (int | Literal["*"]): The specific version integer, or a wildcard string ("*") utilized for globbing operations.
            tmp (bool, optional): Indicates whether to return the path within the temporary directory for atomic operations. Defaults to False.

        Returns:
            Path: The target file path or pattern for the history version.
        """
        return self.get_path(f"latest_history_version_{version}.pt", tmp)
