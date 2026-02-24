from pathlib import Path
from typing import Any

import torch
from loguru import logger

from afl_sim.types import BestCheckpoint, LatestCheckpoint


class CheckpointManager:
    """
    Manages the saving and loading of simulation checkpoints and model weights.
    """

    def __init__(self, checkpoint_dir: Path):
        self.ckpt_dir = checkpoint_dir

        self.latest_path = self.ckpt_dir / "checkpoint_latest.pt"
        self.best_path = self.ckpt_dir / "model_best.pt"

        self.best_acc = -1.0

    def _atomic_write(
        self, data: LatestCheckpoint | BestCheckpoint, file_path: Path
    ) -> None:
        """
        Safely writes data to a file using an atomic operation.
        """
        tmp_path = file_path.parent / ("tmp_" + file_path.name)

        try:
            torch.save(data, tmp_path)
            tmp_path.replace(file_path)

        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def save_latest(self, payload: dict[str, Any], next_event: int) -> None:
        """
        Saves the current simulation state to the 'latest' checkpoint.
        """
        data = LatestCheckpoint(payload=payload, next_event=next_event)
        self._atomic_write(data, self.latest_path)

    def save_best(self, model_state_dict: dict[str, Any], current_acc: float) -> bool:
        """
        Saves the model weights if the test accuracy has improved.
        """
        if current_acc > self.best_acc:
            logger.info(
                f"Validation Improvement: {self.best_acc:.2f}% -> {current_acc:.2f}%"
            )
            self.best_acc = current_acc
            data = BestCheckpoint(
                model_state_dict=model_state_dict, accuracy=current_acc
            )

            self._atomic_write(data, self.best_path)
            return True

        return False

    def load_latest(self) -> LatestCheckpoint:
        """
        Loads the resume-capable checkpoint from disk.
        """
        if not self.latest_path.exists():
            raise FileNotFoundError(f"No resume checkpoint found at {self.latest_path}")

        data: LatestCheckpoint = torch.load(
            self.latest_path, map_location="cpu", weights_only=False
        )

        return data

    def get_best_accuracy(self) -> float:
        """
        Retrieves the best validation accuracy recorded during the session.
        """
        return self.best_acc

    def update_best_accuracy(self, acc: float) -> None:
        """
        Update best accuracy.
        """
        self.best_acc = acc
        logger.success(f"Restored best accuracy tracking: {self.best_acc:.2f}%")
