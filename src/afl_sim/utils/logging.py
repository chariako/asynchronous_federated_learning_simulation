import json
from pathlib import Path
from typing import Any, TextIO

from loguru import logger


class MetricsLogger:
    """
    Manages the disk I/O for recording simulation performance metrics over time.

    Attributes:
        run_dir (Path): The root directory for the current simulation run.
        metrics_file (Path): The resolved file path to the JSON Lines metrics log.
    """

    def __init__(self, run_dir: Path):
        """
        Initializes the metrics logger.

        Args:
            run_dir (Path): The directory where the `metrics.jsonl` file will be created.
        """
        self.run_dir = run_dir
        self.metrics_file = run_dir / "metrics.jsonl"
        self._file: TextIO | None = None
        self._number_of_lines_triggering_flush = 1000
        self._tmp_metrics_file = self.run_dir / "tmp_metrics.jsonl"

    def __enter__(self) -> "MetricsLogger":
        """
        Opens the metrics file when entering a 'with' block.
        """
        self._file = self.metrics_file.open(mode="a", encoding="utf-8")
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """
        Safely closes the metrics file when exiting the 'with' block,
        even if an exception was raised.
        """
        if self._file is not None:  # pragma: no branch
            self._file.close()
            self._file = None

    def log(
        self, global_idx: int, sim_time: float, loss: float, accuracy: float
    ) -> None:
        """
        Appends a single metric entry to the JSON Lines file.

        Args:
            global_idx (int): The current global iteration or round index.
            sim_time (float): The current elapsed time within the simulated environment.
            loss (float): The evaluated global loss value.
            accuracy (float): The evaluated global accuracy metric.

        Raises:
            RuntimeError: If the MetricsLogger is not used within a context manager.
        """
        if self._file is None:
            raise RuntimeError(
                "MetricsLogger must be used within a context manager. "
                "Wrap your execution in: 'with metrics_logger:'"
            )

        entry = {
            "global_idx": global_idx,
            "sim_time": sim_time,
            "loss": loss,
            "accuracy": accuracy,
        }

        self._file.write(json.dumps(entry) + "\n")

        if global_idx % self._number_of_lines_triggering_flush == 0:
            self.flush_log_file()

    def flush_log_file(self) -> None:
        if self._file is not None:  # pragma: no branch
            self._file.flush()

    def trim_history(self, next_global_idx: int) -> None:
        """
        Rewinds the metrics log to a specific global index, removing later entries.

        Used when resuming a simulation from an older checkpoint to prevent duplicate
        or orphaned metric logs. Reads the existing file, writes valid entries to a
        temporary file, and performs an atomic replace.

        Args:
            next_global_idx (int): The global index marking the next valid state.
                Any log entries with a global index greater than or equal to this value
                will be discarded.

        Raises:
            FileNotFoundError: If the metrics file does not exist before trimming.
            ValueError: If the metrics file is corrupted.
        """
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"No metrics file found at {self.metrics_file}")

        try:
            with (
                self.metrics_file.open("r", encoding="utf-8") as f_in,
                self._tmp_metrics_file.open("w", encoding="utf-8") as f_out,
            ):
                for line_num, line in enumerate(f_in):
                    try:
                        data = json.loads(line)

                        if data.get("global_idx", -1) >= next_global_idx:
                            break

                        f_out.write(line)

                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Critical Error: The metrics log file is corrupted at line {line_num} "
                            "and cannot be parsed."
                        ) from error

            self._tmp_metrics_file.replace(self.metrics_file)

        except PermissionError:
            logger.warning(
                f"Could not trim {self.metrics_file.name} (locked by another process). "
                "New metrics will append to the end."
            )
            self._tmp_metrics_file.unlink(missing_ok=True)
