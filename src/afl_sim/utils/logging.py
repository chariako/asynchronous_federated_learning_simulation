import json
from pathlib import Path

from loguru import logger


class MetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metrics_file = run_dir / "metrics.jsonl"

    def log(
        self, event_idx: int, sim_time: float, loss: float, accuracy: float
    ) -> None:
        """
        Logs metrics to file.
        """
        entry = {
            "event_idx": event_idx,
            "sim_time": sim_time,
            "loss": loss,
            "accuracy": accuracy,
        }

        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def trim_history(self, resume_from_idx: int) -> None:
        """
        Rewinds the log to the state at resume_from_idx.
        Removes any entries recorded after this index.
        """
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"No metrics file found at {self.metrics_file}")

        tmp_metrics_file = self.run_dir / "tmp_metrics.jsonl"

        try:
            with (
                self.metrics_file.open("r", encoding="utf-8") as f_in,
                tmp_metrics_file.open("w", encoding="utf-8") as f_out,
            ):
                for line_num, line in enumerate(f_in):
                    try:
                        data = json.loads(line)
                        if data.get("event_idx", -1) > resume_from_idx:
                            continue

                        f_out.write(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping corrupt JSON at line {line_num + 1}")
                        continue

            tmp_metrics_file.replace(self.metrics_file)

        except PermissionError:
            logger.warning(
                f"Could not trim {self.metrics_file.name} (locked by another process). "
                "New metrics will append to the end."
            )
            tmp_metrics_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to clean metrics file: {e}")
            tmp_metrics_file.unlink(missing_ok=True)
