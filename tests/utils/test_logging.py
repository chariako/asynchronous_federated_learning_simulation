import json

import pytest

from afl_sim.utils.logging import MetricsLogger


@pytest.fixture
def metrics_logger(tmp_path):
    return MetricsLogger(run_dir=tmp_path)


def test_default_none_open_file(metrics_logger):
    assert metrics_logger._file is None


def test_metrics_file_open_close(metrics_logger):
    with metrics_logger:
        assert metrics_logger._file is not None
        assert not metrics_logger._file.closed

    assert metrics_logger._file is None


def test_log_raises_runtime_error(metrics_logger):
    with pytest.raises(RuntimeError, match="must be used within a context manager"):
        metrics_logger.log(global_idx=42, sim_time=100, loss=0.1, accuracy=0.6)


@pytest.mark.parametrize(
    ("global_idx", "flush_trigger", "flush_called"), [(100, 10, True), (25, 10, False)]
)
def test_log_flush_frequency(
    metrics_logger, mocker, global_idx, flush_trigger, flush_called
):
    metrics_logger._flush_trigger = flush_trigger

    with metrics_logger:
        spy = mocker.spy(metrics_logger._file, name="flush")
        metrics_logger.log(global_idx=global_idx, sim_time=100, loss=0.8, accuracy=0.9)

        assert spy.called == flush_called


def test_log_appends_to_file(metrics_logger):
    global_idx = metrics_logger._flush_trigger
    sim_time = 100
    loss = 0.7
    accuracy = 0.5

    with metrics_logger:
        metrics_logger.log(
            global_idx=global_idx, sim_time=sim_time, loss=loss, accuracy=accuracy
        )

    assert metrics_logger.metrics_file.exists()

    with metrics_logger.metrics_file.open() as file:
        first_line = file.readline()
        file_contents = json.loads(first_line)

    assert "global_idx" in file_contents
    assert file_contents["global_idx"] == global_idx
    assert "sim_time" in file_contents
    assert file_contents["sim_time"] == sim_time
    assert "loss" in file_contents
    assert file_contents["loss"] == loss
    assert "accuracy" in file_contents
    assert file_contents["accuracy"] == accuracy


def _get_max_idx(metrics_file) -> int:
    max_idx = 0

    with metrics_file.open() as file:
        for line in file:
            data = json.loads(line)
            idx = data.get("global_idx", -1)
            if max_idx < idx:
                max_idx = idx

    return max_idx


def test_trimm_history(metrics_logger):
    num_lines = 10
    metrics_logger._flush_trigger = num_lines

    with metrics_logger:
        for i in range(num_lines):
            metrics_logger.log(global_idx=i, sim_time=42, loss=42, accuracy=0.2)

    old_max_idx = _get_max_idx(metrics_logger.metrics_file)
    assert old_max_idx == 9

    metrics_logger.trim_history(next_global_idx=5)

    new_max_idx = _get_max_idx(metrics_logger.metrics_file)
    assert new_max_idx == 4


def test_trimm_history_raises_file_not_found(metrics_logger):
    with pytest.raises(FileNotFoundError, match="No metrics file found"):
        metrics_logger.trim_history(next_global_idx=10)


def test_trimm_history_raises_json_error(metrics_logger):
    metrics_logger.metrics_file.write_text("this_is_corrupted_text")
    with pytest.raises(ValueError, match="metrics log file is corrupted"):
        metrics_logger.trim_history(next_global_idx=10)


def test_trimm_history_logs_permission_error(metrics_logger, mocker, capture_logs):
    num_lines = 10
    metrics_logger._flush_trigger = num_lines

    with metrics_logger:
        for i in range(num_lines):
            metrics_logger.log(global_idx=i, sim_time=42, loss=42, accuracy=0.2)

    mocker.patch("pathlib.Path.replace", side_effect=PermissionError())

    metrics_logger.trim_history(next_global_idx=5)

    assert "locked by another process" in capture_logs.text
    assert not metrics_logger._tmp_metrics_file.exists()
