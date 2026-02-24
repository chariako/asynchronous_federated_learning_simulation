import pytest

from afl_sim.utils.checkpoint_manager import CheckpointManager


def test_latest_checkpoint_is_saved(tmp_path):
    """Ensures latest checkpoints are successfully saved."""
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmp_path)
    checkpoint_manager.save_latest(payload={"weights": [0.1, 0.2]}, next_event=42)
    expected_file = tmp_path / "checkpoint_latest.pt"
    assert expected_file.exists()


def test_round_trip_integrity(tmp_path):
    """Ensures that data saved is identical to data loaded."""
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmp_path)

    original_payload = {
        "server": {"weights": [0.1, 0.2], "buffer": [0.3, 0.4]},
        "client1": {"weights": [0.5, 0.6], "memory": None},
        "client2": {"weights": [0.7, 0.8], "memory": None},
    }
    event_idx = 42

    # Save
    checkpoint_manager.save_latest(payload=original_payload, next_event=event_idx)

    # Load
    loaded_checkpoint = checkpoint_manager.load_latest()

    assert loaded_checkpoint["next_event"] == event_idx
    assert loaded_checkpoint["payload"] == original_payload


@pytest.mark.parametrize(
    "initial_acc, new_acc, should_save",
    [
        (10.0, 100.0, True),
        (100.0, 10.0, False),
        (50.0, 50.0, False),
    ],
)
def test_save_best_behavior(tmp_path, initial_acc, new_acc, should_save):
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmp_path)
    checkpoint_manager.best_acc = initial_acc

    did_save = checkpoint_manager.save_best(
        model_state_dict={"weights": [0.1, 0.2]}, current_acc=new_acc
    )

    expected_file = tmp_path / "model_best.pt"

    assert did_save == should_save
    assert expected_file.exists() == should_save


def test_not_found_raises_error(tmp_path):
    """Ensures a FileNotFoundError is raised if a valid checkpoint is not found."""
    with pytest.raises(FileNotFoundError, match="No resume checkpoint found"):
        checkpoint_manager = CheckpointManager(tmp_path)
        checkpoint_manager.load_latest()
