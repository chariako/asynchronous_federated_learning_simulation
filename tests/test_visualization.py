import numpy as np

from afl_sim.utils import save_clock_plot, save_partition_plot


def test_clock_plot_async_manual(tmp_path):
    """
    Test if async clock plots are properly rendered.
    """
    timestamps = np.array([0.1, 0.5, 1.2, 2.4, 3.0])
    client_ids = np.array([0, 1, 2, 0, 1])
    num_clients = 3

    output_file = tmp_path / "async_manual.png"

    save_clock_plot(
        timestamps=timestamps,
        client_ids=client_ids,
        num_clients=num_clients,
        filepath=output_file,
        is_async=True,
    )

    assert output_file.exists()


def test_clock_plot_sync_manual(tmp_path):
    """
    Test if sync clock plots are properly rendered.
    """
    timestamps = np.array([1.0, 2.0, 3.0, 4.0])
    client_ids = np.array(
        [
            [0, 1],
            [1, 2],
            [1, 2],
            [0, 2],
        ]
    )  # two clients per round
    num_clients = 3

    output_file = tmp_path / "sync_manual.png"

    save_clock_plot(
        timestamps=timestamps,
        client_ids=client_ids,
        num_clients=num_clients,
        filepath=output_file,
        is_async=False,
    )

    assert output_file.exists()


def test_partition_plot_manual(tmp_path):
    """
    Test partition plot with explicit indices.
    """
    targets = np.array([0, 1, 0, 1])

    client_indices = [np.array([0, 2]), np.array([1, 3])]
    num_clients = 2

    output_file = tmp_path / "partition_manual.png"

    save_partition_plot(
        targets=targets,
        client_indices=client_indices,
        num_clients=num_clients,
        filepath=output_file,
    )

    assert output_file.exists()
