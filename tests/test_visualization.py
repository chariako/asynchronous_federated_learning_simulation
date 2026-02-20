import numpy as np
import pytest

from afl_sim.utils import save_clock_plot, save_partition_plot


def test_clock_plot_manual(tmp_path):
    """
    Test if clock plots are properly rendered.
    """
    clock_len = 1000
    num_clients = 10
    timestamps = np.arange(clock_len * 1.0)
    client_ids = np.repeat(np.arange(num_clients), np.ceil(clock_len / num_clients))[
        :clock_len
    ]

    output_file = tmp_path / "clock_plot_manual.png"

    save_clock_plot(
        timestamps=timestamps,
        client_ids=client_ids,
        num_clients=num_clients,
        filepath=output_file,
    )

    assert output_file.exists()


@pytest.mark.parametrize(
    "timestamps, client_ids, expected_error",
    [
        (np.ones(100), np.ones((100, 2)), "requires 1D client_ids"),
        (np.ones((100, 2)), np.ones(100), "requires 1D timestamps"),
    ],
)
def test_clock_dimension_mismatch_raises_error(
    timestamps, client_ids, expected_error, tmp_path
):
    """
    Ensure that clock visualization raises error
    if inputs are not 1D.
    """
    with pytest.raises(ValueError, match=expected_error):
        save_clock_plot(
            timestamps=timestamps,
            client_ids=client_ids,
            num_clients=10,
            filepath=tmp_path / "output.png",
        )


@pytest.mark.parametrize("num_classes", [10, 100])
def test_partition_plot_manual(tmp_path, num_classes):
    """
    Test partition plot with explicit indices.
    """
    num_samples = 1000
    targets = np.repeat(np.arange(num_classes), num_samples // num_classes)

    num_clients = 10
    client_indices = np.array_split(np.arange(num_samples), num_clients)

    output_file = tmp_path / "partition_manual.png"

    save_partition_plot(
        targets=targets,
        client_indices=client_indices,
        num_clients=num_clients,
        num_classes=num_classes,
        filepath=output_file,
    )

    assert output_file.exists()
