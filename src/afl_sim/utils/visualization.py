import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def save_clock_plot(
    timestamps: NDArray[np.float64],
    client_ids: NDArray[np.int64],
    num_clients: int,
    filepath: Path,
    is_async: bool,
) -> None:
    """
    Generates the clock event dashboard.
    """
    if is_async:
        times = timestamps
    else:
        num_clients_per_round = client_ids.shape[1]
        times = np.repeat(timestamps, num_clients_per_round)
        client_ids = client_ids.flatten().astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- Plot 1: Timeline ---
    zoom_idx = 100 if len(times) > 100 else len(times)

    ax1.scatter(times[:zoom_idx], client_ids[:zoom_idx], alpha=0.7, s=20, c="tab:blue")
    ax1.set_title(f"Event Timeline (First {zoom_idx} events)")
    ax1.set_xlabel("Simulated Time")
    ax1.set_ylabel("Client ID")
    ax1.set_yticks(range(0, num_clients, max(1, num_clients // 10)))
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # --- Plot 2: Participation Counts ---
    counts = np.bincount(client_ids, minlength=num_clients)

    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]

    ax2.bar(range(num_clients), sorted_counts, color="tab:orange", alpha=0.8)
    ax2.set_title("Total Participation per Client (Sorted)")
    ax2.set_xlabel("Client Rank (Most Active -> Least Active)")
    ax2.set_ylabel("Number of Events")

    # Add text stats
    stats_text = (
        f"Total Events: {len(times)}\n"
        f"Max Events: {counts.max()}\n"
        f"Min Events: {counts.min()}\n"
        f"Active Clients: {np.count_nonzero(counts)}/{num_clients}"
    )
    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)


def save_partition_plot(
    targets: NDArray[np.int64],
    client_indices: list[np.ndarray],
    num_clients: int,
    filepath: Path,
) -> None:
    """
    Generates the class distribution stacked bar chart.
    """
    num_classes = len(np.unique(targets))
    counts_matrix = np.zeros((num_clients, num_classes), dtype=int)

    for i in range(num_clients):
        indices = client_indices[i]
        if len(indices) == 0:
            continue

        client_targets = targets[indices]
        counts = np.bincount(client_targets, minlength=num_classes)
        counts_matrix[i] = counts

    fig, ax = plt.subplots(figsize=(15, 8))

    if num_classes <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.get_cmap("turbo")
        colors = [cmap(i / num_classes) for i in range(num_classes)]

    bottom = np.zeros(num_clients)

    for cls_idx in range(num_classes):
        values = counts_matrix[:, cls_idx]
        ax.barh(
            y=range(num_clients),
            width=values,
            left=bottom,
            height=0.8,
            color=colors[cls_idx],
            edgecolor="white",
            linewidth=0.5,
            label=f"Class {cls_idx}",
        )
        bottom += values

    ax.set_ylabel("Client ID")
    ax.set_xlabel("Number of Samples")
    ax.set_title("Data Distribution per Client")
    ax.set_yticks(range(num_clients))

    if num_clients <= 50:
        ax.set_yticklabels([f"Client {i}" for i in range(num_clients)], fontsize=8)
    else:
        ax.set_yticklabels([])

    if num_classes <= 20:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
