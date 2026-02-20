import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from loguru import logger
from numpy.random import SeedSequence, default_rng
from numpy.typing import NDArray

from afl_sim.config import AppConfig, SyncStrategy
from afl_sim.types import PathCollection
from afl_sim.utils import compute_hash_from_dict, save_clock_plot

_MIN_HORIZON = 3000.0
_OVERSAMPLING_FACTOR = 1.2


# --- Define Clock Object ---
class ClockData(TypedDict):
    timestamps: NDArray[np.float64]
    client_ids: NDArray[np.int64]


def _create_config_dict(config: AppConfig) -> dict[str, Any]:
    """
    Generates a canonical dictionary of clock parameters.
    """
    is_async = config.comm_strategy.type == "async"
    return {
        "num_clients": config.simulation.num_clients,
        "sigma": config.simulation.client_rate_std,
        "seed": config.simulation.rate_seed,
        "is_async": is_async,
        "sample_size": (
            config.comm_strategy.sample_size
            if isinstance(config.comm_strategy, SyncStrategy)
            else None
        ),
    }


def read_clock_event(clock: ClockData, idx: int) -> tuple[float, list[int]]:
    """
    Extracts timestamp and client IDs.
    """
    sim_time = float(clock["timestamps"][idx])
    raw_clients = clock["client_ids"][idx]

    if isinstance(raw_clients, np.ndarray):
        return sim_time, raw_clients.tolist()

    return sim_time, [int(raw_clients)]


def get_clock_length(clock: ClockData) -> int:
    """
    Returns the number of events.
    """
    return len(clock["timestamps"])


def get_clock(config: AppConfig, data_dir: Path) -> ClockData:
    """
    Retrieves or generates a simulation clock using Superset Caching.
    """
    output_dir = data_dir / "clocks"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_duration = config.simulation.duration_sim_units
    config_dict = _create_config_dict(config)
    config_hash = compute_hash_from_dict(config_dict)
    visualize = config.visualization.visualize_client_arrivals

    paths = PathCollection.from_hash(output_dir, config_hash)

    # --- Attempt Load ---
    if paths.meta_path.exists() and paths.data_path.exists():
        try:
            with open(paths.meta_path) as f:
                metadata = json.load(f)

            cached_duration = metadata.get("actual_duration", 0.0)

            if cached_duration >= target_duration:
                logger.info(
                    f"Loading existing clock: {config_hash} (T={cached_duration:.1f} >= {target_duration})"
                )
                return _load_and_slice(paths.data_path, target_duration)

            logger.info(
                f"Cache Upgrade: Existing T={cached_duration} < Requested {target_duration}. Regenerating..."
            )

        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt metadata found. Regenerating from scratch.")

    # --- Generate New (Superset) ---
    gen_duration = max(target_duration, _MIN_HORIZON)

    # Generate Rates
    rates = _get_client_rates(
        config.simulation.num_clients,
        config.simulation.client_rate_std,
        config.simulation.rate_seed,
    )

    if config.comm_strategy.type == "async":
        clock_data = _generate_async(rates, gen_duration, config.simulation.rate_seed)
    else:
        if not isinstance(config.comm_strategy, SyncStrategy):
            raise TypeError("Invalid strategy for synchronous generation.")

        clock_data = _generate_sync(
            rates,
            gen_duration,
            config.simulation.rate_seed,
            config.simulation.num_clients,
            config.comm_strategy.sample_size,
        )

    # --- Save & Return ---
    _save_clock_packet(
        clock_data,
        metadata={"actual_duration": gen_duration, "config_hash": config_hash},
        paths=paths,
        config_dict=config_dict,
        visualize=visualize,
    )

    return _slice_clock(clock_data, target_duration)


def _load_and_slice(path: Path, duration: float) -> ClockData:
    with np.load(path) as data:
        clock = {
            "timestamps": data["timestamps"],
            "client_ids": data["client_ids"],
        }
    return _slice_clock(cast(ClockData, clock), duration)


def _slice_clock(clock: ClockData, limit: float) -> ClockData:
    mask = clock["timestamps"] <= limit
    return {
        "timestamps": clock["timestamps"][mask],
        "client_ids": clock["client_ids"][mask],
    }


def _get_client_rates(
    num_clients: int, sigma_rate: float, seed: int
) -> NDArray[np.float64]:
    """
    Generates deterministic client rates (Poisson parameters)
    by sampling a zero-mean lognormal distribution.
    """
    rng = default_rng(seed=seed + 0)
    return rng.lognormal(0.0, sigma_rate, num_clients)


def _generate_async(
    rates: NDArray[np.float64], duration: float, seed: int
) -> ClockData:
    """
    Generates asynchronous events using Independent Streams per client.
    """
    ss = SeedSequence(seed + 1)
    child_states = ss.spawn(len(rates))

    est_events = np.ceil(rates * duration * _OVERSAMPLING_FACTOR).astype(int)
    all_timestamps = []
    all_client_ids = []

    for cid, (rate, child_state) in enumerate(zip(rates, child_states, strict=True)):
        n = est_events[cid]
        if n == 0:
            continue

        rng = default_rng(child_state)
        intervals = rng.exponential(1.0 / rate, size=n)
        times = np.cumsum(intervals)

        # Optimization: Pre-filter before appending to reduce memory pressure
        valid_times = times[times <= duration]

        if len(valid_times) > 0:
            all_timestamps.append(valid_times)
            all_client_ids.append(np.full(len(valid_times), cid, dtype=np.int64))

    if not all_timestamps:
        return {"timestamps": np.array([]), "client_ids": np.array([])}

    flat_times = np.concatenate(all_timestamps)
    flat_ids = np.concatenate(all_client_ids)

    # Sort events by time
    sort_idx = np.argsort(flat_times)

    return {
        "timestamps": flat_times[sort_idx],
        "client_ids": flat_ids[sort_idx],
    }


def _generate_sync(
    rates: NDArray[np.float64],
    duration: float,
    seed: int,
    num_clients: int,
    sample_size: int,
) -> ClockData:
    """
    Generates synchronous rounds.
    """
    ss = SeedSequence(seed + 2)
    rng_select, rng_delay = [default_rng(s) for s in ss.spawn(2)]

    avg_rate = np.mean(rates)
    est_rounds = int(duration * avg_rate * _OVERSAMPLING_FACTOR)

    # --- Client Selection ---
    rand_matrix = rng_select.random((est_rounds, num_clients))

    selections = np.argpartition(rand_matrix, sample_size, axis=1)[:, :sample_size]
    selections = selections.astype(np.int64)

    # --- Delays ---
    sel_rates = rates[selections]
    delays = rng_delay.exponential(1.0 / sel_rates)

    round_durations = np.max(delays, axis=1)
    round_ends = np.cumsum(round_durations)

    valid_mask = round_ends <= duration

    return {
        "timestamps": round_ends[valid_mask],
        "client_ids": selections[valid_mask],
    }


def _save_clock_packet(
    clock: ClockData,
    metadata: dict[str, Any],
    paths: PathCollection,
    config_dict: dict[str, Any],
    visualize: bool,
) -> None:
    """
    Saves clock structure to disk along with metadata.
    Optionally saves a visualization of client arrivals.
    """

    logger.info(
        f"Saving clock (T={metadata['actual_duration']:.1f}) to {paths.data_path.name}"
    )

    # Save Data
    np.savez_compressed(
        paths.data_path,
        timestamps=clock["timestamps"],
        client_ids=clock["client_ids"],
    )

    # Save Metadata
    full_meta = {
        "actual_duration": metadata["actual_duration"],
        "config_hash": metadata["config_hash"],
        "parameters": config_dict,
    }

    with open(paths.meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    # Visualization
    if visualize:
        try:
            logger.info("Saving clock visualization...")

            # Extract plot data
            timestamps = clock["timestamps"]
            client_ids = clock["client_ids"]

            # Prepare sync data for visualization
            if config_dict["sample_size"]:
                timestamps = np.repeat(timestamps, config_dict["sample_size"])
                client_ids = client_ids.flatten().astype(int)

            save_clock_plot(
                timestamps=timestamps,
                client_ids=client_ids,
                num_clients=config_dict["num_clients"],
                filepath=paths.plot_path,
            )
        except Exception as e:
            logger.warning(f"Skipping clock visualization due to error: {e}")
