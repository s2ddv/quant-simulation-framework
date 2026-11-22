"""Run a new GBM simulation for the dashboard with an explicit memory budget.

This is the only dashboard path that uses the network (historical prices from
yfinance). It records the downloaded closes in a checksummed snapshot, exactly
like the CLI, so a saved run can later be replayed offline. Company
fundamentals are not fetched here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from calibration.fundamentals import normalize_ticker
from simulation.report import SimulationConfig, historical_snapshot, run_simulation

# Interactive sessions keep the path matrix in memory while a run is shown, so
# the dashboard budget is half of the simulator's own 256 MiB working limit.
DEFAULT_MAX_PATH_BYTES = 128 * 1024 * 1024


def path_memory_bytes(steps: int, n_sims: int) -> int:
    """Bytes of the float64 ``(steps + 1, n_sims)`` path matrix."""
    return 8 * (int(steps) + 1) * int(n_sims)


def check_path_budget(config: SimulationConfig,
                      max_bytes: int = DEFAULT_MAX_PATH_BYTES) -> int:
    """Return the path size or raise before any download or simulation."""
    required = path_memory_bytes(config.steps, config.n_sims)
    if required > max_bytes:
        raise ValueError(
            f"The path matrix needs {required / 2**20:.0f} MiB; the dashboard limit is "
            f"{max_bytes / 2**20:.0f} MiB. Reduce steps or simulations.")
    return required


def run_new_simulation(ticker: str, period: str, config: SimulationConfig, *,
                       out_of_sample: dict | None = None,
                       sensitivity: dict | None = None,
                       max_path_bytes: int = DEFAULT_MAX_PATH_BYTES,
                       download: Callable[..., pd.Series] | None = None,
                       now: datetime | None = None) -> tuple[dict, np.ndarray]:
    """Download prices, snapshot them and simulate; returns ``(report, paths)``.

    ``download`` defaults to ``calibration.data.download_prices`` and can be
    replaced in offline tests. ``now`` must carry a timezone when given.
    """
    ticker = normalize_ticker(ticker)
    if not period or not period.strip():
        raise ValueError("period must not be empty.")
    check_path_budget(config, max_path_bytes)
    if download is None:
        from calibration.data import download_prices
        download = download_prices
    prices = download(ticker, period=period)
    retrieved_at = (now or datetime.now(timezone.utc)).isoformat()
    snapshot = historical_snapshot(prices, ticker=ticker, period=period,
                                   retrieved_at=retrieved_at)
    return run_simulation(snapshot, config, out_of_sample, sensitivity)
