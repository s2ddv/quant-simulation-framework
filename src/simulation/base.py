"""Common interface and state for stochastic price simulators."""

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


class BaseStochasticSimulator(ABC):
    """Base for simulators returning paths shaped ``(steps + 1, n_sims)``.

    The horizon is measured in years. Each instance owns a NumPy random
    generator; repeated calls advance its state.
    """

    MAX_WORKING_BYTES = 256 * 1024 * 1024

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None) -> None:
        if (isinstance(s0, bool) or not isinstance(s0, Real)
                or not np.isfinite(s0) or s0 <= 0):
            raise ValueError("s0 must be finite and positive")
        if (isinstance(horizon, bool) or not isinstance(horizon, Real)
                or not np.isfinite(horizon) or horizon <= 0):
            raise ValueError("horizon must be finite and positive")
        if (isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0):
            raise ValueError("steps must be a positive integer")
        if (isinstance(n_sims, bool) or not isinstance(n_sims, Integral) or n_sims <= 0):
            raise ValueError("n_sims must be a positive integer")
        if seed is not None and (isinstance(seed, bool)
                                 or not isinstance(seed, Integral) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")

        self.s0: float = float(s0)
        self.horizon: float = float(horizon)
        self.steps: int = int(steps)
        self.n_sims: int = int(n_sims)
        self.seed: int | None = None if seed is None else int(seed)
        self.dt: float = self.horizon / self.steps
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("horizon / steps must be finite and positive")
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

    def _check_memory(self, *, temporary_arrays: int = 0) -> None:
        """Reject paths whose estimated peak array storage exceeds the budget.

        The returned float64 matrix is mandatory. Each requested temporary
        array has the size of the increment matrix. Small library overhead is
        not included in this estimate.
        """
        output_bytes = 8 * (self.steps + 1) * self.n_sims
        temporary_bytes = 8 * self.steps * self.n_sims * temporary_arrays
        required = output_bytes + temporary_bytes
        if required > self.MAX_WORKING_BYTES:
            raise ValueError(
                f"path arrays require {required} bytes (output {output_bytes}); "
                f"limit is {self.MAX_WORKING_BYTES} bytes"
            )

    @abstractmethod
    def calibrate(self, historical_prices: pd.Series) -> None:
        """Estimate model parameters from chronological historical prices."""

    @abstractmethod
    def generate_paths(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Generate price paths, including the initial price in row zero."""
