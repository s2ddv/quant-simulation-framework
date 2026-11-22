"""Normal GBM paths and a heavy-tailed log-return scenario."""

from typing import Literal
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from calibration.estimation import TRADING_DAYS
from calibration.prices import validate_prices
from simulation.base import BaseStochasticSimulator
from simulation.variance_reduction import VarianceReduction, validate_variance_reduction

DistributionType = Literal["normal", "t_student"]


class GbmSimulator(BaseStochasticSimulator):
    """Simulate log-price increments with normal or standardized t shocks.

    ``mu`` and ``sigma`` are annualized. The normal mode is exact GBM.
    The t mode substitutes independent standardized t shocks at each grid
    step; its distribution depends on the step size and is not exact GBM.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float | None = None,
                 sigma: float | None = None, df: float | None = None) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed)
        self.mu: float | None = None
        self.sigma: float | None = None
        self.calibrated_df: float | None = None
        if (mu is None) != (sigma is None):
            raise ValueError("mu and sigma must be supplied together")
        if mu is not None and sigma is not None:
            if (isinstance(mu, bool) or isinstance(sigma, bool)
                    or not isinstance(mu, Real) or not isinstance(sigma, Real)
                    or not np.isfinite(mu) or not np.isfinite(sigma) or sigma < 0):
                raise ValueError("mu and sigma must be finite; sigma must be nonnegative")
            self.mu, self.sigma = float(mu), float(sigma)
        if df is not None:
            # Explicit daily-shock degrees of freedom (e.g. a stress scenario);
            # validated again in generate_paths, which requires df > 2.
            if isinstance(df, bool) or not isinstance(df, Real) or not np.isfinite(df) or df <= 2:
                raise ValueError("df must be finite and greater than 2")
            self.calibrated_df = float(df)

    def calibrate(self, historical_prices: pd.Series) -> None:
        """Fit annualized normal moments and t degrees of freedom to daily logs.

        The t fit estimates location and scale as nuisance parameters. The
        simulation retains the classical sample-moment ``mu`` and ``sigma``.
        """
        historical_prices = validate_prices(historical_prices, min_observations=3)
        values = historical_prices.to_numpy()

        returns = np.diff(np.log(values))
        if not np.isfinite(returns).all():
            raise ValueError("log returns exceed finite precision")
        daily_sigma = float(np.std(returns, ddof=1))
        sigma = float(np.sqrt(TRADING_DAYS) * daily_sigma)
        mu = float(TRADING_DAYS * np.mean(returns) + 0.5 * sigma**2)
        if not np.isfinite([mu, sigma]).all():
            raise ValueError("calibrated moments exceed finite precision")
        if daily_sigma == 0:
            # A point mass has no identifiable t degrees of freedom.
            self.mu, self.sigma, self.calibrated_df = mu, sigma, None
            return
        df, _, _ = student_t.fit(returns / daily_sigma)
        if not np.isfinite(df) or df <= 0:
            raise ValueError("t fit returned invalid degrees of freedom")
        self.mu, self.sigma, self.calibrated_df = mu, sigma, float(df)

    def _antithetic_pairs(self, variance_reduction: str, dist_type: str) -> int:
        """Validate the sampling mode; return ``m`` pairs (0 for ``"none"``)."""
        if validate_variance_reduction(variance_reduction) == "none":
            return 0
        if dist_type != "normal":
            raise ValueError("antithetic sampling requires dist_type='normal'")
        if self.n_sims % 2:
            raise ValueError("antithetic sampling requires an even n_sims")
        return self.n_sims // 2

    def generate_paths(self, dist_type: DistributionType = "normal", *,
                       variance_reduction: VarianceReduction = "none") -> np.ndarray:
        """Return ``(steps + 1, n_sims)`` positive prices.

        The output holds the working shocks and cumulative log returns in
        place. The t mode requires a calibrated ``df > 2`` and rescales the
        shocks to unit variance before applying the annualized volatility.
        Unrepresentable increments or prices raise ``ValueError``. In t
        mode ``mu`` controls the log-return location through the normal-mode
        convention; it is not a finite expected-price growth rate.

        ``variance_reduction="antithetic"`` (normal mode, even ``n_sims = 2m``)
        draws the ``(steps, m)`` shocks ``Z`` into columns ``0..m-1`` and
        writes ``-Z`` into columns ``m..2m-1``; column ``i`` pairs with
        ``i + m``. Shocks are drawn row by row into the output, which gives
        the same values as one C-order ``(steps, m)`` draw and needs no
        extra matrix. See ``simulation.variance_reduction``. The default
        ``"none"`` keeps the historical draw order and arithmetic.
        """
        if dist_type not in ("normal", "t_student"):
            raise ValueError("dist_type must be 'normal' or 't_student'")
        pairs = self._antithetic_pairs(variance_reduction, dist_type)
        if self.mu is None or self.sigma is None:
            raise ValueError("calibrate or supply mu and sigma before generating paths")
        if (dist_type == "t_student" and
                (isinstance(self.calibrated_df, bool)
                 or not isinstance(self.calibrated_df, Real)
                 or not np.isfinite(self.calibrated_df)
                 or self.calibrated_df <= 2)):
            raise ValueError("t_student requires calibrated degrees of freedom above 2")

        # Preserve the normal mode's arithmetic order for existing replays.
        # For extreme annual parameters, sigma**2 can overflow even when
        # sigma**2 * dt is representable. Retry using the per-step scale.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scale = float(self.sigma * np.sqrt(self.dt))
            try:
                drift = (self.mu - 0.5 * self.sigma**2) * self.dt
            except OverflowError:
                drift = float("nan")
            if not np.isfinite(drift):
                drift = float(self.mu * self.dt - 0.5 * np.float64(scale) ** 2)
            if dist_type == "t_student":
                df = self.calibrated_df
                assert df is not None
                # Form the bounded variance factor first: dt * (df - 2)
                # may overflow even though the final t scale is finite.
                scale *= float(np.sqrt((df - 2) / df))
        if not np.isfinite(drift) or not np.isfinite(scale):
            raise ValueError("simulation increments exceed finite precision")

        # NumPy's t sampler returns a separate full shock matrix; the normal
        # sampler writes directly into the output. Check before either allocates.
        self._check_memory(temporary_arrays=int(dist_type == "t_student"))

        paths = np.empty((self.steps + 1, self.n_sims), dtype=np.float64)
        paths[0].fill(self.s0)
        increments = paths[1:]
        if pairs:
            # Generator.standard_normal needs a contiguous ``out``; each row
            # prefix is contiguous and rows are filled in C order.
            for row in increments:
                self.rng.standard_normal(size=pairs, out=row[:pairs])
            np.negative(increments[:, :pairs], out=increments[:, pairs:])
        elif dist_type == "normal":
            self.rng.standard_normal(size=increments.shape, out=increments)
        else:
            df = self.calibrated_df
            assert df is not None
            increments[:] = self.rng.standard_t(df=df, size=increments.shape)

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            np.multiply(increments, scale, out=increments)
            increments += drift
            np.cumsum(increments, axis=0, out=increments)
            # Only move the initial price into log space when exp alone would
            # overflow or enter the subnormal range before multiplication.
            # Keep the usual arithmetic order for normal-sized replays.
            if (np.max(increments) >= np.log(np.finfo(np.float64).max)
                    or np.min(increments) <= np.log(np.finfo(np.float64).tiny)):
                increments += np.log(self.s0)
                np.exp(increments, out=increments)
            else:
                np.exp(increments, out=increments)
                increments *= self.s0
        # Scalar reductions avoid another full-size boolean mask after the
        # output has been filled. NaN and either infinity fail these checks.
        min_price, max_price = np.min(paths), np.max(paths)
        if (not np.isfinite(min_price) or not np.isfinite(max_price)
                or min_price <= 0):
            raise ValueError("simulated prices exceed finite positive precision")
        return paths

    def generate_terminal(self, *, batch_size: int = 10_000,
                          max_working_bytes: int = 256 * 1024 * 1024,
                          variance_reduction: VarianceReduction = "none") -> np.ndarray:
        """Return normal GBM terminal prices as a ``float64 (n_sims,)`` array.

        Draw one independent standard normal Z per simulation and evaluate
        S(T) = s0 * exp((mu - sigma**2/2)*T + sigma*sqrt(T)*Z).
        Prices use the asset currency, T years, mu year^-1 and sigma
        year^-1/2. The instance's ``steps`` is unused: this exact terminal
        law needs no intermediate grid. This method does not simulate the
        discrete Student t scenario.

        Each batch is a view into the output, filled and transformed in
        place. The array budget is exactly ``8 * n_sims`` bytes, checked
        before allocation or RNG sampling; batching does not remove the
        returned vector's cost. The limit excludes Python/NumPy overhead
        and caller-held arrays and is not a process RSS limit.

        Calls advance the instance's local RNG. Reproduction requires the
        same initial RNG state, parameters, batch size and NumPy version.
        No elementwise identity is promised with full paths, other sampling
        formats or other NumPy versions. Prices outside finite positive
        float64 precision raise ValueError, including underflow to zero.

        ``variance_reduction="antithetic"`` requires an even ``n_sims = 2m``
        and returns ``[originals..., mirrors...]``: ``m`` normals are drawn
        in batches of ``batch_size`` into ``terminal[:m]`` and negated into
        ``terminal[m:]`` before the transform, within the same
        ``8 * n_sims``-byte budget. Element ``i`` pairs with ``i + m``.
        """
        for name, value in (("batch_size", batch_size),
                            ("max_working_bytes", max_working_bytes)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mu is None or self.sigma is None:
            raise ValueError("calibrate or supply mu and sigma before generating terminal prices")
        pairs = self._antithetic_pairs(variance_reduction, "normal")
        required = 8 * self.n_sims
        if required > max_working_bytes:
            raise ValueError(
                f"terminal array requires {required} bytes; "
                f"limit is {max_working_bytes} bytes"
            )

        # Scale before squaring: the annual sigma**2 may overflow even when
        # sigma**2*T is finite. Halving first also avoids a spurious overflow
        # in a representable half-variance.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scale = float(self.sigma * np.sqrt(self.horizon))
            drift = self.mu * self.horizon - (0.5 * scale) * scale
        if not np.isfinite(drift) or not np.isfinite(scale):
            raise ValueError("terminal log returns exceed finite precision")

        terminal = np.empty(self.n_sims, dtype=np.float64)
        log_max = np.log(np.finfo(np.float64).max)
        log_tiny = np.log(np.finfo(np.float64).tiny)

        def transform(batch: np.ndarray) -> None:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                np.multiply(batch, scale, out=batch)
                batch += drift
                if np.max(batch) >= log_max or np.min(batch) <= log_tiny:
                    batch += np.log(self.s0)
                    np.exp(batch, out=batch)
                else:
                    np.exp(batch, out=batch)
                    batch *= self.s0
            # Scalar reductions avoid an extra n_sims-sized boolean mask.
            low, high = np.min(batch), np.max(batch)
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0:
                raise ValueError("terminal prices exceed finite positive precision")

        if pairs:
            for start in range(0, pairs, int(batch_size)):
                batch = terminal[start:min(start + int(batch_size), pairs)]
                mirror = terminal[pairs + start:pairs + start + batch.size]
                self.rng.standard_normal(size=batch.shape, out=batch)
                np.negative(batch, out=mirror)
                transform(batch)
                transform(mirror)
            return terminal
        for start in range(0, self.n_sims, int(batch_size)):
            batch = terminal[start:start + int(batch_size)]
            self.rng.standard_normal(size=batch.shape, out=batch)
            transform(batch)
        return terminal


def simulate_gbm(s0: float, mu: float, sigma: float, horizon: float,
                 steps: int, n_sims: int, seed: int | None = None, *,
                 variance_reduction: VarianceReduction = "none") -> np.ndarray:
    """Compatibility entry point for the existing normal GBM pipeline."""
    return GbmSimulator(s0, horizon, steps, n_sims, seed, mu=mu,
                        sigma=sigma).generate_paths(
                            variance_reduction=variance_reduction)


def simulate_terminal_batched(s0: float, mu: float, sigma: float, horizon: float,
                              n_sims: int, seed: int | None = None, *,
                              batch_size: int = 10_000,
                              max_working_bytes: int = 256 * 1024 * 1024,
                              variance_reduction: VarianceReduction = "none") -> np.ndarray:
    """Sample exact normal GBM terminal prices with a local seeded RNG.

    See ``GbmSimulator.generate_terminal`` for units, memory accounting and
    reproduction guarantees. No path grid is required. To use the existing
    ``risk_summary`` contract, callers can construct initial/terminal rows;
    those additional arrays are outside this function's memory budget.
    """
    return GbmSimulator(s0, horizon, 1, n_sims, seed, mu=mu, sigma=sigma).generate_terminal(
        batch_size=batch_size, max_working_bytes=max_working_bytes,
        variance_reduction=variance_reduction,
    )
