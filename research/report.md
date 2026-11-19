# Research report

## Status

This is a research outline, not an experimental results report. No backtest or
new simulation is claimed here.

## Current model

The implemented single-asset model estimates annualized drift and volatility
from adjusted daily closes, then simulates exact GBM increments. Terminal
statistics are compared with the analytical lognormal distribution. The CLI
reports probability of loss, VaR and CVaR for the selected horizon.
An optional chronological diagnostic now freezes a calibration fit from the
first `n` closes and scores nonoverlapping forecasts against later closes.
It is a historical comparison, separate from Monte Carlo convergence.

## Chronological diagnostic protocol

For origins `i=n-1+k*h`, predict the close at `i+h` with `T=h/252` years.
Use the analytical GBM mean `S_i exp(mu*T)` as the point forecast, median
`S_i exp((mu-sigma²/2)*T)`, and lower quantile
`Q=S_i exp((mu-sigma²/2)*T+sigma*sqrt(T)*Phi⁻¹(1-alpha))`.
VaR is `S_i-Q`; a strict violation is `S_realized<Q`. Record violation count,
observed rate and nominal reference `1-alpha`. For error
`e=S_realized-S_predicted`, report mean bias, mean `abs(e)`, root mean squared
error, and mean `100*abs(e)/S_i`. Prices, errors and VaR are in the asset's
price currency; the last measure is a percentage. Drift uses year⁻¹ and
volatility year⁻¹ᐟ². A zero-volatility point mass has zero strict model
violation probability, even though the report retains `1-alpha` as reference.

Example using a fresh data download (no market outcome is claimed here):

```bash
python main.py --ticker PETR4.SA --period 3y --out-of-sample --calibration-observations 504 --evaluation-horizon-days 21 --evaluation-confidence 0.95 --no-plot --report reports/petr4-oos.json
```

The saved historical snapshot allows offline replay. Few completed horizons
make the observed rate imprecise. Frozen historical parameters may become
unrepresentative after a regime change; they are neither risk-neutral drift nor
implied volatility. This diagnostic does not quantify calibration uncertainty,
model error or future coverage. The existing convergence chart instead measures
Monte Carlo sample error under fixed parameters.

### Moving calibration windows

`--calibration-mode fixed` (default) fits the first `n` closes once.
`rolling` refits on the most recent `n` closes available at each
recalibration origin; `expanding` refits on every close from the start through
that origin. `--recalibration-frequency k` refits every `k` forecast origins
and reuses the latest fit in between. No fit sees a close after its origin.
Moving modes record per-forecast parameters and calibration dates, which make
question 1 below measurable without new code.

### Calibration sensitivity

`--calibration-sensitivity` evaluates the closed-form GBM mean, quantiles and
VaR on a symmetric grid of drift shifts (`--mu-delta`, year⁻¹) and relative
volatility multipliers (`--sigma-relative-delta`). It uses no random draws.
Ranges over the chosen scenarios are not confidence intervals: they exclude
parameter sampling uncertainty, Monte Carlo error and model error.

## Volatility forecasting experiment

`src/forecasting/` (`python -m forecasting`) is separate from GBM calibration.
It forecasts the next session's second moment of the log return,
assuming zero conditional mean, and compares on shared origins:

- historical: trailing mean of squared returns over a window in
  `historical_windows`;
- EWMA: `v_t = lambda*v_{t-1} + (1-lambda)*r_t²`, `lambda` in `ewma_factors`;
- ML: nonnegative least squares of `r_{t+1}²` on trailing means over 1, 5 and
  22 sessions plus a nonnegative intercept, refit on an expanding sample every
  `refit_frequency` origins.

Periods are chronological: train, validation (selects the historical window
and EWMA factor by minimum MSE) and a reserved final test where all choices
are frozen. Risk diagnostics assume `r_next ~ Normal(0, v)`, giving
`q = S_origin*exp(sqrt(v)*Phi⁻¹(1-alpha))` and `VaR = S_origin - q` in price
currency. Metrics: MSE (log-return⁴), MAE (log-return²), price pinball loss,
strict violation count and rate with a 95% Wilson interval. The squared return
is a noisy proxy for latent variance, so low MSE differences may not be
meaningful; no model superiority is claimed here.

## Questions for future research

1. How stable are drift and volatility across calibration windows?
2. How well do predicted terminal quantiles and VaR cover held-out outcomes?
3. How sensitive are tail estimates to simulation count and model assumptions?
4. Does EWMA or NNLS improve out-of-sample one-session VaR coverage over a
   trailing historical window, beyond the noise of the squared-return proxy?

## Evidence needed

Use dated, frozen inputs and chronological train/test splits. Record data
source, observation dates, code revision, dependency versions, random seed,
metrics and uncertainty. Separate Monte Carlo sampling error from calibration
uncertainty and model error. Report results only after running and reviewing
the experiments.
