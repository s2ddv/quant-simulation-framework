# S&P 500 (^GSPC) daily closes, 1990-01-02 to the last session retrieved

- Source: Yahoo Finance via `yfinance`, ticker `^GSPC`, interval 1d, `auto_adjust=False`, field `Close`.
- `^GSPC` is a **price index** (no dividends): `Close` equals `Adj Close`. The snapshot's `price_field: "adjusted_close"` is the project's fixed label and does not mean dividend adjustment. Returns here are price returns, not total returns (unlike the earlier SPY ETF snapshot).
- Missing values: NaN rows dropped (none expected); no interpolation; no calendar gap above 7 days.
- Integrity: `sha256` over the observations; verified by `forecasting.report._session_prices`. Not a point-in-time authenticated database; Yahoo may revise history.
- Created by `research/tail-risk/fetch_sp500.py` (the only step that needs the network).
