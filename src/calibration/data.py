"""Validated adjusted daily prices from yfinance."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import yfinance as yf

from calibration.prices import validate_prices


def download_prices(ticker: str, period: str = "3y") -> pd.Series:
    """Return one chronological adjusted-close series for a single ticker."""
    data = yf.download(ticker, period=period, auto_adjust=True,
                       progress=False, group_by="column")
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Unexpected price response for {ticker}: expected a table.")
    if data.empty:
        raise ValueError(f"No price history returned for {ticker} ({period}).")
    columns = data.columns
    if isinstance(columns, pd.MultiIndex):
        if columns.nlevels != 2:
            raise ValueError(f"Unexpected price columns for {ticker}: require two levels.")
        matches = [column for column in columns if column[0] == "Close"]
        if len(matches) == 1 and matches[0][1] != ticker:
            matches = []
    else:
        matches = [column for column in columns if column == "Close"]
    if len(matches) != 1 or columns.tolist().count(matches[0]) != 1:
        raise ValueError(f"Unexpected price columns for {ticker}: require one adjusted Close series.")
    prices = data[matches[0]]
    if not isinstance(prices, pd.Series):
        raise ValueError(f"Unexpected price shape for {ticker}: require one series.")
    try:
        return validate_prices(prices, min_observations=3, require_dates=True)
    except ValueError as exc:
        raise ValueError(f"Invalid price history for {ticker}: {exc}") from exc


def download_prices_multi(tickers: Sequence[str], period: str = "3y") -> dict[str, pd.Series]:
    """Return validated adjusted closes for several tickers, one request each.

    Each ticker goes through ``download_prices`` (same column checks and
    ``validate_prices``), so every series keeps its own dates: no alignment,
    forward-fill or dropping happens here. Align the series afterwards with
    ``calibration.multi_asset.align_prices`` (inner join on common dates).
    Network access is required; offline tests must monkeypatch ``yf.download``.

    Raises:
        ValueError: If ``tickers`` is empty, has a non-string or empty entry,
            repeats a ticker, or any single download is invalid.
    """
    names = _validate_tickers(tickers)
    return {ticker: download_prices(ticker, period=period) for ticker in names}


def multi_asset_snapshot(prices_by_ticker: Mapping[str, pd.Series], *, period: str,
                         retrieved_at: Any, source: str = "yfinance") -> dict[str, Any]:
    """Snapshot several price series, one hashed single-asset snapshot each.

    Each entry of ``assets`` is exactly ``simulation.report.historical_snapshot``
    for that ticker (its own observations and SHA-256), so a per-asset
    snapshot can be read back with ``prices_from_snapshot``. Series are not
    aligned; ``tickers`` preserves the input order.
    """
    # Imported lazily: simulation.report imports calibration modules.
    from simulation.report import historical_snapshot

    if not isinstance(prices_by_ticker, Mapping):
        raise ValueError("prices_by_ticker must be a mapping of ticker to prices.")
    names = _validate_tickers(list(prices_by_ticker))
    assets = {ticker: historical_snapshot(prices_by_ticker[ticker], ticker=ticker,
                                          period=period, retrieved_at=retrieved_at,
                                          source=source)
              for ticker in names}
    return {"tickers": names, "assets": assets}


def prices_from_multi_asset_snapshot(snapshot: Mapping[str, Any]) -> dict[str, pd.Series]:
    """Rebuild each ticker's prices, verifying every per-asset hash and metadata."""
    from simulation.report import prices_from_snapshot

    try:
        names = _validate_tickers(snapshot["tickers"])
        assets = snapshot["assets"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid multi-asset snapshot: require tickers and assets.") from exc
    if not isinstance(assets, Mapping) or sorted(assets) != sorted(names):
        raise ValueError("Invalid multi-asset snapshot: assets must match tickers.")
    result = {}
    for ticker in names:
        if assets[ticker].get("ticker") != ticker:
            raise ValueError(f"Multi-asset snapshot entry {ticker!r} names another ticker.")
        result[ticker] = prices_from_snapshot(assets[ticker])
    return result


def _validate_tickers(tickers: Any) -> list[str]:
    if isinstance(tickers, str) or not isinstance(tickers, Sequence):
        raise ValueError("tickers must be a sequence of ticker strings.")
    names = list(tickers)
    if not names:
        raise ValueError("tickers must not be empty.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every ticker must be a nonempty string.")
    if len(set(names)) != len(names):
        raise ValueError("tickers must be unique.")
    return names
