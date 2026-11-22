"""Derivative pricing under the risk-neutral measure Q.

Every pricing result must be reported together with ``pricing_metadata``:
the measure is always ``"risk_neutral"`` (drift ``r - q``), and the origin of
``sigma`` is either ``"historical_proxy"`` or ``"user_supplied"``. Pricing with
the historical price drift ``mu`` is not supported.
"""

from pricing.black_scholes import (
    bs_greeks,
    bs_price,
    implied_volatility,
    pricing_metadata,
)

__all__ = ["bs_greeks", "bs_price", "implied_volatility", "pricing_metadata"]
