"""Buy / Hold / Sell target.

The label on day t looks at the NEXT 5 trading days:
    r_t = close[t+5] / close[t] - 1
    Buy  if r_t >  +threshold
    Sell if r_t <  -threshold
    Hold otherwise
This is the only place in the project that looks forward in time, and it is
only ever used as the answer key, never as an input feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5
THRESHOLD = 0.01

SELL, HOLD, BUY = 0, 1, 2
CLASS_NAMES = {SELL: "Sell", HOLD: "Hold", BUY: "Buy"}
CLASS_ORDER = [SELL, HOLD, BUY]


def forward_return(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Return over the next `horizon` trading rows (NaN for the last `horizon` rows)."""
    return close.shift(-horizon) / close - 1


def make_labels(close: pd.Series, horizon: int = HORIZON, threshold: float | pd.Series = THRESHOLD) -> pd.Series:
    """0 = Sell, 1 = Hold, 2 = Buy; NaN where the future is not known yet.

    `threshold` may be a number (fixed band) or a Series (e.g. volatility-scaled band).
    """
    fwd = forward_return(close, horizon)
    lab = pd.Series(np.where(fwd > threshold, BUY, np.where(fwd < -threshold, SELL, HOLD)), index=close.index, dtype=float)
    lab[fwd.isna()] = np.nan
    return lab.rename("label")


def vol_scaled_threshold(close: pd.Series, k: float = 0.4, horizon: int = HORIZON, window: int = 20) -> pd.Series:
    """Alternative band: k x recent daily volatility x sqrt(horizon) (uses past data only)."""
    return k * close.pct_change().rolling(window).std() * np.sqrt(horizon)
