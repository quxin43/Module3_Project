"""The 14 input features.

Every feature on day t uses only data up to and including day t's close
(backward-looking rolling windows and causal EWMs, never .shift(-k)).
tests/test_no_lookahead.py checks this by recomputing on truncated data.

Windows are the standard published defaults and are NOT tuned:
RSI 14 (Wilder), MACD 12/26/9 (Appel), Bollinger 20 / 2 SD, SMA 20/50/200, ATR 14.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "close_sma20", "close_sma50", "sma50_sma200",
    "rsi_14", "macd_hist",
    "bb_pctb", "bb_width",
    "vol_20d", "atr_14",
    "volume_ratio",
]

FEATURE_INFO = {
    "ret_1d": ("Return 1d", "Close / close 1 day ago - 1", "1 day"),
    "ret_5d": ("Return 5d", "Close / close 5 days ago - 1", "1 week"),
    "ret_10d": ("Return 10d", "Close / close 10 days ago - 1", "2 weeks"),
    "ret_20d": ("Return 20d", "Close / close 20 days ago - 1", "1 month"),
    "close_sma20": ("Close / SMA20", "Close / average of last 20 closes", "1 month"),
    "close_sma50": ("Close / SMA50", "Close / average of last 50 closes", "~2.5 months"),
    "sma50_sma200": ("SMA50 / SMA200", "50-day average / 200-day average", "~10 months"),
    "rsi_14": ("RSI(14)", "Wilder RSI, 0-100", "3 weeks"),
    "macd_hist": ("MACD histogram / price", "(EMA12 - EMA26 - signal EMA9) / close", "~1.5 months"),
    "bb_pctb": ("Bollinger %B", "(close - lower) / (upper - lower), SMA20 +/- 2 SD", "1 month"),
    "bb_width": ("Bollinger width", "(upper - lower) / SMA20", "1 month"),
    "vol_20d": ("Historical volatility", "Std of daily returns over 20 days x sqrt(252)", "1 month"),
    "atr_14": ("ATR(14) / price", "Wilder average true range / close", "3 weeks"),
    "volume_ratio": ("Volume ratio", "Volume / 20-day average volume", "1 month"),
}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: smoothed average gain vs average loss, scaled 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram: (EMA_fast - EMA_slow) minus its own EMA_signal."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd - sig


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's average true range (daily range including overnight gaps)."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of the 14 features, same index as df (NaN during warm-up)."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)

    for k in (1, 5, 10, 20):
        out[f"ret_{k}d"] = c / c.shift(k) - 1

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    out["close_sma20"] = c / sma20
    out["close_sma50"] = c / sma50
    out["sma50_sma200"] = sma50 / sma200

    out["rsi_14"] = rsi(c, 14)
    out["macd_hist"] = macd_hist(c) / c

    sd20 = c.rolling(20).std(ddof=0)
    upper, lower = sma20 + 2 * sd20, sma20 - 2 * sd20
    out["bb_pctb"] = (c - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / sma20

    daily_ret = c.pct_change()
    out["vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)
    out["atr_14"] = atr(h, l, c, 14) / c

    out["volume_ratio"] = v / v.rolling(20).mean()
    return out[FEATURES]
