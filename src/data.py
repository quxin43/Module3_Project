"""Load SPY daily data and make adjusted OHLCV.

Source: Yahoo Finance via yfinance. The download is saved once as a frozen
snapshot (data/spy_daily.csv) and reused, so results are reproducible.

All prices are adjusted for splits and dividends: Open, High and Low are scaled
by the same factor as Adj Close / Close, so a dividend does not show up as a
fake price drop and every return is a total return.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "spy_daily.csv"
RAW_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_spy(start: str = "1993-01-01") -> pd.DataFrame:
    """Download raw (unadjusted) SPY OHLCV plus Adj Close from Yahoo Finance."""
    import yfinance as yf

    df = yf.download("SPY", start=start, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ConnectionError("yfinance returned no rows")
    return df[RAW_COLS]


def read_raw_csv(path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Read the CSV written by the download script (robust to yfinance header variants)."""
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)  # drops extra header rows such as "Ticker"
    df.index.name = "Date"
    df = df[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    return df


def adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted open/high/low/close and volume, lower-case columns, sorted by date."""
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    factor = raw["Adj Close"] / raw["Close"]
    out = pd.DataFrame(
        {
            "open": raw["Open"] * factor,
            "high": raw["High"] * factor,
            "low": raw["Low"] * factor,
            "close": raw["Adj Close"],
            "volume": raw["Volume"].astype(float),
        },
        index=raw.index,
    )
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def load_spy(prefer_download: bool = False, save_snapshot: bool = True) -> pd.DataFrame:
    """Adjusted SPY daily OHLCV.

    Uses the frozen snapshot data/spy_daily.csv when it exists, so every notebook
    and every cached result is built from the same rows. If the snapshot is missing
    (or prefer_download=True), downloads with yfinance and saves a new snapshot.
    """
    raw = None
    if prefer_download or not CSV_PATH.exists():
        try:
            raw = download_spy()
            if save_snapshot and not CSV_PATH.exists():
                CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                raw.to_csv(CSV_PATH)
        except Exception as exc:  # no internet, Yahoo blocked, API change
            print(f"yfinance download failed ({type(exc).__name__}); using {CSV_PATH.name}")
    if raw is None:
        if not CSV_PATH.exists() and os.environ.get("SPY_SYNTHETIC") == "1":
            print("WARNING: no SPY data found; using made-up practice prices (SPY_SYNTHETIC=1). Results are NOT real.")
            return synthetic_ohlcv()
        raw = read_raw_csv(CSV_PATH)
    return adjust(raw)


def data_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Simple quality checks; returns a one-column table of results."""
    rows_per_year = df.groupby(df.index.year).size()
    full_years = rows_per_year.iloc[1:-1]  # first and last year are partial
    checks = {
        "first date": df.index.min().date(),
        "last date": df.index.max().date(),
        "rows": len(df),
        "dates strictly increasing": bool(df.index.is_monotonic_increasing and df.index.is_unique),
        "missing values": int(df.isna().sum().sum()),
        "zero-volume days": int((df["volume"] <= 0).sum()),
        "high >= max(open, close) everywhere": bool((df["high"] + 1e-9 >= df[["open", "close"]].max(axis=1)).all()),
        "low <= min(open, close) everywhere": bool((df["low"] - 1e-9 <= df[["open", "close"]].min(axis=1)).all()),
        "rows per full year (min-max)": f"{full_years.min()}-{full_years.max()}",
    }
    return pd.DataFrame.from_dict(checks, orient="index", columns=["result"])


def synthetic_ohlcv(start: str = "1993-01-29", end: str = "2026-10-16", seed: int = 7) -> pd.DataFrame:
    """Made-up practice prices for testing the code when real data is not available.

    Random walk with a small upward drift and changing volatility. NOT real SPY data.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    vol = np.empty(n)
    v = 0.01
    for i in range(n):  # slowly changing daily volatility
        v = float(np.clip(v * np.exp(rng.normal(0, 0.05)), 0.004, 0.05))
        vol[i] = v
    ret = 0.0003 + vol * rng.standard_t(5, n) / np.sqrt(5 / 3)
    close = 45 * np.exp(np.cumsum(ret))
    gap = vol * rng.normal(0, 0.3, n)
    open_ = close / np.exp(ret) * np.exp(gap)
    span = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + span * 0.6
    low = np.minimum(open_, close) - span * 0.6
    volume = rng.lognormal(18, 0.3, n) * (1 + 20 * vol)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=pd.Index(idx, name="Date"))
