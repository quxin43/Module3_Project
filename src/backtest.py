"""Turn predictions into trades and measure money, with vectorbt.

Rules (agreed in the project plan):
- Signal from probabilities: Sell if P(Sell) >= cut-off c; else Buy if P(Buy) is highest; else Hold.
  (cut-off None = plain "highest probability wins".)
- All-in / all-out: Buy -> 100% SPY, Sell -> 100% cash, Hold -> keep position.
- Signal known at day t's close, order filled at day t+1's open.
- Fee 0.1% + slippage 0.05% per trade. Cash earns 0%. Prices are adjusted, so dividends are included
  on days SPY is held.
- Benchmark: Buy & Hold over the same days, bought at the first day's open, same costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from .labels import BUY, HOLD, SELL

FEE = 0.001
SLIPPAGE = 0.0005
INIT_CASH = 10_000.0
CUTOFFS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def signals_from_proba(proba: pd.DataFrame, cutoff: float | None = None) -> pd.Series:
    """Daily signal (0 Sell, 1 Hold, 2 Buy) from the three class probabilities."""
    p = proba[["p_sell", "p_hold", "p_buy"]].to_numpy()
    if cutoff is None:
        sig = np.array([SELL, HOLD, BUY])[p.argmax(axis=1)]
    else:
        buy_top = (p[:, 2] >= p[:, 0]) & (p[:, 2] >= p[:, 1])
        sig = np.where(p[:, 0] >= cutoff, SELL, np.where(buy_top, BUY, HOLD))
    return pd.Series(sig, index=proba.index, name="signal")


def run_portfolio(prices: pd.DataFrame, signal: pd.Series, fee: float = FEE, slippage: float = SLIPPAGE,
                  init_cash: float = INIT_CASH) -> vbt.Portfolio:
    """Backtest one continuous account over the dates in `signal`.

    The signal on day t becomes an order at day t+1's open, so the account runs
    from the first signal date to one day past the last.
    """
    px = prices.loc[signal.index.min():].iloc[: len(signal) + 1]
    sig = signal.reindex(px.index)
    entries = (sig == BUY).shift(1, fill_value=False).astype(bool)
    exits = (sig == SELL).shift(1, fill_value=False).astype(bool)
    return vbt.Portfolio.from_signals(
        close=px["close"], entries=entries, exits=exits, price=px["open"],
        fees=fee, slippage=slippage, init_cash=init_cash, freq="1D",
    )


def run_buy_and_hold(prices: pd.DataFrame, index: pd.DatetimeIndex, fee: float = FEE, slippage: float = SLIPPAGE,
                     init_cash: float = INIT_CASH) -> vbt.Portfolio:
    """Buy & Hold over the same days as a model account.

    It buys at the open of the account's second day, the first day the model could
    trade too (the model's first signal is on day 1's close), and never sells.
    """
    px = prices.loc[index]
    entries = pd.Series(False, index=px.index)
    entries.iloc[1] = True
    return vbt.Portfolio.from_signals(
        close=px["close"], entries=entries, exits=~entries & False, price=px["open"],
        fees=fee, slippage=slippage, init_cash=init_cash, freq="1D",
    )


def costs_paid(pf: vbt.Portfolio, prices: pd.DataFrame) -> float:
    """Fees plus slippage paid, in $ (slippage = fill price vs the open)."""
    orders = pf.orders.records_readable
    if len(orders) == 0:
        return 0.0
    opens = prices["open"].reindex(pd.to_datetime(orders["Timestamp"])).to_numpy()
    slip = (orders["Size"].to_numpy() * np.abs(orders["Price"].to_numpy() - opens)).sum()
    return float(orders["Fees"].sum() + slip)


def summary(pf: vbt.Portfolio, prices: pd.DataFrame) -> dict:
    """Money measures for one account."""
    value = pf.value()
    rets = value.pct_change().dropna()
    years = (value.index[-1] - value.index[0]).days / 365.25
    total = value.iloc[-1] / value.iloc[0] - 1
    dd = value / value.cummax() - 1
    invested = pf.asset_value() > 0
    cost = costs_paid(pf, prices)
    start_value = float(value.iloc[0])
    return {
        "start value $": start_value,
        "end value $": float(value.iloc[-1]),
        "total return": total,
        "CAGR": (1 + total) ** (1 / years) - 1 if years > 0 else np.nan,
        "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan,
        "max drawdown": dd.min(),
        "trades": int(len(pf.orders.records)),
        "% time invested": float(invested.mean()),
        "costs paid $": cost,
        "costs % of start": cost / start_value,
    }


def period_table(model_value: pd.Series, bh_value: pd.Series, model_invested: pd.Series,
                 model_trades: pd.Series, freq: str = "YE") -> pd.DataFrame:
    """Return per period (year 'YE' or quarter 'QE') for model and Buy & Hold, with win / loss / tie.

    Tie = the model was fully invested for the whole period and made no trades in it,
    so it behaved exactly like Buy & Hold.
    """
    def period_returns(v):
        end = v.resample(freq).last()
        start = v.resample(freq).first()
        prev_end = end.shift(1)
        base = prev_end.fillna(start)  # first period starts from the starting value
        return end / base - 1

    m = period_returns(model_value)
    b = period_returns(bh_value)
    inv = model_invested.resample(freq).mean()
    trd = model_trades.resample(freq).sum()
    out = pd.DataFrame({"model return": m, "B&H return": b})
    out["excess return"] = out["model return"] - out["B&H return"]
    tie = (inv >= 0.999) & (trd == 0)
    out["result"] = np.where(tie, "tie", np.where(out["excess return"] > 0, "win", "loss"))
    out["% time invested"] = inv
    out["trades"] = trd.astype(int)
    return out


def daily_trades(pf: vbt.Portfolio) -> pd.Series:
    idx = pf.wrapper.index
    rec = pf.orders.records
    counts = pd.Series(0, index=idx)
    if len(rec):
        c = pd.Series(1, index=idx[rec["idx"]]).groupby(level=0).sum()
        counts.loc[c.index] = c.values
    return counts


def yearly_sharpe(value: pd.Series) -> pd.Series:
    r = value.pct_change().dropna()
    g = r.groupby(r.index.year)
    return (g.mean() / g.std() * np.sqrt(252)).rename("Sharpe")


def evaluate(prices: pd.DataFrame, proba: pd.DataFrame, cutoff: float | None, years=None,
             fee: float = FEE, slippage: float = SLIPPAGE) -> dict:
    """Backtest one model + signal rule against Buy & Hold over the same days.

    Returns the two vectorbt portfolios, a summary dict for each, and yearly and
    quarterly tables (restricted to `years` when given).
    """
    sig = signals_from_proba(proba, cutoff)
    pf = run_portfolio(prices, sig, fee, slippage)
    bh = run_buy_and_hold(prices, pf.wrapper.index, fee, slippage)
    mv, bv = pf.value(), bh.value()
    invested = (pf.asset_value() > 0).astype(float)
    trades = daily_trades(pf)
    yearly = period_table(mv, bv, invested, trades, "YE")
    yearly.index = pd.Index(yearly.index.year, name="year")
    yearly = yearly.join(yearly_sharpe(mv).rename("model Sharpe")).join(yearly_sharpe(bv).rename("B&H Sharpe"))
    quarterly = period_table(mv, bv, invested, trades, "QE")
    if years is not None:
        yearly = yearly[yearly.index.isin(list(years))]
        quarterly = quarterly[quarterly.index.year.isin(list(years))]
    s_m, s_b = summary(pf, prices), summary(bh, prices)
    return {
        "signal": sig, "pf": pf, "bh": bh, "model": s_m, "bh_summary": s_b,
        "yearly": yearly, "quarterly": quarterly,
        "excess total return": s_m["total return"] - s_b["total return"],
        "excess $": s_m["end value $"] - s_b["end value $"],
        "years won": int((yearly["result"] == "win").sum()), "years tied": int((yearly["result"] == "tie").sum()),
        "years": len(yearly),
        "quarters won": int((quarterly["result"] == "win").sum()), "quarters tied": int((quarterly["result"] == "tie").sum()),
        "quarters": len(quarterly),
        "median yearly Sharpe": float(yearly["model Sharpe"].median()),
        "worst year excess": float(yearly["excess return"].min()),
    }


def manual_backtest(prices: pd.DataFrame, signal: pd.Series, fee: float = FEE, slippage: float = SLIPPAGE,
                    init_cash: float = INIT_CASH) -> pd.Series:
    """Plain-pandas re-implementation of the same rules, used to cross-check vectorbt."""
    px = prices.loc[signal.index.min():].iloc[: len(signal) + 1]
    sig = signal.reindex(px.index).shift(1)  # act on yesterday's signal at today's open
    cash, shares, values = init_cash, 0.0, []
    for t, row in px.iterrows():
        s = sig.loc[t]
        if s == BUY and shares == 0:
            fill = row["open"] * (1 + slippage)
            shares = cash / (fill * (1 + fee))
            cash = 0.0
        elif s == SELL and shares > 0:
            fill = row["open"] * (1 - slippage)
            cash = shares * fill * (1 - fee)
            shares = 0.0
        values.append(cash + shares * row["close"])
    return pd.Series(values, index=px.index)
