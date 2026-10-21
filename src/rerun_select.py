"""Rerun: collect the grid-tuning results, shortlist, choose the Sell cut-off with smoothing, pick the model.

Rules frozen in docs/rerun_recipe.md before any rerun result existed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CUTOFFS, evaluate
from .models import macro_f1

V2 = Path(__file__).resolve().parents[1] / "results" / "v2"


def slug(m):
    return m.lower().replace(" ", "_").replace("-", "_")


def load_tuning(name, years=None):
    d = V2 / "tune" / slug(name)
    out = {}
    for f in sorted(d.glob("[0-9][0-9][0-9][0-9].json")):
        r = json.load(open(f))
        if years is None or r["year"] in years:
            out[r["year"]] = r
    return out


def load_proba(name, years):
    d = V2 / "tune" / slug(name)
    return pd.concat([pd.read_parquet(d / f"{y}_proba.parquet") for y in years])


def rule_name(c):
    return "highest prob." if c is None else f"c = {c:.2f}"


def cutoff_table(prices, proba, years):
    """Backtest every rule for one model; add the smoothed excess return (cut-offs only)."""
    rows, runs = [], {}
    for c in [None] + CUTOFFS:
        r = evaluate(prices, proba, c, years=years)
        runs[c] = r
        rows.append({"rule": rule_name(c), "c": c, "excess return": r["excess total return"], "excess $": r["excess $"],
                     "years won": r["years won"], "years tied": r["years tied"],
                     "median yearly Sharpe": r["median yearly Sharpe"], "Sharpe": r["model"]["Sharpe"],
                     "worst year excess": r["worst year excess"], "max drawdown": r["model"]["max drawdown"],
                     "trades": r["model"]["trades"], "costs $": r["model"]["costs paid $"]})
    t = pd.DataFrame(rows)
    ex = t["excess return"].to_numpy()
    sm = []
    for i, c in enumerate(t["c"]):
        if c is None or (isinstance(c, float) and np.isnan(c)):
            sm.append(ex[i])
            continue
        j = [k for k in range(1, len(t)) if abs(t["c"].iloc[k] - c) <= 0.0501]
        sm.append(float(np.mean(ex[j])))
    t["smoothed excess return"] = sm
    return t, runs


def choose_rule(t):
    g = t.copy()
    g["_s"] = g["smoothed excess return"].round(6)
    g = g.sort_values(["_s", "Sharpe"], ascending=False)
    return g.iloc[0]["rule"]


def rank_models(score):
    s = score.copy()
    s["rank: excess return"] = s["excess return"].rank(ascending=False, method="min")
    s["rank: years won"] = s["years won"].rank(ascending=False, method="min")
    s["rank: median Sharpe"] = s["median yearly Sharpe"].rank(ascending=False, method="min")
    s["rank total"] = s[["rank: excess return", "rank: years won", "rank: median Sharpe"]].sum(axis=1)
    s["passes worst-year guard"] = s["worst year excess"] > -0.20
    pool = s[s["passes worst-year guard"]] if s["passes worst-year guard"].any() else s
    chosen = pool.sort_values(["rank total", "excess return"], ascending=[True, False]).index[0]
    s["picked"] = np.where(s.index == chosen, "✓", "")
    return s, chosen


def pooled_f1(proba, y):
    yy = y.loc[proba.index]
    return macro_f1(yy, proba["pred"])
