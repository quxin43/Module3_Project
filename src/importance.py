"""Which indicators each model relied on.

For every model and every validation year 2005-2020:
- refit the model with that year's tuned settings on that fold's training rows
- permutation importance on that year's predictions (never on training rows):
  shuffle one feature -> drop in macro-F1; repeated and averaged
- grouped permutation importance: shuffle a whole family of related indicators
  together, which avoids the "correlated features share credit" problem
- built-in importance where the model has one (tree impurity, XGBoost gain,
  linear coefficients), from the model fitted for 2020
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .features import FEATURES
from .labels import CLASS_ORDER
from .models import make_search

GROUPS = {
    "Returns": ["ret_1d", "ret_5d", "ret_10d", "ret_20d"],
    "Trend (moving averages)": ["close_sma20", "close_sma50", "sma50_sma200"],
    "Momentum (RSI, MACD)": ["rsi_14", "macd_hist"],
    "Bollinger Bands": ["bb_pctb", "bb_width"],
    "Volatility (std, ATR)": ["vol_20d", "atr_14"],
    "Volume": ["volume_ratio"],
}


def _fit(name, params, Xtr, ytr):
    est = make_search(name, len(Xtr)).estimator
    est.set_params(**{f"model__{k}": (tuple(v) if isinstance(v, list) else v) for k, v in params.items()})
    return est.fit(Xtr, ytr)


def _score(est, X, y):
    return f1_score(y, est.predict(X), average="macro", labels=CLASS_ORDER, zero_division=0)


def permutation_by_year(name, X, y, folds, log, n_repeats=5, seed=0):
    """Return (single-feature DataFrame, group DataFrame), rows = years, values = drop in macro-F1."""
    rng = np.random.default_rng(seed)
    single, grouped, fitted = {}, {}, None
    for f in folds:
        params = log[(log.model == name) & (log.year == f.year)]["best_params"].iloc[0]
        est = _fit(name, params, X.iloc[f.train_idx], y.iloc[f.train_idx])
        Xte, yte = X.iloc[f.test_idx], y.iloc[f.test_idx]
        base = _score(est, Xte, yte)
        row = {}
        for col in FEATURES:
            drops = []
            for _ in range(n_repeats):
                Xp = Xte.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                drops.append(base - _score(est, Xp, yte))
            row[col] = float(np.mean(drops))
        single[f.year] = row
        grow = {}
        for g, cols in GROUPS.items():
            drops = []
            for _ in range(n_repeats):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].values[perm]  # shuffle the family together (keeps their mutual link)
                drops.append(base - _score(est, Xp, yte))
            grow[g] = float(np.mean(drops))
        grouped[f.year] = grow
        fitted = est
    return pd.DataFrame(single).T, pd.DataFrame(grouped).T, fitted


def builtin_importance(name, est):
    """Model's own importance, if it has one. Returns (kind, DataFrame) or (None, None)."""
    m = est[-1]
    if hasattr(m, "feature_importances_") and name != "Dummy":
        kind = "XGBoost gain share" if name == "XGBoost" else "impurity decrease share"
        return kind, pd.DataFrame({"importance": m.feature_importances_}, index=FEATURES)
    if hasattr(m, "coef_"):
        cols = {0: "Sell", 1: "Hold", 2: "Buy"}
        return "coefficient on standardised feature", pd.DataFrame(m.coef_.T, index=FEATURES, columns=[cols[c] for c in m.classes_])
    return None, None
