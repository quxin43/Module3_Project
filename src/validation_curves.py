"""Validation curves: how the practice-year score changes when ONE setting moves.

For each model and each predicted year, start from that year's winning settings,
then vary one setting at a time across a dense range (the others stay at the
winner's values) and score every value on the same 5 practice blocks used for
tuning. This uses the training window only - never the predicted year.

Logistic regression, KNN and the perceptron also draw one line per value of
their categorical setting (penalty / weights), because those change the shape
of the whole curve.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV

from .models import MODEL_SPECS
from .splits import inner_splits

SWEEPS = {
    "K-nearest neighbours": {"lines": ("weights", ["uniform", "distance"]),
                             "x": {"n_neighbors": [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 175, 200]}},
    "Logistic regression": {"lines": ("l1_ratio", [0.0, 1.0]),
                            "x": {"C": [float(c) for c in np.logspace(-3, 1, 25)]}},
    "Decision tree": {"x": {"max_depth": [2, 3, 4, 5, 6, 7, 8],
                            "min_samples_leaf": [20, 30, 40, 60, 80, 100, 125, 150, 175, 200]}},
    "Random forest": {"x": {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10], "min_samples_leaf": [20, 40, 60, 80, 120, 160, 200],
                            "n_estimators": [100, 200, 300], "max_features": ["sqrt", 0.5]}},
    "HistGradientBoosting": {"x": {"learning_rate": [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2], "max_iter": [100, 200, 300, 400, 500],
                                   "max_leaf_nodes": [8, 12, 16, 20, 26, 31], "l2_regularization": [0.0, 0.25, 0.5, 0.75, 1.0]}},
    "XGBoost": {"x": {"learning_rate": [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2], "max_depth": [2, 3, 4, 5, 6],
                      "n_estimators": [100, 200, 300, 400, 500], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                      "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0]}},
    "Perceptron": {"lines": ("penalty", ["l2", "l1", "elasticnet"]),
                   "x": {"alpha": [float(a) for a in np.logspace(-5, -2, 13)]}},
    "MLP": {"x": {"hidden_layer_sizes": [(16,), (32,), (32, 16)], "alpha": [float(a) for a in np.logspace(-4, -1, 10)],
                  "learning_rate_init": [float(a) for a in np.logspace(-4, -2, 10)]}},
}


def _fix(v):
    return tuple(v) if isinstance(v, list) else v


def curves_for_fold(name, X_train, y_train, winner_params, n_jobs=2):
    """Return {setting: [{"line": value_or_None, "x": value, "mean": ..., "sd": ..., "folds": [...]}, ...]}."""
    spec = SWEEPS[name]
    base = {f"model__{k}": _fix(v) for k, v in winner_params.items()}
    grid, tags = [], []
    line_key, line_vals = spec.get("lines", (None, [None]))
    for p, values in spec["x"].items():
        for lv in line_vals:
            for v in values:
                g = dict(base)
                g[f"model__{p}"] = v
                if line_key is not None:
                    g[f"model__{line_key}"] = lv
                grid.append({k: [val] for k, val in g.items()})
                tags.append((p, lv, v))
    search = GridSearchCV(MODEL_SPECS[name][0], grid, scoring="f1_macro", cv=inner_splits(len(X_train)),
                          refit=False, n_jobs=n_jobs, error_score=np.nan)
    search.fit(X_train, y_train)
    cv = search.cv_results_
    out = {}
    for i, (p, lv, v) in enumerate(tags):
        out.setdefault(p, []).append({
            "line": lv, "x": list(v) if isinstance(v, tuple) else v,
            "mean": round(float(cv["mean_test_score"][i]), 4), "sd": round(float(cv["std_test_score"][i]), 4),
            "folds": [round(float(cv[f"split{k}_test_score"][i]), 4) for k in range(5)],
        })
    return out
