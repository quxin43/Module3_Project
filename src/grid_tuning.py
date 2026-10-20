"""Rerun tuning rule (frozen in docs/rerun_recipe.md): fixed grid + neighbour smoothing.

For each model and each predicted year:
1. Score every setting in a fixed grid on the same 5 practice blocks as before
   (mean macro-F1; the train-part score is recorded for display only).
2. Smooth: each setting's score = mean of its own score and the scores of its
   grid neighbours one step away on each numeric setting (categorical settings
   are never averaged across).
3. Winner = highest smoothed score; ties (to 3 decimals) go to the simpler setting.
4. Refit the winner on the whole training window and predict the year.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from .labels import CLASS_ORDER
from .models import MODEL_SPECS
from .splits import inner_splits


def _log(a, b, n):
    return [float(round(v, 8)) for v in np.logspace(np.log10(a), np.log10(b), n)]


LR7 = [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2]

# axes: ordered dict of setting -> values (order = grid order, used for neighbours)
# numeric: settings that are smoothed across; the rest are categorical
# fixed: settings held constant (not tuned)
# simple: key function, smaller = simpler (tie-break)
GRIDS = {
    "Dummy": dict(axes={"strategy": ["most_frequent", "stratified"]}, numeric=[], fixed={},
                  simple=lambda p: (p["strategy"] != "most_frequent",)),
    "K-nearest neighbours": dict(axes={"n_neighbors": list(range(5, 201, 5)), "weights": ["uniform", "distance"]},
                                 numeric=["n_neighbors"], fixed={},
                                 simple=lambda p: (-p["n_neighbors"], p["weights"] != "uniform")),
    "Logistic regression": dict(axes={"C": _log(1e-3, 10, 17), "l1_ratio": [0.0, 1.0]}, numeric=["C"], fixed={},
                                simple=lambda p: (p["C"], -p["l1_ratio"])),
    "Decision tree": dict(axes={"max_depth": list(range(2, 9)), "min_samples_leaf": [20, 40, 60, 80, 100, 125, 150, 175, 200]},
                          numeric=["max_depth", "min_samples_leaf"], fixed={},
                          simple=lambda p: (p["max_depth"], -p["min_samples_leaf"])),
    "Random forest": dict(axes={"max_depth": list(range(3, 11)), "min_samples_leaf": [20, 50, 100, 150, 200],
                                "max_features": ["sqrt", 0.5]},
                          numeric=["max_depth", "min_samples_leaf"], fixed={"n_estimators": 300},
                          simple=lambda p: (p["max_depth"], -p["min_samples_leaf"], p["max_features"] != "sqrt")),
    "HistGradientBoosting": dict(axes={"learning_rate": LR7, "max_iter": [100, 200, 300, 400, 500],
                                       "l2_regularization": [0.0, 0.5, 1.0]},
                                 numeric=["learning_rate", "max_iter", "l2_regularization"], fixed={"max_leaf_nodes": 31},
                                 simple=lambda p: (p["max_iter"], p["learning_rate"], -p["l2_regularization"])),
    "XGBoost": dict(axes={"learning_rate": LR7, "n_estimators": [100, 200, 300, 400, 500],
                          "colsample_bytree": [0.6, 0.8, 1.0]},
                    numeric=["learning_rate", "n_estimators", "colsample_bytree"], fixed={"max_depth": 6, "subsample": 1.0},
                    simple=lambda p: (p["n_estimators"], p["learning_rate"], p["colsample_bytree"])),
    "Perceptron": dict(axes={"alpha": _log(1e-5, 1e-2, 13), "penalty": ["l2", "l1", "elasticnet"]}, numeric=["alpha"], fixed={},
                       simple=lambda p: (-p["alpha"], ["l2", "l1", "elasticnet"].index(p["penalty"]))),
    "MLP": dict(axes={"learning_rate_init": _log(1e-4, 1e-2, 9), "hidden_layer_sizes": [(16,), (32,), (32, 16)]},
                numeric=["learning_rate_init"], fixed={"alpha": 1e-4},
                simple=lambda p: ([(16,), (32,), (32, 16)].index(tuple(p["hidden_layer_sizes"])), p["learning_rate_init"])),
}


def grid_points(name):
    g = GRIDS[name]
    keys = list(g["axes"])
    return keys, [dict(zip(keys, combo)) for combo in itertools.product(*g["axes"].values())]


def base_pipeline(name):
    pipe = clone(MODEL_SPECS[name][0])
    if GRIDS[name]["fixed"]:
        pipe.set_params(**{f"model__{k}": v for k, v in GRIDS[name]["fixed"].items()})
    return pipe


def smooth(name, keys, points, means):
    """Neighbour-averaged score for each grid point (same order as points)."""
    g = GRIDS[name]
    axes = g["axes"]
    pos = {tuple(axes[k].index(p[k]) for k in keys): i for i, p in enumerate(points)}
    out = []
    for i, p in enumerate(points):
        idx = [axes[k].index(p[k]) for k in keys]
        vals = [means[i]]
        for a, k in enumerate(keys):
            if k not in g["numeric"]:
                continue
            for d in (-1, 1):
                j = list(idx)
                j[a] += d
                if tuple(j) in pos:
                    vals.append(means[pos[tuple(j)]])
        out.append(float(np.mean(vals)))
    return out


def pick(name, points, smoothed):
    best = round(max(smoothed), 3)
    tied = [i for i, s in enumerate(smoothed) if round(s, 3) == best]
    return min(tied, key=lambda i: GRIDS[name]["simple"](points[i]))


def on_edge(name, p):
    axes, num = GRIDS[name]["axes"], GRIDS[name]["numeric"]
    return [k for k in num if p[k] in (axes[k][0], axes[k][-1])]


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


ROUNDS = {"HistGradientBoosting": "max_iter", "XGBoost": "n_estimators"}


def _staged_scores(name, Xtr, ytr, points):
    """Boosting shortcut: train once per (other settings) with the most rounds, score after each round count.

    A boosting model with k rounds is exactly the first k rounds of the same model trained longer
    (no early stopping, no row subsampling), so this gives the same scores as fitting each point.
    Returns arrays (test[n_points, 5], train[n_points, 5]) in the order of points.
    """
    from sklearn.metrics import f1_score
    rk = ROUNDS[name]
    rounds = GRIDS[name]["axes"][rk]
    kmax = max(rounds)
    test = np.full((len(points), 5), np.nan)
    train = np.full((len(points), 5), np.nan)
    others = {}
    for i, p in enumerate(points):
        others.setdefault(tuple((k, v) for k, v in p.items() if k != rk), {})[p[rk]] = i
    f1 = lambda a, b: f1_score(a, b, average="macro", labels=CLASS_ORDER, zero_division=0)
    for s, (tr, va) in enumerate(inner_splits(len(Xtr))):
        Xa, ya, Xb, yb = Xtr.iloc[tr], ytr.iloc[tr], Xtr.iloc[va], ytr.iloc[va]
        for combo, idx in others.items():
            est = base_pipeline(name).set_params(**{f"model__{k}": v for k, v in combo}, **{f"model__{rk}": kmax})
            est.fit(Xa, ya)
            m = est[-1]
            if name == "HistGradientBoosting":
                sb = {k + 1: pr for k, pr in enumerate(m.staged_predict(Xb)) if k + 1 in idx}
                sa = {k + 1: pr for k, pr in enumerate(m.staged_predict(Xa)) if k + 1 in idx}
            else:
                sb = {k: m.predict(Xb, iteration_range=(0, k)) for k in idx}
                sa = {k: m.predict(Xa, iteration_range=(0, k)) for k in idx}
            for k, i in idx.items():
                test[i, s] = f1(yb, sb[k])
                train[i, s] = f1(ya, sa[k])
    return test, train


def tune_one_year(name, X, y, fold, out_dir: Path, n_jobs=1):
    """Run the grid for one model and one predicted year; save JSON + probabilities. Skips if done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jf, pf = out_dir / f"{fold.year}.json", out_dir / f"{fold.year}_proba.parquet"
    if jf.exists() and pf.exists():
        return json.load(open(jf))
    t0 = time.time()
    Xtr, ytr = X.iloc[fold.train_idx], y.iloc[fold.train_idx].astype(int)
    keys, points = grid_points(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        if name in ROUNDS:
            test, train = _staged_scores(name, Xtr, ytr, points)
        else:
            grid = [{f"model__{k}": [v] for k, v in p.items()} for p in points]
            search = GridSearchCV(base_pipeline(name), grid, scoring="f1_macro", cv=inner_splits(len(Xtr)), refit=False,
                                  n_jobs=n_jobs, return_train_score=True, error_score="raise")
            search.fit(Xtr, ytr)
            cv = search.cv_results_
            # cv_results_ order follows the list of single-point grids = order of points
            test = np.array([[cv[f"split{k}_test_score"][i] for k in range(5)] for i in range(len(points))])
            train = np.array([[cv[f"split{k}_train_score"][i] for k in range(5)] for i in range(len(points))])
    means = [float(v) for v in test.mean(axis=1)]
    sm = smooth(name, keys, points, means)
    w = pick(name, points, sm)
    rows = [{"params": {k: _jsonable(v) for k, v in p.items()}, "mean": round(means[i], 5),
             "sd": round(float(test[i].std()), 5), "smoothed": round(sm[i], 5),
             "train_mean": round(float(train[i].mean()), 5),
             "folds": [round(float(v), 5) for v in test[i]]}
            for i, p in enumerate(points)]
    raw_best = int(np.argmax(means))
    # refit winner on the whole training window, predict the year
    est = base_pipeline(name).set_params(**{f"model__{k}": v for k, v in points[w].items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        est.fit(Xtr, ytr)
    Xte = X.iloc[fold.test_idx]
    p = est.predict_proba(Xte)
    classes = list(est[-1].classes_)
    p_full = np.zeros((len(Xte), len(CLASS_ORDER)))
    for j, c in enumerate(classes):
        p_full[:, CLASS_ORDER.index(c)] = p[:, j]
    df = pd.DataFrame(p_full, index=Xte.index, columns=["p_sell", "p_hold", "p_buy"])
    df["pred"] = np.array(CLASS_ORDER)[p_full.argmax(axis=1)]
    df["year"] = fold.year
    extra = {}
    m = est[-1]
    if name == "MLP":
        extra["mlp_curves"] = {"train_loss": [round(float(v), 5) for v in m.loss_curve_],
                               "val_accuracy": [round(float(v), 5) for v in (m.validation_scores_ or [])],
                               "best_epoch": int(np.argmax(m.validation_scores_)) + 1 if m.validation_scores_ else None,
                               "n_epochs": int(m.n_iter_)}
    res = {"model": name, "year": fold.year, "phase": fold.phase, "keys": keys, "numeric": GRIDS[name]["numeric"],
           "fixed": {k: _jsonable(v) for k, v in GRIDS[name]["fixed"].items()},
           "grid": rows, "winner": w, "raw_best": raw_best, "edge": on_edge(name, points[w]),
           "seconds": round(time.time() - t0, 1), **extra}
    df.to_parquet(pf)
    json.dump(res, open(jf, "w"))
    return res
