"""The 9 classification models, their search spaces, and the walk-forward runner.

Every model is a Pipeline, so scaling and oversampling are fitted on the
training rows of each (inner or outer) fold only.
Tuning: RandomizedSearchCV over the 5 inner folds, scored by macro-F1.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .labels import CLASS_ORDER
from .splits import Fold, inner_splits

SEED = 42


class BalancedXGB(XGBClassifier):
    """XGBoost that weights classes inversely to their frequency, computed inside fit.

    Computing the weights inside fit means they use only the rows of the fold
    being trained, never the validation rows.
    """

    def fit(self, X, y, **kwargs):
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kwargs)


class ProbaPerceptron(Perceptron):
    """Perceptron with a predict_proba made from a softmax of its decision scores.

    These are scores squeezed to sum to 1, not calibrated probabilities.
    """

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def _scaled(clf):
    return Pipeline([("scale", StandardScaler()), ("model", clf)])


def _scaled_oversampled(clf):
    return ImbPipeline([("scale", StandardScaler()), ("oversample", RandomOverSampler(random_state=SEED)), ("model", clf)])


# name -> (pipeline, search space, n_iter or None for full grid, scaling, imbalance handling)
MODEL_SPECS = {
    "Dummy": (
        Pipeline([("model", DummyClassifier(random_state=SEED))]),
        {"model__strategy": ["most_frequent", "stratified"]},
        None, "No", "n/a",
    ),
    "K-nearest neighbours": (
        _scaled_oversampled(KNeighborsClassifier()),
        {"model__n_neighbors": randint(5, 201), "model__weights": ["uniform", "distance"]},
        20, "Yes", "oversample training folds",
    ),
    "Logistic regression": (
        _scaled(LogisticRegression(class_weight="balanced", solver="saga", max_iter=3000)),
        # l1_ratio 0 = L2 penalty, 1 = L1 penalty (scikit-learn 1.8 spelling of `penalty`)
        {"model__C": loguniform(1e-3, 10), "model__l1_ratio": [0.0, 1.0]},
        20, "Yes", "class_weight",
    ),
    "Decision tree": (
        Pipeline([("model", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
        {"model__max_depth": randint(2, 9), "model__min_samples_leaf": randint(20, 201)},
        20, "No", "class_weight",
    ),
    "Random forest": (
        Pipeline([("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED))]),
        {"model__n_estimators": [100, 200, 300], "model__max_depth": randint(3, 11),
         "model__min_samples_leaf": randint(20, 201), "model__max_features": ["sqrt", 0.5]},
        12, "No", "class_weight",
    ),
    "HistGradientBoosting": (
        Pipeline([("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]),
        {"model__learning_rate": loguniform(0.01, 0.2), "model__max_iter": randint(100, 501),
         "model__max_leaf_nodes": randint(8, 32), "model__l2_regularization": uniform(0, 1)},
        12, "No", "class_weight",
    ),
    "XGBoost": (
        Pipeline([("model", BalancedXGB(n_jobs=2, random_state=SEED, tree_method="hist", eval_metric="mlogloss"))]),
        {"model__n_estimators": randint(100, 501), "model__learning_rate": loguniform(0.01, 0.2),
         "model__max_depth": randint(2, 7), "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.6, 0.4)},
        12, "No", "balanced sample weights",
    ),
    "Perceptron": (
        _scaled(ProbaPerceptron(class_weight="balanced", random_state=SEED, max_iter=1000)),
        {"model__alpha": loguniform(1e-5, 1e-2), "model__penalty": ["l2", "l1", "elasticnet"]},
        20, "Yes", "class_weight",
    ),
    "MLP": (
        _scaled_oversampled(MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED)),
        {"model__hidden_layer_sizes": [(16,), (32,), (32, 16)], "model__alpha": loguniform(1e-4, 1e-1),
         "model__learning_rate_init": loguniform(1e-4, 1e-2)},
        12, "Yes", "oversample training folds",
    ),
}
MODEL_NAMES = list(MODEL_SPECS)


def make_search(name: str, n_train_rows: int):
    pipe, space, n_iter, *_ = MODEL_SPECS[name]
    cv = inner_splits(n_train_rows)
    common = dict(scoring="f1_macro", cv=cv, refit=True, n_jobs=1, error_score="raise")
    if n_iter is None:
        return GridSearchCV(pipe, space, **common)
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, random_state=SEED, **common)


def _clean_params(params: dict) -> dict:
    return {k.replace("model__", ""): (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}


def run_walk_forward(name: str, X: pd.DataFrame, y: pd.Series, folds: list[Fold], verbose: bool = True):
    """Tune + fit one model in every fold; return out-of-sample probabilities and a tuning log.

    Returns
    -------
    proba : DataFrame indexed by date, columns Sell/Hold/Buy probabilities, plus 'pred' (argmax class) and 'year'
    log   : DataFrame, one row per fold: best settings, inner macro-F1 mean and SD, time
    """
    probas, logs = [], []
    for f in folds:
        t0 = time.time()
        Xtr, ytr = X.iloc[f.train_idx], y.iloc[f.train_idx].astype(int)
        Xte = X.iloc[f.test_idx]
        search = make_search(name, len(Xtr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            search.fit(Xtr, ytr)
        best = search.best_estimator_
        classes = list(best.classes_) if hasattr(best, "classes_") else list(best[-1].classes_)
        p = best.predict_proba(Xte)
        p_full = np.zeros((len(Xte), len(CLASS_ORDER)))
        for j, c in enumerate(classes):
            p_full[:, CLASS_ORDER.index(c)] = p[:, j]
        df = pd.DataFrame(p_full, index=Xte.index, columns=["p_sell", "p_hold", "p_buy"])
        df["pred"] = np.array(CLASS_ORDER)[p_full.argmax(axis=1)]
        df["year"] = f.year
        probas.append(df)
        i = search.best_index_
        cvres = search.cv_results_
        logs.append({
            "model": name, "fold": f.number, "year": f.year, "phase": f.phase,
            "inner_f1_mean": cvres["mean_test_score"][i], "inner_f1_sd": cvres["std_test_score"][i],
            "best_params": _clean_params(search.best_params_), "seconds": round(time.time() - t0, 1),
        })
        if verbose:
            print(f"  {name:<22} {f.year}  inner F1 {logs[-1]['inner_f1_mean']:.3f} ± {logs[-1]['inner_f1_sd']:.3f}  ({logs[-1]['seconds']}s)")
    return pd.concat(probas), pd.DataFrame(logs)


def always_buy_proba(index: pd.Index, years) -> pd.DataFrame:
    df = pd.DataFrame({"p_sell": 0.0, "p_hold": 0.0, "p_buy": 1.0}, index=index)
    df["pred"] = 2
    df["year"] = years
    return df


def block_bootstrap_ci(y_true, y_pred, metric, block: int = 20, n_boot: int = 1000, seed: int = SEED, level: float = 0.95):
    """Confidence interval for a score by resampling blocks of consecutive days.

    Blocks (not single days) are resampled because neighbouring days share
    overlapping 5-day labels, so single-day resampling would overstate precision.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        s = rng.choice(starts, size=k)
        idx = (s[:, None] + np.arange(block)).ravel()[:n]
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)


def search_space_text(name: str) -> dict:
    """Human-readable search space for the notebook tables."""
    _, space, n_iter, scaling, imbalance = MODEL_SPECS[name]
    parts = []
    for k, v in space.items():
        k = k.replace("model__", "")
        if isinstance(v, list):
            parts.append(f"{k}: {', '.join(map(str, v))}")
        elif hasattr(v, "dist") and v.dist.name == "loguniform":
            a, b = v.support()
            parts.append(f"{k}: {a:g} to {b:g} (log scale)")
        elif hasattr(v, "dist") and v.dist.name == "randint":
            a, b = v.support()
            parts.append(f"{k}: {int(a)} to {int(b)}")
        elif hasattr(v, "dist") and v.dist.name == "uniform":
            a, b = v.support()
            parts.append(f"{k}: {a:g} to {b:g}")
        else:
            parts.append(f"{k}: {v}")
    n = n_iter if n_iter else int(np.prod([len(v) for v in space.values()]))
    return {"model": name, "scaling": scaling, "class imbalance": imbalance,
            "settings searched": "; ".join(parts), "candidates per fold": n}


def classification_scores(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
