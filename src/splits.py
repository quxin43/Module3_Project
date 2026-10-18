"""Walk-forward folds by calendar year, plus the inner folds used for tuning.

Outer fold for year Y:
    train   = every row dated before Y, minus the last GAP rows
              (their 5-day labels reach into year Y)
    predict = every row dated in Y
Inner folds (tuning, inside the training rows only):
    TimeSeriesSplit(n_splits=5, test_size=252, gap=5)
    = the 5 most recent ~1-year blocks, each trained on everything before it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

GAP = 5
FIRST_VALID_YEAR = 2005
FIRST_TEST_YEAR = 2021
INNER = TimeSeriesSplit(n_splits=5, test_size=252, gap=GAP)


@dataclass
class Fold:
    number: int
    year: int
    phase: str  # "validate" or "test"
    train_idx: np.ndarray  # integer positions into the modelling table
    test_idx: np.ndarray

    def describe(self, dates: pd.DatetimeIndex) -> dict:
        tr, te = dates[self.train_idx], dates[self.test_idx]
        return {
            "fold": self.number, "phase": self.phase, "predict year": self.year,
            "train from": tr.min().date(), "train to": tr.max().date(), "train rows": len(tr),
            "predict from": te.min().date(), "predict to": te.max().date(), "predict rows": len(te),
        }


def walk_forward_folds(dates: pd.DatetimeIndex, first_year: int = FIRST_VALID_YEAR,
                       first_test_year: int = FIRST_TEST_YEAR, gap: int = GAP) -> list[Fold]:
    """One fold per calendar year from `first_year` to the last year in `dates`."""
    years = dates.year
    pos = np.arange(len(dates))
    folds = []
    for i, y in enumerate(range(first_year, int(years.max()) + 1), start=1):
        before = pos[years < y]
        train_idx = before[: len(before) - gap] if gap else before
        test_idx = pos[years == y]
        if len(test_idx) == 0:
            continue
        phase = "test" if y >= first_test_year else "validate"
        folds.append(Fold(i, y, phase, train_idx, test_idx))
    return folds


def inner_splits(n_train_rows: int):
    """The 5 inner (practice-exam) splits for a training window of n rows."""
    return list(INNER.split(np.arange(n_train_rows)))
