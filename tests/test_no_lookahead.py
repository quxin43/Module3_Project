"""Look-ahead and leakage checks. Run with:  python -m pytest -q

1. Truncation test: every feature on day t must be identical whether it is computed
   on the full history or on data that stops at day t. If a feature peeked at the
   future, cutting the data off would change it.
2. Labels look exactly 5 rows ahead, and the last 5 rows have no label.
3. Walk-forward folds: training always ends before the predicted year, with a 5-row gap.
4. Inner folds: validation always comes after training, with a 5-row gap.
5. Backtest: trades happen the day AFTER the signal; vectorbt matches a plain re-implementation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import manual_backtest, run_portfolio  # noqa: E402
from src.data import CSV_PATH, adjust, read_raw_csv, synthetic_ohlcv  # noqa: E402
from src.features import FEATURES, build_features  # noqa: E402
from src.labels import BUY, SELL, forward_return, make_labels  # noqa: E402
from src.splits import GAP, inner_splits, walk_forward_folds  # noqa: E402


@pytest.fixture(scope="module")
def prices():
    if CSV_PATH.exists():
        return adjust(read_raw_csv(CSV_PATH))
    return synthetic_ohlcv()


def test_features_do_not_use_future_data(prices):
    full = build_features(prices)
    rng = np.random.default_rng(0)
    cut_points = sorted(rng.choice(np.arange(250, len(prices) - 10), size=25, replace=False))
    for cut in cut_points:
        truncated = build_features(prices.iloc[: cut + 1])
        a = full.iloc[cut]
        b = truncated.iloc[-1]
        assert np.allclose(a.values, b.values, rtol=1e-10, atol=1e-12, equal_nan=True), (
            f"feature changed when future rows were removed at {prices.index[cut].date()}: "
            f"{(a - b).abs().sort_values(ascending=False).head(3).to_dict()}"
        )


def test_feature_list_is_complete(prices):
    f = build_features(prices)
    assert list(f.columns) == FEATURES and len(FEATURES) == 14


def test_labels_look_exactly_five_rows_ahead(prices):
    close = prices["close"]
    fwd = forward_return(close)
    i = 1000
    assert np.isclose(fwd.iloc[i], close.iloc[i + 5] / close.iloc[i] - 1)
    lab = make_labels(close)
    assert lab.iloc[-5:].isna().all() and lab.iloc[:-5].notna().all()
    assert set(lab.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_walk_forward_folds_never_train_on_the_future(prices):
    dates = prices.index[260:-5]
    for f in walk_forward_folds(dates):
        last_train, first_test = f.train_idx.max(), f.test_idx.min()
        assert dates[last_train].year < f.year
        assert first_test - last_train - 1 >= GAP  # 5 rows removed between train and predict
        assert (dates[f.test_idx].year == f.year).all()


def test_inner_folds_validate_after_training():
    for tr, va in inner_splits(3000):
        assert va.min() - tr.max() - 1 == GAP
        assert len(va) == 252


def test_backtest_trades_the_day_after_the_signal(prices):
    px = prices.iloc[-300:]
    sig = pd.Series(1, index=px.index[:-1])  # Hold everywhere...
    sig.iloc[10] = BUY  # ...except one Buy
    sig.iloc[50] = SELL  # ...and one Sell
    pf = run_portfolio(px, sig)
    order_days = pf.wrapper.index[pf.orders.records["idx"]]
    assert list(order_days) == [px.index[11], px.index[51]]


def test_vectorbt_matches_manual_backtest(prices):
    px = prices.iloc[-600:]
    rng = np.random.default_rng(1)
    sig = pd.Series(rng.choice([0, 1, 2], size=len(px) - 1, p=[0.2, 0.5, 0.3]), index=px.index[:-1])
    v_vbt = run_portfolio(px, sig).value()
    v_man = manual_backtest(px, sig)
    assert np.allclose(v_vbt.values, v_man.values, rtol=1e-9)
