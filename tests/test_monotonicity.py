"""Patton-Timmermann 單調性檢定。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats.monotonicity import mr_test, quantile_profile


def _panel(means, n=240, sd=0.04, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    data = {f"Q{i + 1}": rng.standard_normal(n) * sd + m for i, m in enumerate(means)}
    return pd.DataFrame(data, index=idx)


def test_rejects_null_for_strongly_monotone_pattern():
    """報酬隨分位嚴格遞增且幅度夠大時，MR 檢定應拒絕虛無。

    參數要選得讓**每一個**相鄰差都明顯為正，因為統計量是 min。
    這裡相鄰差 0.003、每個分位平均數的標準誤 0.01/√400 = 0.0005，
    相鄰差的標準誤約 0.0007 → 每個差約 4.2 個標準誤。
    （用 sd=0.03、n=300 時相鄰差只有 0.8 個標準誤，min 必然為負，
     MR 不會拒絕 —— 那是檢定嚴格的正確表現，不是 bug。）
    """
    df = _panel(np.linspace(0.0, 0.027, 10), n=400, sd=0.01, seed=1)
    res = mr_test(df, reps=800, seed=1)
    assert res.pvalue < 0.05
    assert res.n_positive == 9
    assert res.monotone is True
    assert res.spearman == pytest.approx(1.0)


def test_does_not_reject_for_flat_pattern():
    """所有分位平均相同時，不應拒絕虛無。"""
    df = _panel(np.zeros(10), n=300, seed=2)
    res = mr_test(df, reps=800, seed=2)
    assert res.pvalue > 0.05


def test_does_not_reject_when_only_extremes_differ():
    """只有兩端有差、中間是平的 —— Q10-Q1 會顯著，但 MR 不該拒絕。

    這正是 MR 檢定的價值：它擋得住「價差來自兩端特質」這種情況。
    用與單調測試相同的雜訊水準（sd=0.01, n=400），差別只在型態，
    所以結論的差異只能來自型態而非檢定力。
    """
    means = np.zeros(10)
    means[0] = -0.015
    means[-1] = 0.015
    df = _panel(means, n=400, sd=0.01, seed=3)
    res = mr_test(df, reps=800, seed=3)
    assert res.top_minus_bottom == pytest.approx(0.03, abs=0.01)
    assert res.pvalue > 0.05, "中間是平的，MR 不該宣稱單調"
    assert res.n_positive < 9


def test_down_direction_detects_decreasing_pattern():
    df = _panel(np.linspace(0.027, 0.0, 10), n=400, sd=0.01, seed=4)
    up = mr_test(df, direction="up", reps=500, seed=4)
    down = mr_test(df, direction="down", reps=500, seed=4)
    assert down.pvalue < 0.05
    assert up.pvalue > 0.05


def test_requires_minimum_quantiles_and_observations():
    df = _panel(np.zeros(2), n=100)
    with pytest.raises(ValueError, match="3 個分位"):
        mr_test(df)
    with pytest.raises(ValueError, match="觀察值"):
        mr_test(_panel(np.zeros(5), n=8))


def test_invalid_direction():
    with pytest.raises(ValueError):
        mr_test(_panel(np.zeros(5), n=100), direction="sideways")


def test_quantile_profile_shape_and_ordering():
    df = _panel(np.linspace(0.0, 0.01, 10), n=200, seed=5)
    prof = quantile_profile(df)
    assert len(prof) == 10
    assert set(["quantile", "mean_m", "t_hac", "sharpe"]).issubset(prof.columns)
    assert prof["mean_m"].iloc[-1] > prof["mean_m"].iloc[0]
