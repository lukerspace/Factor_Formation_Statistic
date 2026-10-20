"""定態 bootstrap 的性質測試。"""

import numpy as np
import pytest

from factorlab.stats import bootstrap as bs


def test_indices_shape_and_range():
    idx = bs.stationary_bootstrap_indices(50, 20, block=5.0, rng=np.random.default_rng(1))
    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_indices_reproducible_with_seed():
    a = bs.stationary_bootstrap_indices(30, 10, 4.0, np.random.default_rng(42))
    b = bs.stationary_bootstrap_indices(30, 10, 4.0, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_longer_block_means_more_contiguity():
    """區塊越長，相鄰索引為連續的比例越高 —— 這是定態 bootstrap 的定義性質。"""
    def contiguity(block):
        idx = bs.stationary_bootstrap_indices(500, 200, block, np.random.default_rng(5))
        return float((np.diff(idx, axis=1) == 1).mean())

    assert contiguity(20.0) > contiguity(2.0)


def test_circular_block_shape():
    idx = bs.circular_block_bootstrap_indices(40, 15, block=6, rng=np.random.default_rng(2))
    assert idx.shape == (15, 40)
    assert idx.max() < 40


def test_bootstrap_mean_ci_covers_truth():
    x = np.random.default_rng(9).standard_normal(400) * 0.05 + 0.01
    ci = bs.bootstrap_mean(x, reps=1000, block=4.0, seed=3)
    assert ci.lower < 0.01 < ci.upper
    assert ci.pvalue < 0.05


def test_bootstrap_test_size_is_near_alpha():
    """型一錯誤率檢定：虛無為真時，p < 0.05 的比例應該接近 0.05。

    不用單一種子斷言「p 值很大」—— 任何單一抽樣都可能剛好極端
    （rng(10) 的 300 個標準常態樣本均值就落在 -3.6 個標準差）。
    檢定的 size 才是該驗的性質。
    """
    rejects = 0
    trials = 60
    for s in range(trials):
        x = np.random.default_rng(1000 + s).standard_normal(240) * 0.05
        ci = bs.bootstrap_mean(x, reps=400, block=4.0, seed=s)
        rejects += int(ci.pvalue < 0.05)
    rate = rejects / trials
    assert rate < 0.20, f"型一錯誤率 {rate:.3f} 明顯高於名目水準 0.05"


def test_bootstrap_ci_brackets_sample_mean():
    x = np.random.default_rng(10).standard_normal(300) * 0.05
    ci = bs.bootstrap_mean(x, reps=1000, block=4.0, seed=3)
    assert ci.lower < ci.point < ci.upper


def test_bootstrap_sharpe_point_matches_analytic():
    x = np.random.default_rng(21).standard_normal(240) * 0.04 + 0.006
    ci = bs.bootstrap_sharpe(x, reps=500, seed=1)
    expected = x.mean() / x.std(ddof=1) * np.sqrt(12)
    assert ci.point == pytest.approx(expected)
    assert ci.lower < ci.point < ci.upper


def test_short_series_returns_nan():
    ci = bs.bootstrap_mean([0.01, 0.02], reps=100)
    assert np.isnan(ci.point)
    assert ci.reps == 0


def test_bootstrap_statistic_generic():
    x = np.random.default_rng(4).standard_normal(200)
    out = bs.bootstrap_statistic(x, lambda a: float(np.median(a)), reps=100, seed=2)
    assert out.shape == (100,)
    assert np.isfinite(out).all()
