"""資訊係數。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats.ic import information_coefficient


def _frames(n_months=120, n_stocks=100, noise=1.0, seed=0, sign=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    cols = [f"T{i}" for i in range(n_stocks)]
    sig = pd.DataFrame(rng.standard_normal((n_months, n_stocks)), index=idx, columns=cols)
    fwd = sign * sig * 0.01 + rng.standard_normal((n_months, n_stocks)) * noise * 0.05
    return sig, pd.DataFrame(fwd, index=idx, columns=cols)


def test_perfect_signal_gives_ic_one():
    sig, _ = _frames(noise=0.0, seed=1)
    res = information_coefficient(sig, sig * 0.01, min_stocks=10)
    assert res.mean_ic == pytest.approx(1.0)
    assert res.hit_rate == 1.0


def test_inverted_signal_gives_ic_minus_one():
    sig, _ = _frames(seed=2)
    res = information_coefficient(sig, -sig, min_stocks=10)
    assert res.mean_ic == pytest.approx(-1.0)


def test_pure_noise_signal_has_ic_near_zero():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    cols = [f"T{i}" for i in range(100)]
    sig = pd.DataFrame(rng.standard_normal((200, 100)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((200, 100)) * 0.05, index=idx, columns=cols)
    res = information_coefficient(sig, fwd, min_stocks=50)
    assert abs(res.mean_ic) < 0.03
    assert res.pvalue > 0.05


def test_weak_but_real_signal_is_detected():
    """IC 的檢定效率：微弱但真實的訊號，IC 仍抓得到。"""
    sig, fwd = _frames(n_months=200, noise=1.0, seed=4)
    res = information_coefficient(sig, fwd, min_stocks=50)
    assert res.mean_ic > 0
    assert res.pvalue < 0.05
    assert res.ic_ir > 0


def test_spearman_is_robust_to_outliers_relative_to_pearson():
    """單一極端值可以主導 Pearson，但動不了等級相關。"""
    sig, fwd = _frames(n_months=120, noise=0.5, seed=5)
    fwd.iloc[0, 0] = 1e6
    sp = information_coefficient(sig, fwd, "spearman", min_stocks=50)
    pe = information_coefficient(sig, fwd, "pearson", min_stocks=50)
    assert abs(sp.series.iloc[0]) > abs(pe.series.iloc[0])


def test_rejects_unknown_method():
    sig, fwd = _frames(seed=6)
    with pytest.raises(ValueError):
        information_coefficient(sig, fwd, method="kendall")


def test_raises_when_no_month_has_enough_stocks():
    sig, fwd = _frames(n_stocks=10, seed=7)
    with pytest.raises(ValueError, match="最低股票數"):
        information_coefficient(sig, fwd, min_stocks=50)


def test_series_index_matches_used_months():
    sig, fwd = _frames(n_months=60, seed=8)
    res = information_coefficient(sig, fwd, min_stocks=50)
    assert res.nobs_months == len(res.series)
    assert res.series.index.isin(sig.index).all()
