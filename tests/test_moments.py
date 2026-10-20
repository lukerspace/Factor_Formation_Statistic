"""描述統計與 Sharpe 標準誤。"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from factorlab.stats.moments import describe, max_drawdown, sharpe_se


def test_sharpe_matches_manual_calculation():
    r = pd.Series(np.random.default_rng(1).standard_normal(240) * 0.04 + 0.006)
    m = describe(r)
    assert m.sharpe == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12))


def test_max_drawdown_known_path():
    # 淨值 1 → 1.5 → 0.75：最大回撤 = 0.75/1.5 - 1 = -50%
    r = pd.Series([0.5, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_monotone_up_is_zero():
    assert max_drawdown(pd.Series([0.01] * 20)) == pytest.approx(0.0)


def test_sharpe_se_reduces_to_iid_for_normal_data():
    """常態報酬（偏態 0、超峰態 0、Sharpe 小）時，Sharpe 標準誤 ≈ √(12/T)。"""
    r = np.random.default_rng(5).standard_normal(5000) * 0.04
    se = sharpe_se(r)
    assert se == pytest.approx(np.sqrt(12 / 5000), rel=0.10)


def test_negative_skew_inflates_sharpe_se():
    """負偏態會讓 Sharpe 的真實標準誤變大 —— 天真做法會高估顯著性。

    Lo (2002)：Var(SR) ∝ 1 + SR²/2 - SR·γ3 + SR²(γ4-3)/4。
    SR > 0 且 γ3 < 0 時，-SR·γ3 > 0，標準誤被推高。
    """
    n = 4000
    # 負偏態序列：先去均值再縮放，否則 skewnorm 自身的平均數會讓
    # r 的平均變成負的，SR < 0 時 -SR·γ3 反而會縮小標準誤。
    base = -sps.skewnorm.rvs(a=8, size=n, random_state=11)
    base = base - base.mean()
    r = base / base.std() * 0.04 + 0.01
    assert sps.skew(r) < -0.5
    assert r.mean() > 0
    se_corrected = sharpe_se(r)
    se_iid = np.sqrt(12 / n)
    assert se_corrected > se_iid


def test_describe_fields_are_finite():
    r = pd.Series(np.random.default_rng(3).standard_normal(200) * 0.05 + 0.004)
    m = describe(r)
    for name in ("mean_m", "vol_ann", "sharpe", "sharpe_se", "skew",
                 "excess_kurtosis", "hit_rate", "max_drawdown", "cum_return"):
        assert np.isfinite(getattr(m, name)), name
    assert 0.0 <= m.hit_rate <= 1.0
    assert m.nobs == 200


def test_describe_handles_short_series():
    m = describe(pd.Series([0.01]))
    assert m.nobs == 1
    assert m.mean_m == pytest.approx(0.01)
    assert np.isnan(m.sharpe)


def test_describe_handles_empty_series():
    m = describe(pd.Series([], dtype="float64"))
    assert m.nobs == 0
    assert np.isnan(m.mean_m)


def test_describe_handles_zero_variance():
    m = describe(pd.Series([0.01] * 50))
    assert np.isnan(m.sharpe)
