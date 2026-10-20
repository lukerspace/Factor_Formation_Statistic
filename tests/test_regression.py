"""迴歸與 GRS 聯合檢定。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats.regression import grs_test, ols_hac


@pytest.fixture
def known_dgp():
    """y = 0.01 + 1.5·f1 - 0.5·f2 + ε，係數已知。"""
    rng = np.random.default_rng(42)
    n = 600
    f = pd.DataFrame(rng.standard_normal((n, 2)) * 0.04, columns=["f1", "f2"])
    y = 0.01 + 1.5 * f["f1"] - 0.5 * f["f2"] + rng.standard_normal(n) * 0.01
    return y, f


def test_ols_recovers_known_coefficients(known_dgp):
    y, f = known_dgp
    res = ols_hac(y, f)
    assert res.alpha == pytest.approx(0.01, abs=0.002)
    assert res.coef("f1") == pytest.approx(1.5, abs=0.05)
    assert res.coef("f2") == pytest.approx(-0.5, abs=0.05)
    assert res.r2 > 0.95


def test_ols_standard_errors_positive(known_dgp):
    y, f = known_dgp
    res = ols_hac(y, f)
    assert (res.se > 0).all()
    assert np.isfinite(res.tstats).all()


def test_alpha_is_the_intercept(known_dgp):
    y, f = known_dgp
    res = ols_hac(y, f)
    assert res.names[0] == "const"
    assert res.alpha == res.params[0]
    assert abs(res.alpha_t) > 3


def test_zero_alpha_is_not_significant():
    rng = np.random.default_rng(1)
    n = 500
    f = pd.DataFrame(rng.standard_normal((n, 1)) * 0.04, columns=["mkt"])
    y = 1.0 * f["mkt"] + rng.standard_normal(n) * 0.02
    res = ols_hac(y, f)
    assert res.alpha_p > 0.05


def test_hac_lags_increase_standard_errors_for_persistent_residuals():
    """殘差有自相關時，HAC 標準誤應大於 lags=0 的版本。"""
    rng = np.random.default_rng(8)
    n = 500
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal() * 0.01
    f = pd.DataFrame(rng.standard_normal((n, 1)) * 0.04, columns=["mkt"])
    y = pd.Series(0.005 + e)
    se0 = ols_hac(y, f, lags=0).se[0]
    se12 = ols_hac(y, f, lags=12).se[0]
    assert se12 > se0


def test_ols_aligns_on_index_and_drops_nan():
    idx = pd.date_range("2010-01-31", periods=50, freq="ME")
    y = pd.Series(np.arange(50.0), index=idx)
    y.iloc[5] = np.nan
    x = pd.DataFrame({"x": np.arange(50.0)}, index=idx)
    res = ols_hac(y, x)
    assert res.nobs == 49


def test_ols_raises_on_empty_overlap():
    y = pd.Series([1.0, 2.0], index=pd.date_range("2010-01-31", periods=2, freq="ME"))
    x = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    with pytest.raises(ValueError):
        ols_hac(y, x)


# ------------------------------------------------- GRS


def _grs_data(alpha_vec, n=300, seed=0):
    rng = np.random.default_rng(seed)
    k = 2
    f = pd.DataFrame(rng.standard_normal((n, k)) * 0.04, columns=["mkt", "smb"])
    betas = np.linspace(0.6, 1.4, len(alpha_vec))
    r = {}
    for i, a in enumerate(alpha_vec):
        r[f"Q{i + 1}"] = a + betas[i] * f["mkt"] + 0.3 * f["smb"] + rng.standard_normal(n) * 0.02
    return pd.DataFrame(r), f


def test_grs_does_not_reject_when_alphas_are_zero():
    r, f = _grs_data(np.zeros(10), seed=3)
    res = grs_test(r, f)
    assert res.pvalue > 0.05
    assert res.n_assets == 10
    assert res.n_factors == 2


def test_grs_rejects_when_alphas_are_nonzero():
    r, f = _grs_data(np.linspace(-0.01, 0.01, 10), seed=4)
    res = grs_test(r, f)
    assert res.pvalue < 0.01
    assert res.mean_abs_alpha > 0.002


def test_grs_statistic_is_positive():
    r, f = _grs_data(np.zeros(10), seed=5)
    res = grs_test(r, f)
    assert res.statistic > 0
    assert res.sr_squared_alpha >= 0


def test_grs_requires_enough_observations():
    """T <= N + K 時必須明確拒絕 —— 這正是 3 年樣本做不了十分位 GRS 的原因。"""
    r, f = _grs_data(np.zeros(10), n=11, seed=6)
    with pytest.raises(ValueError, match="T > N \\+ K"):
        grs_test(r, f)


def test_grs_size_is_approximately_nominal():
    """虛無為真時，GRS 在 5% 水準的拒絕率應該接近 5%。"""
    rejects = 0
    trials = 40
    for s in range(trials):
        r, f = _grs_data(np.zeros(10), n=300, seed=100 + s)
        rejects += int(grs_test(r, f).pvalue < 0.05)
    assert rejects / trials < 0.25
