"""
HAC 的核心測試：重疊報酬的 √h 膨脹。

這是整個 repo 最重要的一組測試。如果它過了，就證明原始碼的
`ttest_1samp` 在 fwd_12m 上高估了約 3.46 倍的 t 值。
"""

import numpy as np
import pytest

from factorlab.stats import hac
from conftest import make_overlapping


def test_kernel_weights_bartlett_decays_to_zero():
    w = hac.kernel_weights(4, "bartlett")
    assert np.allclose(w, [0.8, 0.6, 0.4, 0.2])


def test_kernel_weights_truncated_is_flat():
    assert np.allclose(hac.kernel_weights(3, "truncated"), [1.0, 1.0, 1.0])


def test_kernel_weights_rejects_unknown():
    with pytest.raises(ValueError):
        hac.kernel_weights(2, "quadratic-spectral")


def test_lags_for_overlap():
    assert hac.lags_for_overlap(1) == 0
    assert hac.lags_for_overlap(12) == 11


def test_inflation_factor_is_sqrt_h():
    assert hac.inflation_factor(12) == pytest.approx(np.sqrt(12))


def test_iid_series_hac_se_close_to_iid_se():
    """沒有自相關時，HAC 標準誤應該跟 i.i.d. 標準誤很接近。"""
    x = np.random.default_rng(7).standard_normal(4000) * 0.05
    res = hac.hac_mean_test(x)
    assert res.inflation == pytest.approx(1.0, abs=0.12)


@pytest.mark.parametrize("h", [3, 6, 12])
def test_hansen_hodrick_recovers_sqrt_h_inflation(h):
    """Hansen-Hodrick（截斷核、h-1 落後期）應回復理論上的 √h 膨脹。

    理論：若底層報酬 i.i.d.、方差 σ²，h 期重疊和的長期變異數是 h²σ²，
    而 γ0 = hσ²，故 S/γ0 = h，標準誤比值 = √h。
    """
    x = np.random.default_rng(11).standard_normal(20000) * 0.04
    ov = make_overlapping(x, h)
    res = hac.hansen_hodrick_mean_test(ov, horizon=h)
    assert res.inflation == pytest.approx(np.sqrt(h), rel=0.12)


@pytest.mark.parametrize("h", [3, 6, 12])
def test_bartlett_underestimates_overlap_inflation(h):
    """Bartlett 核對重疊報酬會低估膨脹 —— 介於 1 與 √h 之間。

    這不是 bug，是 Bartlett 權重 (1-j/L+1) 與重疊結構的真實權重
    (1-j/h) 不一致造成的，代價是換來半正定保證。報告同時列出兩者。
    """
    x = np.random.default_rng(13).standard_normal(20000) * 0.04
    ov = make_overlapping(x, h)
    res = hac.newey_west_mean_test(ov, horizon=h)
    assert 1.0 < res.inflation < np.sqrt(h)


def test_overlap_tstat_is_materially_inflated():
    """最關鍵的一條：天真 t 值相對 HAC t 值被高估到什麼程度。"""
    x = np.random.default_rng(17).standard_normal(6000) * 0.05 + 0.002
    ov = make_overlapping(x, 12)
    res = hac.hansen_hodrick_mean_test(ov, horizon=12)
    assert abs(res.iid_tstat) > abs(res.tstat)
    assert abs(res.iid_tstat) / abs(res.tstat) == pytest.approx(np.sqrt(12), rel=0.15)


def test_newey_west_bandwidth_grows_with_sample():
    assert hac.newey_west_bandwidth(100) <= hac.newey_west_bandwidth(1000)
    assert hac.newey_west_bandwidth(1) == 0


def test_andrews_bandwidth_larger_for_persistent_series():
    rng = np.random.default_rng(3)
    iid = rng.standard_normal(2000)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.8 * ar[t - 1] + rng.standard_normal()
    assert hac.andrews_bandwidth(ar) > hac.andrews_bandwidth(iid)


def test_short_series_returns_nan_not_garbage():
    res = hac.hac_mean_test([0.01, 0.02])
    assert np.isnan(res.tstat)
    assert res.nobs == 2


def test_long_run_variance_flags_non_psd():
    """截斷核在特定序列上可能算出負的長期變異數，必須回報 nan 而不是假數字。"""
    # 強烈負自相關會讓截斷核的 S 變負。
    x = np.array([1.0, -1.0] * 40)
    s, lags, failed = hac.long_run_variance(x, lags=1, kernel="truncated")
    assert failed is True
    assert np.isnan(s)


def test_psd_failure_propagates_to_result():
    x = np.array([1.0, -1.0] * 40)
    res = hac.hac_mean_test(x, lags=1, kernel="truncated")
    assert res.psd_failed is True
    assert np.isnan(res.tstat)
