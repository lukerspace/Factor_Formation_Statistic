"""
檢定力分析 —— 這組測試把「3 年樣本測不出東西」變成可驗證的事實。
"""

import numpy as np
import pytest

from factorlab.stats import power


def test_mde_matches_closed_form():
    sigma, n = 0.0558, 23
    expected = (1.959963985 + 0.8416212336) * sigma / np.sqrt(n)
    assert power.minimum_detectable_effect(sigma, n) == pytest.approx(expected, rel=1e-6)


def test_mde_shrinks_with_sample_size():
    s = 0.05
    assert power.minimum_detectable_effect(s, 400) < power.minimum_detectable_effect(s, 100)


def test_mde_scales_with_one_over_sqrt_n():
    s = 0.05
    a = power.minimum_detectable_effect(s, 100)
    b = power.minimum_detectable_effect(s, 400)
    assert a / b == pytest.approx(2.0, rel=1e-6)


def test_original_study_mde_is_absurdly_large():
    """重現原始輸出的 12M Q10-Q1：n=23、σ=5.576%。

    原始碼報 p = 0.2654 並隱含「沒有動能」。實際上這份樣本在 80% 檢定力
    下只能偵測到月 3% 以上的效果，年化超過 40% —— 遠高於任何文獻估計。
    """
    mde = power.minimum_detectable_effect(0.05576, 23)
    assert mde > 0.030
    assert (1 + mde) ** 12 - 1 > 0.40


def test_power_at_literature_effect_is_tiny_for_three_year_sample():
    """對文獻動能溢酬（月 0.8%），3 年樣本的檢定力極低。"""
    p23 = power.power_at_effect(0.008, 0.05576, 23)
    assert p23 < 0.15, f"3 年樣本檢定力 {p23:.3f} 應該極低"


def test_power_improves_materially_with_long_history():
    """215 個月（2007–2024）的檢定力仍不足 80%，但已進入同一數量級。"""
    p23 = power.power_at_effect(0.008, 0.05576, 23)
    p215 = power.power_at_effect(0.008, 0.05576, 215)
    assert p215 > 3 * p23
    assert p215 < 0.80, "即使 215 個月，對月 0.8% 的檢定力仍低於 80%"


def test_power_at_mde_equals_target():
    """定義性質：效果恰為 MDE 時，檢定力應約等於目標值。"""
    sigma, n, target = 0.05, 200, 0.80
    mde = power.minimum_detectable_effect(sigma, n, 0.05, target)
    assert power.power_at_effect(mde, sigma, n, 0.05) == pytest.approx(target, abs=0.03)


def test_power_is_monotone_in_effect_and_n():
    assert power.power_at_effect(0.012, 0.05, 100) > power.power_at_effect(0.006, 0.05, 100)
    assert power.power_at_effect(0.008, 0.05, 400) > power.power_at_effect(0.008, 0.05, 100)


def test_power_at_zero_effect_equals_alpha():
    assert power.power_at_effect(0.0, 0.05, 200, alpha=0.05) == pytest.approx(0.05, abs=0.005)


def test_stricter_alpha_lowers_power():
    """多重檢定校正後 α 變嚴，檢定力下降、MDE 上升。"""
    loose = power.power_at_effect(0.008, 0.05, 215, alpha=0.05)
    strict = power.power_at_effect(0.008, 0.05, 215, alpha=0.05 / 32)
    assert strict < loose
    assert (power.minimum_detectable_effect(0.05, 215, alpha=0.05 / 32)
            > power.minimum_detectable_effect(0.05, 215, alpha=0.05))


def test_required_nobs_reaches_target_power():
    n = power.required_nobs(0.008, 0.05576, 0.05, 0.80)
    assert power.power_at_effect(0.008, 0.05576, int(n), 0.05) >= 0.80
    assert n > 300, "要偵測月 0.8% 需要數百個月的樣本"


def test_required_nobs_decreases_with_larger_effect():
    assert power.required_nobs(0.02, 0.05) < power.required_nobs(0.005, 0.05)


def test_diagnose_from_returns():
    """重現原始輸出的 12M Q10-Q1：n=23、月均 1.33%、月標準差 5.576%。

    23 個抽樣的樣本標準差離母體值很遠，所以這裡把序列標準化到剛好
    0.05576，讓測試檢驗的是 diagnose 的邏輯而不是抽樣運氣。
    """
    raw = np.random.default_rng(1).standard_normal(23)
    r = (raw - raw.mean()) / raw.std(ddof=1) * 0.05576 + 0.0133
    res = power.diagnose(r)
    assert res.nobs == 23
    assert res.sigma_m == pytest.approx(0.05576)
    assert res.observed_mean_m == pytest.approx(0.0133)
    assert res.underpowered is True
    assert res.mde_m > 0.030
    assert res.power_at_reference < 0.15
    assert res.required_n_for_reference > 300


def test_diagnose_requires_inputs():
    with pytest.raises(ValueError):
        power.diagnose()


def test_power_table_shape_and_monotonicity():
    tbl = power.power_table(0.05)
    assert "nobs_months" in tbl.columns
    assert tbl["mde_m_at_80pct"].is_monotonic_decreasing
    eff_cols = [c for c in tbl.columns if c.startswith("power_at_")]
    assert len(eff_cols) == 4
    for c in eff_cols:
        assert tbl[c].is_monotonic_increasing


def test_nan_guards():
    assert np.isnan(power.minimum_detectable_effect(0.0, 100))
    assert np.isnan(power.minimum_detectable_effect(0.05, 1))
    assert np.isnan(power.power_at_effect(0.01, -1.0, 100))
    assert np.isnan(power.required_nobs(0.0, 0.05))
