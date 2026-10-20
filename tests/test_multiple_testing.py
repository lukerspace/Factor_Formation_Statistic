"""多重檢定校正、HLZ haircut、Deflated Sharpe。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats import multiple_testing as mt


def test_bonferroni_scales_by_m():
    p = [0.01, 0.02, 0.5, 0.9]
    assert np.allclose(mt.bonferroni(p), [0.04, 0.08, 1.0, 1.0])


def test_holm_is_at_least_as_large_as_raw_and_at_most_bonferroni():
    p = np.array([0.001, 0.008, 0.03, 0.2, 0.7])
    holm = mt.holm(p)
    assert (holm >= p - 1e-12).all()
    assert (holm <= mt.bonferroni(p) + 1e-12).all()


def test_holm_is_monotone_in_sorted_order():
    p = np.array([0.001, 0.008, 0.03, 0.2, 0.7])
    holm = mt.holm(p)
    assert (np.diff(holm[np.argsort(p)]) >= -1e-12).all()


def test_by_is_more_conservative_than_bh():
    """BY 多了 c(m) = Σ1/i 的懲罰，因為它不假設檢定間獨立。"""
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.3, 0.6])
    assert (mt.benjamini_yekutieli(p) >= mt.benjamini_hochberg(p) - 1e-12).all()


def test_all_corrections_bounded_by_one():
    p = np.array([0.4, 0.5, 0.9, 0.99])
    for fn in (mt.bonferroni, mt.holm, mt.benjamini_hochberg, mt.benjamini_yekutieli):
        assert (fn(p) <= 1.0).all()


def test_p_to_t_roundtrip():
    t = np.array([1.96, 2.58, 3.0])
    p = 2.0 * (1.0 - np.array([0.975, 0.995, 0.99865]))
    assert np.allclose(mt.p_to_t(p), t, atol=0.02)


def test_asymptotic_approximation_overstates_exact_expected_max():
    """√(2 ln M) 在 M 小時會高估 E[max|t|]。

    M = 32：漸近近似 2.633，但精確（Monte Carlo）值約 2.35。
    兩者都遠高於 1.96，結論不變；但報告必須並列，不能拿好看的那個
    當事實。5% 水準的實際門檻是 max|t| 的 95% 分位，約 3.0。
    """
    res = mt.expected_max_abs_t(32, rho=0.0, reps=40000, seed=1)
    assert res["asymptotic_sqrt_2logM"] == pytest.approx(2.633, abs=0.01)
    assert res["mc_expected_max_abs_t"] == pytest.approx(2.35, abs=0.06)
    assert res["asymptotic_sqrt_2logM"] > res["mc_expected_max_abs_t"]
    assert res["mc_q95_max_abs_t"] == pytest.approx(3.0, abs=0.2)
    assert res["mc_expected_max_abs_t"] > 1.96


def test_expected_max_abs_t_decreases_with_correlation():
    """檢定高度相關時，有效獨立檢定數下降，max|t| 的期望隨之降低。"""
    indep = mt.expected_max_abs_t(32, rho=0.0, reps=20000, seed=2)
    corr = mt.expected_max_abs_t(32, rho=0.9, reps=20000, seed=2)
    assert corr["mc_expected_max_abs_t"] < indep["mc_expected_max_abs_t"]
    # 即使 rho=0.9，門檻仍明顯高於 1.96。
    assert corr["mc_q95_max_abs_t"] > 1.96


def test_expected_max_abs_t_grows_with_trials():
    a = mt.expected_max_abs_t(4, reps=20000, seed=3)["mc_expected_max_abs_t"]
    b = mt.expected_max_abs_t(1000, reps=20000, seed=3)["mc_expected_max_abs_t"]
    assert b > a


def test_haircut_reduces_sharpe():
    t = np.array([2.1, 2.5, 3.5])
    s = np.array([0.5, 0.6, 0.9])
    out = mt.haircut_sharpe(t, s)
    for m in ("bonferroni", "holm", "by"):
        assert (out[f"{m}_sharpe_haircut"] <= s + 1e-12).all()
        assert (out[f"{m}_haircut_pct"] >= -1e-9).all()


def test_haircut_is_larger_for_weaker_tstats():
    """Bonferroni 下所有檢定共用同一個乘數，弱 t 值的相對折價較大。"""
    t = np.array([2.0, 5.0])
    s = np.array([0.5, 0.5])
    out = mt.haircut_sharpe(t, s)
    assert out["bonferroni_haircut_pct"].iloc[0] > out["bonferroni_haircut_pct"].iloc[1]


def test_holm_leaves_the_largest_pvalue_unadjusted():
    """Holm 的逐步下降：最大的 p 值乘數是 1，因此不被調整、折價為零。

    這是 Holm 相對 Bonferroni 更有檢定力的來源，但也意味著「最弱的那個
    策略沒有被折價」—— 解讀時要看的是它的原始 p 值本來就不顯著。
    """
    t = np.array([2.0, 5.0])
    out = mt.haircut_sharpe(t, np.array([0.5, 0.5]))
    assert out["holm_haircut_pct"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["holm_p_adj"].iloc[0] == pytest.approx(out["p_orig"].iloc[0])


def test_haircut_flags_hlz_threshold():
    out = mt.haircut_sharpe([2.5, 3.5], [0.5, 0.7], hlz_threshold=3.0)
    assert out["survives_hlz_t3"].tolist() == [False, True]


def test_haircut_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mt.haircut_sharpe([1.0, 2.0], [0.5])


def test_haircut_rejects_unknown_method():
    with pytest.raises(ValueError):
        mt.haircut_sharpe([2.0], [0.5], methods=("sidak",))


def test_expected_max_sharpe_grows_with_trials_and_variance():
    a = mt.expected_max_sharpe(0.01, 10)
    b = mt.expected_max_sharpe(0.01, 1000)
    c = mt.expected_max_sharpe(0.04, 1000)
    assert b > a > 0
    assert c > b


def test_expected_max_sharpe_nan_for_zero_variance():
    assert np.isnan(mt.expected_max_sharpe(0.0, 100))


def test_deflated_sharpe_low_for_noise_selected_from_many_trials():
    """純噪音裡選出來的最佳策略，DSR 應該遠低於 0.95。"""
    rng = np.random.default_rng(5)
    trials = rng.standard_normal((215, 200)) * 0.05
    sharpes = trials.mean(axis=0) / trials.std(axis=0, ddof=1) * np.sqrt(12)
    best = trials[:, int(np.argmax(sharpes))]
    res = mt.deflated_sharpe_ratio(best, n_trials=200, trial_sharpes=sharpes)
    assert res.sharpe_ann > 0.5, "挑出來的最佳策略在樣本內看起來很好"
    assert res.dsr < 0.95, f"但 DSR={res.dsr:.3f} 應該擋下來"
    assert res.expected_max_sharpe_ann > 0


def test_deflated_sharpe_high_for_genuinely_strong_single_strategy():
    r = np.random.default_rng(7).standard_normal(215) * 0.03 + 0.015
    res = mt.deflated_sharpe_ratio(r, n_trials=1, trial_sharpes=None)
    assert res.dsr > 0.95
    assert res.significant is True


def test_deflated_sharpe_penalty_grows_with_trials():
    r = np.random.default_rng(9).standard_normal(215) * 0.04 + 0.008
    few = mt.deflated_sharpe_ratio(r, n_trials=2)
    many = mt.deflated_sharpe_ratio(r, n_trials=5000)
    assert many.expected_max_sharpe_ann > few.expected_max_sharpe_ann
    assert many.dsr < few.dsr


def test_deflated_sharpe_short_series_returns_nan():
    res = mt.deflated_sharpe_ratio([0.01] * 5, n_trials=10)
    assert np.isnan(res.dsr)
