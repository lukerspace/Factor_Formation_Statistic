"""Fama-MacBeth 兩階段橫斷面迴歸。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats.famamacbeth import fama_macbeth


def _setup(n_months=150, n_stocks=120, gamma=0.004, seed=0, control_effect=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    cols = [f"T{i}" for i in range(n_stocks)]
    sig = pd.DataFrame(rng.standard_normal((n_months, n_stocks)), index=idx, columns=cols)
    ctrl = pd.DataFrame(rng.standard_normal((n_months, n_stocks)), index=idx, columns=cols)
    fwd = (
        gamma * sig
        + control_effect * ctrl
        + pd.DataFrame(rng.standard_normal((n_months, n_stocks)) * 0.05, index=idx, columns=cols)
    )
    return sig, fwd, ctrl


def test_recovers_known_gamma():
    """訊號已 z 分數化，所以 γ 的單位就是「每一個橫斷面標準差」。"""
    sig, fwd, _ = _setup(gamma=0.004, seed=1)
    res = fama_macbeth(sig, fwd, min_stocks=50)
    g = res.means[res.names.index("signal")]
    assert g == pytest.approx(0.004, abs=0.0008)
    assert res.tstats[res.names.index("signal")] > 3


def test_no_relation_gives_insignificant_gamma():
    sig, fwd, _ = _setup(gamma=0.0, seed=2)
    res = fama_macbeth(sig, fwd, min_stocks=50)
    assert res.pvalues[res.names.index("signal")] > 0.05


def test_control_absorbs_its_own_effect():
    """控制變數真正有效果時，它的係數應顯著，且訊號係數不受污染。"""
    sig, fwd, ctrl = _setup(gamma=0.004, control_effect=0.006, seed=3)
    res = fama_macbeth(sig, fwd, controls={"ctrl": ctrl}, min_stocks=50)
    i_sig = res.names.index("signal")
    i_ctrl = res.names.index("ctrl")
    assert res.tstats[i_ctrl] > 3
    assert res.means[i_sig] == pytest.approx(0.004, abs=0.001)


def test_correlated_omitted_variable_biases_gamma():
    """訊號與一個有報酬效果的特徵相關時，不控制它會讓 γ 被高估。

    這就是為什麼本研究的 Fama-MacBeth 一定要控制規模代理、短期反轉
    與波動度 —— 動能的高分位系統性地偏向這些特徵。
    """
    rng = np.random.default_rng(4)
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    cols = [f"T{i}" for i in range(120)]
    ctrl = pd.DataFrame(rng.standard_normal((200, 120)), index=idx, columns=cols)
    # 訊號與 ctrl 相關 0.7，但訊號自己對報酬完全沒有效果。
    sig = 0.7 * ctrl + np.sqrt(1 - 0.49) * pd.DataFrame(
        rng.standard_normal((200, 120)), index=idx, columns=cols
    )
    fwd = 0.008 * ctrl + pd.DataFrame(
        rng.standard_normal((200, 120)) * 0.04, index=idx, columns=cols
    )

    naive = fama_macbeth(sig, fwd, min_stocks=50)
    controlled = fama_macbeth(sig, fwd, controls={"ctrl": ctrl}, min_stocks=50)
    i = naive.names.index("signal")
    assert naive.tstats[i] > 3, "不控制時訊號看起來顯著"
    assert abs(controlled.tstats[controlled.names.index("signal")]) < 3, "控制後消失"


def test_coefs_frame_shape_and_names():
    sig, fwd, ctrl = _setup(seed=5)
    res = fama_macbeth(sig, fwd, controls={"ctrl": ctrl}, min_stocks=50)
    assert list(res.coefs.columns) == ["intercept", "signal", "ctrl"]
    assert len(res.coefs) == res.nobs_months
    assert res.avg_n_stocks == pytest.approx(120.0)


def test_raises_when_no_month_qualifies():
    sig, fwd, _ = _setup(n_stocks=10, seed=6)
    with pytest.raises(ValueError, match="最低股票數"):
        fama_macbeth(sig, fwd, min_stocks=50)


def test_to_frame_has_expected_columns():
    sig, fwd, _ = _setup(seed=7)
    res = fama_macbeth(sig, fwd, min_stocks=50)
    assert list(res.to_frame().columns) == ["gamma", "se", "t", "p"]
