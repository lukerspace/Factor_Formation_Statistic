"""
投組層測試 —— 日曆時間重疊投組是整份重構的核心，必須逐條驗。
"""

import numpy as np
import pandas as pd
import pytest

from factorlab.config import PortfolioSpec
from factorlab.portfolios import (
    apply_costs,
    build_calendar_time,
    overlapping_forward_returns,
)


@pytest.fixture
def deterministic():
    """A 每月 +2%、B 每月 -2%，A 永遠在高分位、B 永遠在低分位。"""
    idx = pd.date_range("2010-01-31", periods=36, freq="ME")
    mret = pd.DataFrame({"A": 0.02, "B": -0.02}, index=idx)
    q = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    return mret, q


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_deterministic_long_short_is_exact(deterministic, h):
    """報酬固定時，任何持有期的多空報酬都該是 4%/月。"""
    mret, q = deterministic
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=h))
    ls = res.long_short.dropna()
    assert len(ls) > 0
    assert np.allclose(ls.to_numpy(), 0.04)


def test_no_lookahead_first_month_has_no_return(deterministic):
    """第一個月沒有前一期訊號，不可以有報酬。"""
    mret, q = deterministic
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=1))
    assert np.isnan(res.returns.iloc[0]).all()
    assert res.n_sleeves.iloc[0] == 0


def test_signal_at_t_earns_return_at_t_plus_1():
    """訊號只在下一個月才生效 —— 時序約定的回歸測試。"""
    idx = pd.date_range("2010-01-31", periods=4, freq="ME")
    # B 只在第 3 個月(index 2)大漲。
    mret = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.50, 0.0]}, index=idx)
    # 在 index 1（第 2 個月末）把 B 放進高分位。
    q = pd.DataFrame({"A": [0.0, 0.0, 1.0, 1.0], "B": [1.0, 1.0, 0.0, 0.0]}, index=idx)
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=1))
    # index 1 的訊號（B 在高分位）賺 index 2 的報酬 = +50%。
    assert res.returns.iloc[2, 1] == pytest.approx(0.50)


def test_h1_equals_simple_monthly_decile_returns():
    """h=1 時日曆時間投組應等於「每月分位等權報酬」。"""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    cols = [f"T{i}" for i in range(40)]
    mret = pd.DataFrame(rng.standard_normal((60, 40)) * 0.05, index=idx, columns=cols)
    sig = pd.DataFrame(rng.standard_normal((60, 40)), index=idx, columns=cols)
    q = pd.DataFrame(
        np.ceil(sig.rank(axis=1, pct=True) * 4) - 1, index=idx, columns=cols
    )
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=4, horizon=1))

    # 手算：m 月的 q 分位報酬 = m-1 月末屬於該分位的股票在 m 月的等權平均。
    manual = mret.where(q.shift(1) == 3).mean(axis=1)
    got = res.returns[3]
    both = pd.concat([manual, got], axis=1).dropna()
    assert len(both) > 50
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_sleeve_count_ramps_up_to_horizon(deterministic):
    mret, q = deterministic
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=6))
    assert res.n_sleeves.iloc[1] == 1
    assert res.n_sleeves.iloc[6] == 6
    assert res.n_sleeves.max() == 6


def test_turnover_scales_inversely_with_horizon():
    """每月全換手的訊號下，單邊週轉應該約為 1/h。

    這是日曆時間投組最實際的好處：h 越大，每月只有 1/h 的部位需要
    換手。原始碼那種「重疊 h 月報酬」的算法根本無法回答成本問題。
    """
    idx = pd.date_range("2010-01-31", periods=80, freq="ME")
    cols = [f"T{i}" for i in range(20)]
    mret = pd.DataFrame(0.0, index=idx, columns=cols)  # 零報酬 → 權重不漂移
    rng = np.random.default_rng(7)
    # 每月重新隨機分組 → 成員完全洗牌。
    sig = pd.DataFrame(rng.standard_normal((80, 20)), index=idx, columns=cols)
    q = pd.DataFrame(np.ceil(sig.rank(axis=1, pct=True) * 2) - 1, index=idx, columns=cols)

    def mean_turnover(h):
        res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=h))
        # 跳過暖機期
        return float(res.turnover[1].iloc[h + 2 :].mean())

    t1, t4 = mean_turnover(1), mean_turnover(4)
    assert t1 > t4
    assert t4 / t1 == pytest.approx(0.25, abs=0.10)


def test_zero_turnover_when_membership_is_constant(deterministic):
    """成員與權重都不變時，週轉應為零。"""
    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    mret = pd.DataFrame(0.0, index=idx, columns=["A", "B"])  # 零報酬 → 無漂移
    q = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=1))
    assert np.allclose(res.turnover.iloc[3:].to_numpy(), 0.0)


def test_missing_return_drops_stock_and_renormalizes():
    """個股當月缺報酬時剔除並重新歸一化，不可當成 0%。"""
    idx = pd.date_range("2010-01-31", periods=3, freq="ME")
    mret = pd.DataFrame(
        {"A": [0.0, 0.10, 0.0], "B": [0.0, np.nan, 0.0], "C": [0.0, 0.0, 0.0]},
        index=idx,
    )
    q = pd.DataFrame({"A": [1.0] * 3, "B": [1.0] * 3, "C": [0.0] * 3}, index=idx)
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=1))
    # 高分位有 A、B；B 缺值被剔除 → 報酬 = A 的 10%，不是 (10%+0%)/2 = 5%。
    assert res.returns.iloc[1, 1] == pytest.approx(0.10)
    assert res.counts.iloc[1, 1] == 1


def _drift_fixture():
    """A 在第 2 個月翻倍，第 3 個月兩檔表現不同 —— 權重漂移才看得出差異。

    注意必須讓每個分位都有成員：`_membership_weights` 在任一分位為空時
    回傳 None，整個 sleeve 不會建立（這是刻意的 —— 多空價差沒有定義）。
    所以這裡放第三檔 C 在低分位。
    """
    idx = pd.date_range("2010-01-31", periods=3, freq="ME")
    mret = pd.DataFrame(
        {
            "A": [0.0, 1.00, 0.20],   # 第 2 個月翻倍，第 3 個月 +20%
            "B": [0.0, 0.00, 0.00],
            "C": [0.0, 0.00, 0.00],
        },
        index=idx,
    )
    q = pd.DataFrame({"A": [1.0] * 3, "B": [1.0] * 3, "C": [0.0] * 3}, index=idx)
    return mret, q


def test_buy_and_hold_weights_drift():
    """買入持有：A 翻倍後權重變成 2/3，第 3 個月報酬 = 2/3 × 20% = 13.33%。"""
    mret, q = _drift_fixture()
    spec = PortfolioSpec(n_quantiles=2, horizon=2, monthly_rebalance_within_sleeve=False)
    res = build_calendar_time(q, mret, spec)
    # 第 3 個月有兩個 sleeve：age=1 的（權重已漂移）與 age=0 的（等權）。
    # 漂移 sleeve：2/3×0.20 = 0.13333；新 sleeve：0.5×0.20 = 0.10
    assert res.returns.iloc[2, 1] == pytest.approx((2 / 3 * 0.20 + 0.5 * 0.20) / 2)


def test_monthly_rebalance_resets_weights_to_equal():
    """每月再平衡：權重回到 50/50，第 3 個月兩個 sleeve 都是 10%。"""
    mret, q = _drift_fixture()
    spec = PortfolioSpec(n_quantiles=2, horizon=2, monthly_rebalance_within_sleeve=True)
    res = build_calendar_time(q, mret, spec)
    assert res.returns.iloc[2, 1] == pytest.approx(0.10)


def test_drift_and_rebalance_differ():
    """兩種 sleeve 內再平衡方式必須產生不同結果，否則參數是死的。"""
    mret, q = _drift_fixture()
    drift = build_calendar_time(
        q, mret, PortfolioSpec(n_quantiles=2, horizon=2, monthly_rebalance_within_sleeve=False)
    )
    rebal = build_calendar_time(
        q, mret, PortfolioSpec(n_quantiles=2, horizon=2, monthly_rebalance_within_sleeve=True)
    )
    assert drift.returns.iloc[2, 1] > rebal.returns.iloc[2, 1]
    assert drift.meta["monthly_rebalance_within_sleeve"] is False
    assert rebal.meta["monthly_rebalance_within_sleeve"] is True


def test_sleeve_not_built_when_a_quantile_is_empty():
    """任一分位為空時不建立 sleeve —— 多空價差在那個月沒有定義。"""
    idx = pd.date_range("2010-01-31", periods=4, freq="ME")
    mret = pd.DataFrame({"A": 0.01, "B": 0.01}, index=idx)
    q = pd.DataFrame({"A": [1.0] * 4, "B": [1.0] * 4}, index=idx)  # 分位 0 全空
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=1))
    assert res.returns.isna().all().all()
    assert (res.n_sleeves == 0).all()


def test_horizon_must_be_positive(deterministic):
    mret, q = deterministic
    with pytest.raises(ValueError):
        build_calendar_time(q, mret, PortfolioSpec(n_quantiles=2, horizon=0))


def test_meta_records_legs():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    mret = pd.DataFrame({"A": 0.01, "B": 0.0, "C": -0.01}, index=idx)
    q = pd.DataFrame({"A": 2.0, "B": 1.0, "C": 0.0}, index=idx)
    res = build_calendar_time(q, mret, PortfolioSpec(n_quantiles=3, horizon=1))
    assert res.meta["long_leg"] == 2
    assert res.meta["short_leg"] == 0


# ------------------------------------------------- 重疊報酬對照


def test_overlapping_forward_returns_compounds_correctly():
    """重疊 h 月報酬必須是複利，不是簡單相加。"""
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    mret = pd.DataFrame({"A": 0.10, "B": 0.10}, index=idx)
    q = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
    ov = overlapping_forward_returns(q, mret, horizon=3, n_quantiles=2)
    # 連三個月 +10% → 1.1^3 - 1 = 33.1%
    assert ov[1].dropna().iloc[0] == pytest.approx(1.1**3 - 1)


def test_overlapping_sample_is_smaller_and_autocorrelated():
    """重疊報酬的樣本數比月頻少 h 期，而且有強自相關 —— 問題的根源。

    這裡用**固定的分位成員**，才能隔離出純粹由重疊抽樣造成的自相關。
    理論值：h 期重疊和的一階自相關 ≈ 1 - 1/h，h=12 時約 0.92。
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    cols = [f"T{i}" for i in range(20)]
    mret = pd.DataFrame(rng.standard_normal((200, 20)) * 0.05, index=idx, columns=cols)
    # 成員固定：前 10 檔永遠在高分位，後 10 檔永遠在低分位。
    q = pd.DataFrame(
        np.tile(np.r_[np.ones(10), np.zeros(10)], (200, 1)), index=idx, columns=cols
    )

    ov = overlapping_forward_returns(q, mret, horizon=12, n_quantiles=2)
    ls = (ov[1] - ov[0]).dropna()
    ac1 = float(pd.Series(ls).autocorr(lag=1))
    assert len(ls) < 200
    assert ac1 > 0.7, f"重疊 12 期的一階自相關只有 {ac1:.3f}，應接近 1-1/12 = 0.92"


def test_fast_membership_turnover_masks_overlap_autocorrelation():
    """成員每月洗牌時，重疊造成的自相關會被掩蓋 —— 值得記錄的細節。

    每月重新隨機分組時，相鄰兩期的多空對比權重幾乎獨立，所以即使
    forward 報酬高度重疊，多空序列的自相關仍接近零。這意味著
    「看序列自相關低」**不能**用來證明重疊抽樣沒問題：膨脹的根源是
    抽樣設計，不是觀察到的自相關。真實動能的分位成員是黏著的
    （lookback 越長越黏），所以實務上兩者都在。
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    cols = [f"T{i}" for i in range(20)]
    mret = pd.DataFrame(rng.standard_normal((200, 20)) * 0.05, index=idx, columns=cols)
    sig = pd.DataFrame(rng.standard_normal((200, 20)), index=idx, columns=cols)
    q = pd.DataFrame(np.ceil(sig.rank(axis=1, pct=True) * 2) - 1, index=idx, columns=cols)

    ov = overlapping_forward_returns(q, mret, horizon=12, n_quantiles=2)
    ls = (ov[1] - ov[0]).dropna()
    assert abs(float(pd.Series(ls).autocorr(lag=1))) < 0.4


def test_apply_costs_reduces_returns():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    gross = pd.Series(0.01, index=idx, name="ls")
    turnover = pd.Series(1.0, index=idx)
    net = apply_costs(gross, turnover, one_way_bps=10.0)
    assert np.allclose(net.to_numpy(), 0.01 - 0.001)
