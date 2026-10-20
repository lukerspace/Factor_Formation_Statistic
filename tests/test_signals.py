"""
訊號層測試 —— 重點是原始碼那兩個實作 bug 的回歸測試。
"""

import numpy as np
import pandas as pd
import pytest

from factorlab.config import DataSpec, SignalSpec
from factorlab.data import Panel
from factorlab.signals import (
    assign_quantiles,
    build_signal_panel,
    liquidity_universe,
    momentum_signal,
    winsorize_cross_section,
)


def _panel_from_mclose(mclose: pd.DataFrame, adv: pd.DataFrame | None = None) -> Panel:
    """只有月頻資料的最小 Panel，夠訊號層測試用。"""
    valid = mclose.notna()
    mret = mclose / mclose.shift(1) - 1.0
    mret = mret.where(valid & valid.shift(1).fillna(False))
    if adv is None:
        adv = pd.DataFrame(1e9, index=mclose.index, columns=mclose.columns)
    empty = pd.DataFrame(index=mclose.index, columns=mclose.columns, dtype="float64")
    return Panel(
        close=empty, volume=empty, dret=empty,
        mclose=mclose, mret=mret, dollar_vol=empty,
        adv=adv, valid=valid, spec=DataSpec(),
    )


@pytest.fixture
def ramp_panel():
    """A 每月漲 10%、B 持平、C 每月跌 5%。動能排序完全已知。"""
    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    mclose = pd.DataFrame(
        {
            "A": 100 * 1.10 ** np.arange(24),
            "B": np.full(24, 100.0),
            "C": 100 * 0.95 ** np.arange(24),
        },
        index=idx,
    )
    return _panel_from_mclose(mclose)


# ------------------------------------------------- bug 1：skip 在形成期


def test_momentum_uses_formation_period_not_holding(ramp_panel):
    """skip 必須縮短形成期的尾端，不是把進場往後推。

    A 每月 +10%。lookback=12, skip=0 → P[t]/P[t-12]-1 = 1.1^12 - 1。
    lookback=12, skip=1 → P[t-1]/P[t-12]-1 = 1.1^11 - 1。
    若 skip 被（錯誤地）實作成延後進場，訊號值會與 skip=0 相同。
    """
    s0 = momentum_signal(ramp_panel, lookback=12, skip=0)
    s1 = momentum_signal(ramp_panel, lookback=12, skip=1)
    t = ramp_panel.mclose.index[20]
    assert s0.loc[t, "A"] == pytest.approx(1.10**12 - 1)
    assert s1.loc[t, "A"] == pytest.approx(1.10**11 - 1)
    assert s0.loc[t, "A"] != pytest.approx(s1.loc[t, "A"])


def test_skip_shortens_formation_window_length(ramp_panel):
    """12-2 的形成期長度是 10 個月，不是 12 個月。"""
    s = momentum_signal(ramp_panel, lookback=12, skip=2)
    t = ramp_panel.mclose.index[20]
    assert s.loc[t, "A"] == pytest.approx(1.10**10 - 1)


def test_skip_must_be_less_than_lookback(ramp_panel):
    with pytest.raises(ValueError, match="skip"):
        momentum_signal(ramp_panel, lookback=3, skip=3)
    with pytest.raises(ValueError):
        momentum_signal(ramp_panel, lookback=3, skip=-1)


def test_momentum_ordering_is_correct(ramp_panel):
    s = momentum_signal(ramp_panel, lookback=6, skip=0)
    t = ramp_panel.mclose.index[12]
    assert s.loc[t, "A"] > s.loc[t, "B"] > s.loc[t, "C"]
    assert s.loc[t, "B"] == pytest.approx(0.0)


def test_momentum_does_not_fill_gaps():
    """形成期任一端缺值時訊號必須是 NaN，不可 ffill。"""
    idx = pd.date_range("2010-01-31", periods=14, freq="ME")
    mclose = pd.DataFrame({"A": np.linspace(100, 200, 14)}, index=idx)
    mclose.iloc[0, 0] = np.nan
    panel = _panel_from_mclose(mclose)
    s = momentum_signal(panel, lookback=12, skip=0)
    assert np.isnan(s.iloc[12, 0])


# ------------------------------------------------- bug 2：流動性篩選的時點


def test_liquidity_filter_uses_rebalance_month():
    """篩選必須用再平衡當月的 ADV，不是形成期開始前一個月。

    設計：B 在最後一個月流動性才衝高。用當月 ADV 篩選時，最後一個月
    B 必須入選；若（錯誤地）用 13 個月前的 ADV，B 就不會入選。
    """
    idx = pd.date_range("2010-01-31", periods=14, freq="ME")
    cols = [f"T{i}" for i in range(10)]
    # ADV 各月都是 1e6 * (1..10)，中位數明確，沒有同值造成的邊界歧義。
    adv = pd.DataFrame(
        np.tile(np.arange(1, 11, dtype="float64") * 1e6, (14, 1)),
        index=idx, columns=cols,
    )
    # T0 原本最不流動（1e6，排最後），只有最後一個月衝到最高。
    adv.iloc[-1, adv.columns.get_loc("T0")] = 1e12
    mclose = pd.DataFrame(100.0, index=idx, columns=cols)
    panel = _panel_from_mclose(mclose, adv=adv)

    univ = liquidity_universe(panel, quantile=0.5, min_stocks=2)
    assert univ.iloc[-1]["T0"], "最後一個月 T0 的當月 ADV 最高，必須入選"
    assert not univ.iloc[0]["T0"], "第一個月 T0 的 ADV 最低，不應入選"
    # 若（錯誤地）用 13 個月前的 ADV，最後一個月的 T0 就會被判為不流動。


def test_liquidity_quantile_is_cross_sectional_each_month():
    """門檻必須逐月重算，不可用固定金額。"""
    idx = pd.date_range("2010-01-31", periods=3, freq="ME")
    cols = [f"T{i}" for i in range(10)]
    adv = pd.DataFrame(
        np.vstack([np.arange(1, 11) * 1e6, np.arange(1, 11) * 1e9, np.arange(1, 11) * 1e3]),
        index=idx, columns=cols,
    )
    mclose = pd.DataFrame(100.0, index=idx, columns=cols)
    panel = _panel_from_mclose(mclose, adv=adv)
    univ = liquidity_universe(panel, 0.5, min_stocks=2)
    # 每個月都剛好一半入選，不管絕對金額差了 6 個數量級。
    assert univ.sum(axis=1).tolist() == [5, 5, 5]


def test_liquidity_invalid_quantile():
    idx = pd.date_range("2010-01-31", periods=2, freq="ME")
    panel = _panel_from_mclose(pd.DataFrame(100.0, index=idx, columns=["A"]))
    with pytest.raises(ValueError):
        liquidity_universe(panel, quantile=1.0)


def test_thin_month_is_voided():
    """通過篩選的檔數低於 min_stocks 時，整月作廢。"""
    idx = pd.date_range("2010-01-31", periods=2, freq="ME")
    adv = pd.DataFrame({"A": [1e9, 1e9], "B": [1e8, 1e8]}, index=idx)
    mclose = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    panel = _panel_from_mclose(mclose, adv=adv)
    univ = liquidity_universe(panel, 0.5, min_stocks=50)
    assert not univ.to_numpy().any()


# ------------------------------------------------- 分位分組


def test_assign_quantiles_ties_share_a_quantile_and_never_mislabel():
    """大量同值時，分位數會少於 n —— 這是資料的性質，不是 bug。

    關鍵是**失敗的方式**要安全：同值共用同一個分位，而且分位索引
    仍是訊號的單調函數，不會像 qcut(duplicates='drop') 那樣把第 7
    分位重新貼上 "Q10" 的標籤。
    """
    idx = pd.date_range("2010-01-31", periods=2, freq="ME")
    # 100 檔：兩端各 30 檔連續值，中間 40 檔訊號完全相同。
    # 同值放在中段，極端分位仍有成員，所以這個月不會被作廢。
    vals = np.concatenate(
        [np.linspace(-1.0, -0.1, 30), np.zeros(40), np.linspace(0.1, 1.0, 30)]
    )
    sig = pd.DataFrame([vals, vals], index=idx, columns=[f"T{i}" for i in range(100)])
    q = assign_quantiles(sig, n_quantiles=10, min_stocks=50)

    # 40 檔同值全部落在同一個分位（相同訊號得到相同處置）。
    tied = q.iloc[0, 30:70]
    assert tied.nunique() == 1
    # 分位索引對訊號單調不遞減 —— 標籤與分布位置的對應沒有錯亂。
    assert (np.diff(q.iloc[0].to_numpy()) >= 0).all()
    # 兩端都有成員，Q10-Q1 有定義；但總組數因同值而少於 10。
    assert q.iloc[0].min() == 0
    assert q.iloc[0].max() == 9
    assert q.iloc[0].nunique() < 10


def test_assign_quantiles_voids_month_when_extreme_quantile_empty():
    """極端分位為空時整月作廢 —— 那個月沒有「十分位兩端」可言。"""
    idx = pd.date_range("2010-01-31", periods=1, freq="ME")
    # 80 檔同值在最低端，會把 Q1~Q8 全部擠空。
    vals = np.concatenate([np.zeros(80), np.arange(1, 21, dtype="float64")])
    sig = pd.DataFrame([vals], index=idx, columns=[f"T{i}" for i in range(100)])
    q = assign_quantiles(sig, n_quantiles=10, min_stocks=50)
    assert q.isna().all().all()


def test_assign_quantiles_keeps_month_when_signal_is_continuous():
    """真實資料（連續訊號、無同值）下兩端都有成員，整月保留。"""
    idx = pd.date_range("2010-01-31", periods=1, freq="ME")
    vals = np.linspace(-0.5, 0.5, 200)
    sig = pd.DataFrame([vals], index=idx, columns=[f"T{i}" for i in range(200)])
    q = assign_quantiles(sig, 10, min_stocks=50)
    assert q.iloc[0].nunique() == 10
    assert (q.iloc[0] == 0).sum() == 20
    assert (q.iloc[0] == 9).sum() == 20


def test_assign_quantiles_is_monotone_in_signal():
    idx = pd.date_range("2010-01-31", periods=1, freq="ME")
    vals = np.arange(100, dtype="float64")
    sig = pd.DataFrame([vals], index=idx, columns=[f"T{i}" for i in range(100)])
    q = assign_quantiles(sig, 10, min_stocks=50).iloc[0]
    assert q.iloc[0] == 0
    assert q.iloc[-1] == 9
    assert (np.diff(q.to_numpy()) >= 0).all()


def test_assign_quantiles_voids_thin_months():
    idx = pd.date_range("2010-01-31", periods=1, freq="ME")
    sig = pd.DataFrame([np.arange(10.0)], index=idx, columns=[f"T{i}" for i in range(10)])
    q = assign_quantiles(sig, 10, min_stocks=50)
    assert q.isna().all().all()


def test_assign_quantiles_rejects_tiny_n():
    sig = pd.DataFrame([[1.0, 2.0]], index=pd.date_range("2010-01-31", periods=1, freq="ME"))
    with pytest.raises(ValueError):
        assign_quantiles(sig, n_quantiles=1)


# ------------------------------------------------- 其他


def test_winsorize_clips_cross_section():
    idx = pd.date_range("2010-01-31", periods=1, freq="ME")
    vals = np.concatenate([[-100.0], np.zeros(98), [100.0]])
    df = pd.DataFrame([vals], index=idx)
    out = winsorize_cross_section(df, 0.05)
    assert out.to_numpy().min() > -100.0
    assert out.to_numpy().max() < 100.0


def test_winsorize_zero_is_noop():
    df = pd.DataFrame([[1.0, 2.0, 3.0]])
    assert winsorize_cross_section(df, 0.0).equals(df)


def test_build_signal_panel_requires_tradable_next_month(ramp_panel):
    """最後一個月沒有下月報酬，不應進入 universe。"""
    sp = build_signal_panel(ramp_panel, SignalSpec(lookback=6, skip=0, min_stocks=2))
    assert sp.n_stocks.iloc[-1] == 0
