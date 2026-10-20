"""風險因子基準。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.config import DataSpec
from factorlab.data import Panel
from factorlab.benchmarks import FACTOR_NAMES, build_factors, factor_summary


@pytest.fixture
def synthetic_panel():
    """80 檔、180 個月的合成面板，含日頻資料以便算已實現波動。"""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2008-01-01", periods=180 * 21)
    cols = [f"T{i}" for i in range(80)]
    dret = pd.DataFrame(rng.standard_normal((len(days), 80)) * 0.015, index=days, columns=cols)
    close = 100 * (1 + dret).cumprod()
    volume = pd.DataFrame(
        rng.lognormal(12, 1.0, (len(days), 80)), index=days, columns=cols
    )
    dollar_vol = close * volume

    grouper = close.index.to_period("M")
    month_ends = pd.DatetimeIndex(
        pd.Series(close.index, index=grouper).groupby(level=0).max().values
    ).sort_values()
    valid = pd.DataFrame(True, index=month_ends, columns=cols)
    mclose = close.groupby(grouper).last()
    mclose.index = month_ends
    adv = dollar_vol.groupby(grouper).mean()
    adv.index = month_ends
    mret = mclose / mclose.shift(1) - 1.0

    return Panel(
        close=close, volume=volume, dret=dret, mclose=mclose, mret=mret,
        dollar_vol=dollar_vol, adv=adv, valid=valid, spec=DataSpec(),
    )


def test_build_factors_returns_all_four(synthetic_panel):
    f = build_factors(synthetic_panel, min_stocks=20)
    assert list(f.columns) == list(FACTOR_NAMES)
    assert len(f) == len(synthetic_panel.mret)


def test_factors_are_mostly_populated(synthetic_panel):
    f = build_factors(synthetic_panel, min_stocks=20)
    for c in FACTOR_NAMES:
        assert f[c].notna().mean() > 0.9, c


def test_market_factor_is_equal_weighted_universe_return(synthetic_panel):
    """MKT 必須是 universe 的等權報酬，且用上一期的 universe。"""
    from factorlab.signals import liquidity_universe

    panel = synthetic_panel
    f = build_factors(panel, min_stocks=20)
    univ = liquidity_universe(panel, 0.5, 20)
    manual = panel.mret.where(univ.shift(1).fillna(False)).mean(axis=1)
    both = pd.concat([manual, f["MKT"]], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_factors_have_no_lookahead(synthetic_panel):
    """特徵在 m-1 月末觀察、賺 m 月報酬：第一個月必須是 NaN。"""
    f = build_factors(synthetic_panel, min_stocks=20)
    assert f.iloc[0].isna().all()


def test_factors_are_not_perfectly_collinear(synthetic_panel):
    """四個因子不可高度共線，否則迴歸的截距無法識別。"""
    f = build_factors(synthetic_panel, min_stocks=20).dropna()
    corr = f.corr().to_numpy()
    off = corr[~np.eye(4, dtype=bool)]
    assert np.abs(off).max() < 0.95
    assert np.linalg.matrix_rank(f.to_numpy()) == 4


def test_factor_summary_columns(synthetic_panel):
    f = build_factors(synthetic_panel, min_stocks=20)
    s = factor_summary(f)
    assert len(s) == 4
    assert {"factor", "mean_m", "t_hac", "sharpe", "vol_ann"}.issubset(s.columns)
