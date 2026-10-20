"""交易成本。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.config import CostSpec
from factorlab.costs import analyze, breakeven_bps, net_returns


@pytest.fixture
def series():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    gross = pd.Series(0.01, index=idx, name="ls")
    turnover = pd.Series(2.0, index=idx)   # 多空兩條腿各 100%
    return gross, turnover


def test_one_way_bps_sums_components():
    assert CostSpec(commission_bps=1.0, half_spread_bps=5.0).one_way_bps == 6.0


def test_net_returns_subtract_cost(series):
    gross, turnover = series
    net = net_returns(gross, turnover, CostSpec(commission_bps=1.0, half_spread_bps=5.0))
    assert net.iloc[0] == pytest.approx(0.01 - 2.0 * 6e-4)


def test_disabled_costs_are_a_noop(series):
    gross, turnover = series
    net = net_returns(gross, turnover, CostSpec(enabled=False))
    assert np.allclose(net.to_numpy(), gross.to_numpy())


def test_breakeven_bps_zeroes_net_return(series):
    gross, turnover = series
    be = breakeven_bps(gross, turnover)
    net = net_returns(gross, turnover, CostSpec(commission_bps=be, half_spread_bps=0.0))
    assert net.mean() == pytest.approx(0.0, abs=1e-12)


def test_breakeven_is_higher_for_lower_turnover():
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    gross = pd.Series(0.01, index=idx)
    assert breakeven_bps(gross, pd.Series(0.5, index=idx)) > breakeven_bps(
        gross, pd.Series(2.0, index=idx)
    )


def test_breakeven_nan_for_zero_turnover():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    assert np.isnan(breakeven_bps(pd.Series(0.01, index=idx), pd.Series(0.0, index=idx)))


def test_analyze_reports_consistent_numbers(series):
    gross, turnover = series
    rep = analyze(gross, turnover, CostSpec(commission_bps=1.0, half_spread_bps=5.0))
    assert rep.avg_monthly_turnover_one_way == pytest.approx(2.0)
    assert rep.annual_turnover_one_way == pytest.approx(24.0)
    assert rep.annual_cost == pytest.approx(24.0 * 6e-4)
    assert rep.net_mean_m < rep.gross_mean_m
    assert rep.breakeven_one_way_bps == pytest.approx(50.0)


def test_cost_drag_scales_with_turnover():
    """週轉放大 10 倍，年化成本也放大 10 倍，淨 Sharpe 下降。

    gross 必須有變異 —— 常數報酬序列的 Sharpe 沒有定義。
    """
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    gross = pd.Series(
        np.random.default_rng(1).standard_normal(60) * 0.04 + 0.01, index=idx
    )
    spec = CostSpec(commission_bps=1.0, half_spread_bps=5.0)
    lo = analyze(gross, pd.Series(0.2, index=idx), spec)
    hi = analyze(gross, pd.Series(2.0, index=idx), spec)
    assert hi.annual_cost == pytest.approx(10 * lo.annual_cost)
    assert np.isfinite(lo.net_sharpe) and np.isfinite(hi.net_sharpe)
    assert hi.net_sharpe < lo.net_sharpe < lo.gross_sharpe


def test_sharpe_is_nan_for_constant_series():
    """常數報酬序列的 Sharpe 沒有定義，必須回 nan 而不是巨大的數字。"""
    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    rep = analyze(pd.Series(0.01, index=idx), pd.Series(1.0, index=idx), CostSpec())
    assert np.isnan(rep.gross_sharpe)
    assert np.isnan(rep.net_sharpe)
