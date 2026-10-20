"""研究規格。重點是 n_trials —— 多重檢定的分母。"""

import json

import pytest

from factorlab.config import (
    CostSpec,
    DataSpec,
    SignalSpec,
    StudySpec,
    strategy_tag,
)


def test_n_trials_is_product_of_grid():
    spec = StudySpec()
    assert spec.n_trials == 4 * 2 * 4 == 32
    assert len(spec.grid()) == spec.n_trials


def test_n_trials_tracks_custom_grid():
    spec = StudySpec(lookbacks=(6, 12), skips=(1,), horizons=(1, 3, 6))
    assert spec.n_trials == 6
    assert len(spec.grid()) == 6


def test_signal_grid_excludes_horizon():
    spec = StudySpec()
    assert len(spec.signal_grid()) == 8


def test_strategy_tag_encodes_skip_as_lookback_minus_skip():
    """沿用學界的 12-2 記法，skip 的語意不會被誤讀成持有期。"""
    assert strategy_tag(12, 1, 1) == "mom12-1_h1"
    assert strategy_tag(12, 0, 6) == "mom12_h6"


def test_strategy_tags_are_unique_across_grid():
    spec = StudySpec()
    tags = {strategy_tag(*combo) for combo in spec.grid()}
    assert len(tags) == spec.n_trials


def test_signal_spec_inherits_study_settings():
    spec = StudySpec(liquidity_quantile=0.7, min_stocks=80)
    s = spec.signal_spec(6, 1)
    assert s.lookback == 6 and s.skip == 1
    assert s.liquidity_quantile == 0.7
    assert s.min_stocks == 80


def test_portfolio_spec_carries_horizon_and_quantiles():
    spec = StudySpec(n_quantiles=5)
    p = spec.portfolio_spec(3)
    assert p.horizon == 3 and p.n_quantiles == 5


def test_to_json_is_valid_and_round_trips():
    parsed = json.loads(StudySpec().to_json())
    assert parsed["data"]["history"] == "long"
    assert parsed["stats"]["hlz_tstat_threshold"] == 3.0


def test_specs_are_frozen():
    with pytest.raises(Exception):
        DataSpec().history = "3y"
    with pytest.raises(Exception):
        SignalSpec().skip = 5


def test_cost_spec_one_way_bps():
    assert CostSpec(commission_bps=2.0, half_spread_bps=8.0).one_way_bps == 10.0


def test_default_skips_include_zero_and_one():
    """skip=0 與 skip=1 都要掃，才能看出短期反轉的影響。"""
    assert set(StudySpec().skips) == {0, 1}
