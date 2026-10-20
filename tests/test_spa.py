"""Hansen SPA 與 White Reality Check。"""

import numpy as np
import pandas as pd
import pytest

from factorlab.stats.spa import reality_check, spa_test


def _strategies(means, n=215, sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {f"s{i}": rng.standard_normal(n) * sd + m for i, m in enumerate(means)},
        index=idx,
    )


def test_does_not_reject_when_all_strategies_are_noise():
    """全是噪音時，不該宣稱「最好的那個優於零」。"""
    df = _strategies(np.zeros(32), seed=1)
    res = spa_test(df, reps=1000, seed=1)
    assert res.p_consistent > 0.05
    assert res.p_upper >= res.p_consistent - 1e-9
    assert res.n_strategies == 32


def test_rejects_when_one_strategy_is_genuinely_strong():
    means = np.zeros(32)
    means[7] = 0.020
    df = _strategies(means, seed=2)
    res = spa_test(df, reps=1000, seed=2)
    assert res.p_consistent < 0.05
    assert res.best_strategy == "s7"


def test_pvalue_ordering_lower_le_consistent_le_upper():
    """Hansen 的三個 p 值必須滿足 p_lower <= p_consistent <= p_upper。"""
    means = np.r_[0.006, np.zeros(20) - 0.004]
    df = _strategies(means, seed=3)
    res = spa_test(df, reps=1500, seed=3)
    assert res.p_lower <= res.p_consistent + 1e-9
    assert res.p_consistent <= res.p_upper + 1e-9


def test_many_bad_strategies_dilute_white_reality_check():
    """加入一堆明顯很差的策略，RC 的 p 值會被稀釋，Hansen 版不會。

    這是 Hansen (2005) 對 White (2000) 的核心批評：RC 把表現很差的
    策略也當成虛無下的候選者，稀釋了分配、讓檢定過度保守。
    """
    good = np.r_[0.010]
    df_small = _strategies(good, seed=4)
    df_big = pd.concat(
        [df_small, _strategies(np.full(40, -0.030), seed=5).add_prefix("bad_")], axis=1
    )
    small = spa_test(df_small, reps=1500, seed=4)
    big = spa_test(df_big, reps=1500, seed=4)
    # 加入垃圾策略後，RC 的 p 值變大（更保守）。
    assert big.p_upper > small.p_upper
    # Hansen 一致版把它們移心到 0，受到的稀釋小得多。
    assert big.p_consistent - small.p_consistent < big.p_upper - small.p_upper
    assert big.n_retained_consistent < big.n_strategies


def test_requires_long_enough_sample():
    with pytest.raises(ValueError, match="共同樣本"):
        spa_test(_strategies(np.zeros(5), n=15), reps=200)


def test_reality_check_is_spa_upper():
    df = _strategies(np.zeros(10), seed=6)
    a = spa_test(df, reps=800, seed=7)
    b = reality_check(df, reps=800, seed=7)
    assert a.p_upper == pytest.approx(b.p_upper)


def test_all_losing_strategies_give_pvalue_one():
    """所有策略都在虧錢時，統計量為 0，三個 p 值都必須是 1。

    回歸測試：p = (1/B)·Σ1{T*_b > T_SPA} 在 T_SPA = 0 時會算出 0，
    把「全部虧錢」誤判成「拒絕虛無、有策略優於零」—— 結論剛好相反。
    """
    df = _strategies(np.full(10, -0.02), seed=8)
    res = spa_test(df, reps=500, seed=8)
    assert res.statistic == 0.0
    assert res.p_lower == 1.0
    assert res.p_consistent == 1.0
    assert res.p_upper == 1.0
    assert res.best_studentized < 0
