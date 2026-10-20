"""
factorlab.stats.spa — White Reality Check 與 Hansen SPA 檢定。

跟 multiple_testing 的差別
--------------------------
Holm / BY 是對「一組 p 值」做事後校正，不知道策略之間的相關結構。
SPA 直接在 bootstrap 裡重抽整個 (T × L) 的報酬矩陣，相關結構自動被保留，
因此檢定力明顯高於 Bonferroni 家族 —— 這很重要，因為本研究的 32 個策略
相關係數高達 0.7~0.95，Bonferroni 在這種情況下過度保守。

檢定的問題是：「這一整組策略裡，最好的那個，真的比基準（報酬為零）好嗎？」

    H0: max_k E[d_k] <= 0
    統計量 T_SPA = max(0, max_k √T·d̄_k / ω̂_k)

Hansen (2005) 的關鍵改良是「移心」。White (2000) 的 Reality Check 把所有
策略都移心到自己的樣本平均，表現很差的策略也被當成虛無下的候選者，
稀釋了分配、讓檢定過度保守。Hansen 只保留「樣本表現沒有差到不可能是
最好」的策略：門檻 A_k = ω̂_k·√(2·loglog T / T)。

本模組回傳三個 p 值（Hansen 的記號）：
- p_upper  : 所有策略都移心 = White RC，最保守（上界）
- p_consistent : Hansen 的一致 p 值，主結果
- p_lower  : 只保留樣本平均非負的策略，最寬鬆（下界）
三個一起看：若連 p_lower 都不顯著，就沒有任何解讀空間。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bootstrap import stationary_bootstrap_indices
from .hac import long_run_variance


@dataclass
class SPAResult:
    """SPA / Reality Check 結果。"""

    statistic: float
    p_lower: float
    p_consistent: float
    p_upper: float
    best_strategy: str
    best_mean: float
    best_studentized: float
    n_strategies: int
    nobs: int
    reps: int
    n_retained_consistent: int

    def as_dict(self, prefix: str = "spa_") -> dict:
        return {
            f"{prefix}statistic": self.statistic,
            f"{prefix}p_lower": self.p_lower,
            f"{prefix}p_consistent": self.p_consistent,
            f"{prefix}p_upper_rc": self.p_upper,
            f"{prefix}best": self.best_strategy,
            f"{prefix}best_mean_m": self.best_mean,
            f"{prefix}best_t": self.best_studentized,
            f"{prefix}n_strategies": self.n_strategies,
            f"{prefix}nobs": self.nobs,
            f"{prefix}n_retained": self.n_retained_consistent,
        }


def spa_test(
    performance: pd.DataFrame,
    reps: int = 2000,
    block: float = 6.0,
    seed: int | None = None,
    benchmark: float = 0.0,
) -> SPAResult:
    """Hansen (2005) Superior Predictive Ability 檢定。

    Parameters
    ----------
    performance : DataFrame (T × L)，每欄一個策略的月報酬。
                  只保留所有策略都有觀察值的月份（SPA 需要共同樣本）。
    benchmark   : 基準報酬（純量）。預設 0 = 檢定「有沒有任何策略賺錢」。
    """
    data = performance.dropna(how="any")
    t, l = data.shape
    if t < 20:
        raise ValueError(f"SPA 需要足夠長的共同樣本，目前只有 {t} 期")
    if l < 1:
        raise ValueError("至少需要一個策略")

    d = data.to_numpy(dtype="float64") - benchmark
    d_bar = d.mean(axis=0)

    # ω̂_k：每個策略損益差的 HAC 長期變異數平方根。
    omega = np.empty(l, dtype="float64")
    for k in range(l):
        s, _, failed = long_run_variance(d[:, k], lags=None, kernel="bartlett")
        if failed or not np.isfinite(s) or s <= 0:
            s = float(np.var(d[:, k], ddof=1))
        omega[k] = np.sqrt(max(s, 1e-18))

    studentized = np.sqrt(t) * d_bar / omega
    stat = float(max(0.0, studentized.max()))
    best = int(np.argmax(studentized))

    # Hansen 的移心門檻 A_k = ω̂_k·√(2·loglog T / T)。
    loglog = np.log(np.log(t)) if t > np.e else 1.0
    a_k = omega * np.sqrt(max(2.0 * loglog, 0.0) / t)

    g_upper = d_bar.copy()                              # White RC
    g_consistent = np.where(d_bar >= -a_k, d_bar, 0.0)  # Hansen 一致版
    g_lower = np.where(d_bar >= 0.0, d_bar, 0.0)        # 最寬鬆

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, reps, block, rng)
    boot_mean = d[idx].mean(axis=1)                     # (reps, L)

    def _pval(g: np.ndarray) -> float:
        # 統計量為 0 表示連表現最好的策略，其 studentized 平均數都 <= 0，
        # 也就是完全沒有證據。此時 p 值必須是 1。
        #
        # 不能直接套 p = (1/B)·Σ1{T*_b > T_SPA}：因為 T*_b >= 0 而 T_SPA = 0，
        # 指示函數 1{0 > 0} 恆為 0，會算出 p = 0 —— 於是「所有策略都在虧錢」
        # 反而被判定成「拒絕虛無、有策略優於零」，結論剛好相反。
        if stat <= 0.0:
            return 1.0
        z = np.sqrt(t) * (boot_mean - g[None, :]) / omega[None, :]
        tb = np.maximum(0.0, z.max(axis=1))
        return float((tb > stat).mean())

    return SPAResult(
        statistic=stat,
        p_lower=_pval(g_lower),
        p_consistent=_pval(g_consistent),
        p_upper=_pval(g_upper),
        best_strategy=str(data.columns[best]),
        best_mean=float(d_bar[best]),
        best_studentized=float(studentized[best]),
        n_strategies=l,
        nobs=t,
        reps=reps,
        n_retained_consistent=int((d_bar >= -a_k).sum()),
    )


def reality_check(
    performance: pd.DataFrame,
    reps: int = 2000,
    block: float = 6.0,
    seed: int | None = None,
) -> SPAResult:
    """White (2000) Reality Check = SPA 的 p_upper。回傳同一個結果物件。"""
    return spa_test(performance, reps=reps, block=block, seed=seed)
