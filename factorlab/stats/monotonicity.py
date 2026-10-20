"""
factorlab.stats.monotonicity — 分位報酬的單調性檢定。

為什麼需要
----------
原始碼只回報 Q10-Q1 與 Q9-Q2，其餘分位的資訊全部丟掉。這有兩個問題：

1. Q10-Q1 顯著**不等於**訊號有排序能力。只要 Q1 被幾檔崩跌股拉下去，
   差值就會顯著，中間八個分位可以完全是平的或亂的。真正的因子應該讓
   報酬隨分位單調遞增。
2. Q10-Q1 是兩個最極端分位的差，恰恰是離群值影響最大、流動性最差、
   實務上最難執行的那一段。

Patton-Timmermann (2010) 的 MR 檢定直接檢定單調性：

    H0: Δ_i <= 0 對所有 i        （沒有單調遞增關係）
    H1: Δ_i >  0 對所有 i        （完全單調遞增）
    統計量 J = min_i Δ̂_i，Δ̂_i = μ̂_{i+1} - μ̂_i

取 **min** 是關鍵：只有在「每一個相鄰差都為正」時 J 才大。這是一個刻意
嚴格的檢定 —— 它拒絕虛無，意味著整條分位曲線都站得住，不是兩端的把戲。

p 值用定態 bootstrap 求：J*_b = min_i(Δ̂*_{b,i} - Δ̂_i) 是把虛無推到
邊界 (Δ = 0) 後的重抽統計量，p = P(J* > J)。

另外提供兩個比較寬鬆的對照：
- 分位序號與分位平均報酬的 Spearman 等級相關（bootstrap p 值）
- 相鄰差全為正的個數（純描述，不是檢定）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .bootstrap import stationary_bootstrap_indices


@dataclass
class MRResult:
    """Patton-Timmermann MR 檢定結果。"""

    direction: str
    statistic: float
    pvalue: float
    deltas: np.ndarray
    n_positive: int
    n_deltas: int
    nobs: int
    reps: int
    spearman: float
    spearman_p: float
    top_minus_bottom: float

    @property
    def monotone(self) -> bool:
        """所有相鄰差是否同向（純描述，不代表統計顯著）。"""
        return self.n_positive == self.n_deltas

    def as_dict(self, prefix: str = "mr_") -> dict:
        return {
            f"{prefix}direction": self.direction,
            f"{prefix}J": self.statistic,
            f"{prefix}p": self.pvalue,
            f"{prefix}n_positive_deltas": self.n_positive,
            f"{prefix}n_deltas": self.n_deltas,
            f"{prefix}all_same_sign": self.monotone,
            f"{prefix}spearman": self.spearman,
            f"{prefix}spearman_p": self.spearman_p,
            f"{prefix}top_minus_bottom_m": self.top_minus_bottom,
            f"{prefix}nobs": self.nobs,
        }


def mr_test(
    quantile_returns: pd.DataFrame,
    direction: str = "up",
    reps: int = 2000,
    block: float = 6.0,
    seed: int | None = None,
) -> MRResult:
    """Patton-Timmermann 單調性檢定。

    Parameters
    ----------
    quantile_returns : DataFrame (T × N)，欄位**必須已按分位由低到高排序**。
    direction        : "up" 檢定遞增，"down" 檢定遞減。
    """
    if direction not in {"up", "down"}:
        raise ValueError("direction 只接受 'up' / 'down'")

    data = quantile_returns.dropna(how="any")
    t, n = data.shape
    if n < 3:
        raise ValueError("至少需要 3 個分位才談得上單調性")
    if t < 12:
        raise ValueError(f"觀察值只有 {t} 期，bootstrap 單調性檢定不可靠")

    arr = data.to_numpy(dtype="float64")
    sign = 1.0 if direction == "up" else -1.0

    mu = arr.mean(axis=0)
    deltas = sign * np.diff(mu)
    stat = float(deltas.min())

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, reps, block, rng)
    # 重抽平均數 → 重抽相鄰差 → 移心到虛無邊界。
    boot_mu = arr[idx].mean(axis=1)              # (reps, N)
    boot_d = sign * np.diff(boot_mu, axis=1)     # (reps, N-1)
    boot_j = (boot_d - deltas[None, :]).min(axis=1)
    pval = float((boot_j > stat).mean())

    # 對照一：分位序號 vs 分位平均報酬的 Spearman 等級相關。
    ranks = np.arange(n, dtype="float64")
    sp = sps.spearmanr(ranks, sign * mu)
    boot_sp = np.array(
        [sps.spearmanr(ranks, sign * boot_mu[b]).statistic for b in range(min(reps, 500))]
    )
    boot_sp = boot_sp[np.isfinite(boot_sp)]
    sp_stat = float(sp.statistic)
    if boot_sp.size:
        centered = boot_sp - boot_sp.mean()
        sp_p = float((centered >= sp_stat).mean())
    else:
        sp_p = float("nan")

    return MRResult(
        direction=direction,
        statistic=stat,
        pvalue=pval,
        deltas=deltas,
        n_positive=int((deltas > 0).sum()),
        n_deltas=int(deltas.size),
        nobs=t,
        reps=reps,
        spearman=sp_stat,
        spearman_p=sp_p,
        top_minus_bottom=float(sign * (mu[-1] - mu[0])),
    )


def quantile_profile(quantile_returns: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """每個分位的平均報酬、t 值、Sharpe —— 給報告畫分位曲線用。"""
    from .hac import hac_mean_test

    rows = []
    for col in quantile_returns.columns:
        s = quantile_returns[col].dropna()
        res = hac_mean_test(s)
        sd = float(s.std(ddof=1))
        rows.append(
            {
                "quantile": col,
                "nobs": int(len(s)),
                "mean_m": float(s.mean()),
                "mean_ann": float((1.0 + s.mean()) ** periods_per_year - 1.0),
                "t_hac": res.tstat,
                "p_hac": res.pvalue,
                "vol_ann": sd * np.sqrt(periods_per_year),
                "sharpe": float(s.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
