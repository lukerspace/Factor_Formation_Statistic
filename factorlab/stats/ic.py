"""
factorlab.stats.ic — 資訊係數（Information Coefficient）。

分位投組的 Q10-Q1 只用到訊號分布的兩端，中間 80% 的股票對結論毫無貢獻。
IC 是訊號與下期報酬的橫斷面相關係數，用到全部股票，因此對「訊號到底有
沒有排序能力」這個問題的統計效率高得多 —— 同樣的資料，IC 的檢定力通常
明顯高於 Q10-Q1。

主要看 Spearman 等級相關（對離群值免疫，而且與分位分組的邏輯一致），
Pearson 當對照。IC 的時間序列平均除以其標準差就是 IC 的 IR；
t 值一律用 Newey-West（IC 序列有自相關，特別是 lookback 長的訊號）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .hac import hac_mean_test


@dataclass
class ICResult:
    """IC 檢定結果。"""

    method: str
    mean_ic: float
    std_ic: float
    ic_ir: float
    tstat: float
    pvalue: float
    nobs_months: int
    hit_rate: float
    avg_n_stocks: float
    lags: int
    series: pd.Series

    def as_dict(self, prefix: str = "ic_") -> dict:
        return {
            f"{prefix}method": self.method,
            f"{prefix}mean": self.mean_ic,
            f"{prefix}std": self.std_ic,
            f"{prefix}ir": self.ic_ir,
            f"{prefix}t": self.tstat,
            f"{prefix}p": self.pvalue,
            f"{prefix}months": self.nobs_months,
            f"{prefix}positive_rate": self.hit_rate,
            f"{prefix}avg_n_stocks": self.avg_n_stocks,
            f"{prefix}hac_lags": self.lags,
        }


def information_coefficient(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_stocks: int = 50,
    lags: int | None = None,
) -> ICResult:
    """逐月橫斷面 IC，再對 IC 序列做 HAC 平均數檢定。

    `fwd_returns` 必須已由呼叫端對齊成「下期報酬」。
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("method 只接受 'spearman' / 'pearson'")

    months = signal.index.intersection(fwd_returns.index)
    cols = signal.columns.intersection(fwd_returns.columns)
    sig = signal.loc[months, cols]
    fwd = fwd_returns.loc[months, cols]

    vals, ns, idx = [], [], []
    for m in months:
        s = sig.loc[m].to_numpy(dtype="float64")
        r = fwd.loc[m].to_numpy(dtype="float64")
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < min_stocks:
            continue
        s_ok, r_ok = s[ok], r[ok]
        if np.std(s_ok) <= 0 or np.std(r_ok) <= 0:
            continue
        if method == "spearman":
            c = sps.spearmanr(s_ok, r_ok).statistic
        else:
            c = np.corrcoef(s_ok, r_ok)[0, 1]
        if not np.isfinite(c):
            continue
        vals.append(float(c))
        ns.append(int(ok.sum()))
        idx.append(m)

    if not vals:
        raise ValueError("IC 沒有任何月份達到最低股票數要求")

    series = pd.Series(vals, index=pd.DatetimeIndex(idx), name=f"ic_{method}")
    res = hac_mean_test(series, lags=lags)
    sd = float(series.std(ddof=1))

    return ICResult(
        method=method,
        mean_ic=float(series.mean()),
        std_ic=sd,
        ic_ir=float(series.mean() / sd) if sd > 0 else float("nan"),
        tstat=res.tstat,
        pvalue=res.pvalue,
        nobs_months=len(series),
        hit_rate=float((series > 0).mean()),
        avg_n_stocks=float(np.mean(ns)),
        lags=res.lags,
        series=series,
    )
