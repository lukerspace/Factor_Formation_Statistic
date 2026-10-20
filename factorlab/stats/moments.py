"""
factorlab.stats.moments — 描述統計、Sharpe 與其標準誤。

重點在 `sharpe_se`：Sharpe 是比率，它的標準誤在報酬非常態時不是
1/√T。Lo (2002) 與 Mertens (2002) 給出考慮偏態與峰態的修正：

    Var(SR) ≈ (1/T)·[1 + SR²/2 - SR·γ3 + SR²·(γ4-3)/4]

動能的多空組合是出名的負偏態（崩盤風險），負偏態會讓真實標準誤
**比 1/√T 大**，也就是天真做法會高估 Sharpe 的顯著性。這跟重疊報酬
的問題是獨立的兩層膨脹，報告裡兩個都要攤開。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS_PER_YEAR = 12


@dataclass
class Moments:
    nobs: int
    mean_m: float
    std_m: float
    mean_ann: float
    vol_ann: float
    sharpe: float
    sharpe_se: float
    sharpe_se_iid: float
    sharpe_t: float
    skew: float
    excess_kurtosis: float
    hit_rate: float
    max_drawdown: float
    sortino: float
    cum_return: float
    jarque_bera_p: float

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def max_drawdown(returns) -> float:
    """以複利淨值計算的最大回撤（負值）。"""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def sharpe_se(returns, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Lo (2002) / Mertens (2002) 的年化 Sharpe 標準誤（含偏態與峰態修正）。"""
    r = pd.Series(returns).dropna().to_numpy(dtype="float64")
    t = r.size
    if t < 8:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    sr = r.mean() / sd  # 期頻 Sharpe
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))
    var = (1.0 + 0.5 * sr**2 - sr * g3 + 0.25 * sr**2 * (g4 - 3.0)) / t
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(np.sqrt(var) * np.sqrt(periods_per_year))


def _empty_moments(nobs: int, mean: float = float("nan")) -> Moments:
    """樣本太短或零變異時的回傳值。逐欄命名而不是靠位置傳 15 個 nan ——
    位置傳參在欄位增減時會靜默錯位（這裡原本就漏掉一個欄位）。"""
    nan = float("nan")
    return Moments(
        nobs=nobs, mean_m=mean, std_m=nan, mean_ann=nan, vol_ann=nan,
        sharpe=nan, sharpe_se=nan, sharpe_se_iid=nan, sharpe_t=nan,
        skew=nan, excess_kurtosis=nan, hit_rate=nan, max_drawdown=nan,
        sortino=nan, cum_return=nan, jarque_bera_p=nan,
    )


def describe(returns, periods_per_year: int = MONTHS_PER_YEAR) -> Moments:
    """單一報酬序列的完整描述統計。"""
    r = pd.Series(returns).dropna().to_numpy(dtype="float64")
    t = r.size
    nan = float("nan")
    if t < 2:
        return _empty_moments(t, float(r[0]) if t == 1 else nan)

    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        # 零變異：Sharpe、偏態、峰態都沒有定義。直接回傳而不讓 scipy
        # 在近乎相同的數列上發 catastrophic-cancellation 警告。
        return _empty_moments(t, mu)
    sr = float(mu / sd * np.sqrt(periods_per_year))
    se = sharpe_se(r, periods_per_year)
    se_iid = float(np.sqrt(periods_per_year / t)) if t else nan

    downside = r[r < 0]
    dd_std = float(downside.std(ddof=1)) if downside.size > 1 else nan
    sortino = float(mu / dd_std * np.sqrt(periods_per_year)) if dd_std and dd_std > 0 else nan

    jb_p = nan
    if t >= 8:
        try:
            jb_p = float(sps.jarque_bera(r).pvalue)
        except Exception:
            jb_p = nan

    return Moments(
        nobs=t,
        mean_m=mu,
        std_m=sd,
        mean_ann=float((1.0 + mu) ** periods_per_year - 1.0),
        vol_ann=float(sd * np.sqrt(periods_per_year)),
        sharpe=sr,
        sharpe_se=se,
        sharpe_se_iid=se_iid,
        sharpe_t=float(sr / se) if se and np.isfinite(se) and se > 0 else nan,
        skew=float(sps.skew(r, bias=False)) if t >= 3 else nan,
        excess_kurtosis=float(sps.kurtosis(r, fisher=True, bias=False)) if t >= 4 else nan,
        hit_rate=float((r > 0).mean()),
        max_drawdown=max_drawdown(r),
        sortino=sortino,
        cum_return=float(np.prod(1.0 + r) - 1.0),
        jarque_bera_p=jb_p,
    )
