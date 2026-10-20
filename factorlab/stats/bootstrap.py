"""
factorlab.stats.bootstrap — 定態／區塊 bootstrap。

用途有三個，都無法用解析式解決：
1. Patton-Timmermann 單調性檢定的 p 值（統計量是 min，沒有好用的分配）。
2. Sharpe 比率的信賴區間（Sharpe 是比率，delta method 在小樣本很差）。
3. Hansen SPA 檢定的虛無分配（多策略最大值的分配）。

為什麼不用 i.i.d. bootstrap：月報酬有自相關（尤其日曆時間重疊投組仍留
少量），i.i.d. 重抽會破壞它、低估變異。Politis-Romano (1994) 的定態
bootstrap 以幾何分配的區塊長度重抽，重抽後的序列仍是定態的，是時間序列
bootstrap 的預設選擇。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapCI:
    point: float
    lower: float
    upper: float
    pvalue: float
    reps: int
    alpha: float

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}point": self.point,
            f"{prefix}ci_lo": self.lower,
            f"{prefix}ci_hi": self.upper,
            f"{prefix}boot_p": self.pvalue,
            f"{prefix}reps": self.reps,
        }


def stationary_bootstrap_indices(
    nobs: int,
    reps: int,
    block: float = 6.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis-Romano 定態 bootstrap 的索引矩陣 (reps, nobs)。

    每一步以機率 p = 1/block 重新隨機跳到一個新起點，否則往前走一格
    （環狀繞回）。平均區塊長度因此是 block。
    """
    if nobs <= 0:
        raise ValueError("nobs 必須為正")
    rng = rng or np.random.default_rng()
    p = 1.0 / max(block, 1.0)

    idx = np.empty((reps, nobs), dtype="int64")
    idx[:, 0] = rng.integers(0, nobs, size=reps)
    jump = rng.random((reps, nobs)) < p
    fresh = rng.integers(0, nobs, size=(reps, nobs))
    for t in range(1, nobs):
        cont = (idx[:, t - 1] + 1) % nobs
        idx[:, t] = np.where(jump[:, t], fresh[:, t], cont)
    return idx


def circular_block_bootstrap_indices(
    nobs: int,
    reps: int,
    block: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """固定長度環狀區塊 bootstrap 的索引矩陣 (reps, nobs)。"""
    rng = rng or np.random.default_rng()
    block = int(max(1, min(block, nobs)))
    n_blocks = int(np.ceil(nobs / block))
    starts = rng.integers(0, nobs, size=(reps, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % nobs
    return idx.reshape(reps, -1)[:, :nobs]


def _resample(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """對 1 維或 2 維（時間 × 序列）陣列按時間索引重抽。"""
    return arr[idx]


def bootstrap_mean(
    x,
    reps: int = 2000,
    block: float = 6.0,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapCI:
    """平均數的定態 bootstrap 信賴區間與雙尾 p 值。

    p 值的虛無是「平均數為 0」，做法是把重抽平均數移心到 0，
    再看原始平均數有多極端（等價於檢定 H0: μ = 0）。
    """
    arr = np.asarray(pd.Series(x).dropna().to_numpy(), dtype="float64")
    t = arr.size
    if t < 5:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), float("nan"), 0, alpha)

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, reps, block, rng)
    boot = _resample(arr, idx).mean(axis=1)

    point = float(arr.mean())
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    centered = boot - boot.mean()
    pval = float((np.abs(centered) >= abs(point)).mean())
    return BootstrapCI(point, lo, hi, pval, reps, alpha)


def bootstrap_sharpe(
    x,
    reps: int = 2000,
    block: float = 6.0,
    alpha: float = 0.05,
    periods_per_year: int = 12,
    seed: int | None = None,
) -> BootstrapCI:
    """年化 Sharpe 的定態 bootstrap 信賴區間。"""
    arr = np.asarray(pd.Series(x).dropna().to_numpy(), dtype="float64")
    t = arr.size
    if t < 8:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), float("nan"), 0, alpha)

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, reps, block, rng)
    samples = _resample(arr, idx)
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(sd > 0, mu / sd, np.nan) * np.sqrt(periods_per_year)
    boot = boot[np.isfinite(boot)]

    sd0 = arr.std(ddof=1)
    point = float(arr.mean() / sd0 * np.sqrt(periods_per_year)) if sd0 > 0 else float("nan")
    if boot.size == 0:
        return BootstrapCI(point, float("nan"), float("nan"), float("nan"), reps, alpha)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    centered = boot - boot.mean()
    pval = float((np.abs(centered) >= abs(point)).mean())
    return BootstrapCI(point, lo, hi, pval, int(boot.size), alpha)


def bootstrap_statistic(
    data: np.ndarray,
    stat_fn,
    reps: int = 2000,
    block: float = 6.0,
    seed: int | None = None,
) -> np.ndarray:
    """對任意統計量做定態 bootstrap，回傳 reps 個重抽統計量。

    `data` 第一個維度必須是時間；`stat_fn` 吃一個與 data 同形狀的陣列。
    """
    arr = np.asarray(data, dtype="float64")
    t = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, reps, block, rng)
    out = np.empty(reps, dtype="float64")
    for b in range(reps):
        out[b] = stat_fn(arr[idx[b]])
    return out
