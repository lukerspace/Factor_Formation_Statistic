"""
factorlab.stats.power — 檢定力與最小可偵測效果（MDE）。

這個模組回答原始研究裡最被誤讀的一件事
----------------------------------------
原始碼報 `12M Q10-Q1: n=23, mean=1.329%, t=1.143, p=0.2654`，很容易被
讀成「動能在這段期間不存在」。那是錯的。p=0.27 在 23 個觀察值、月標準差
5.58% 的情況下，只說明「這份資料沒有能力偵測這個大小的效果」。

把數字攤開：23 個月、σ=5.58%，在 α=0.05、檢定力 80% 的要求下，
最小可偵測的月報酬是

    MDE = (z_{0.975} + z_{0.80})·σ/√n = 2.802 × 5.58% / 4.796 ≈ 3.26%/月

年化約 47%。意思是：**只有年化 47% 以上的多空報酬，這份樣本才有八成
機會抓到。** 真實的動能溢酬文獻值大約是年化 8~12%（月 0.7~1.0%），
遠在偵測門檻之下。所以「沒顯著」是預期中的結果，不是證據。

反過來問：要偵測月 0.8% 的效果需要多少樣本？

    n = ((z_{0.975} + z_{0.80})·σ/effect)² = (2.802 × 5.58 / 0.8)² ≈ 382 個月

約 32 年。這就是為什麼本專案預設用 history="long"（2007–2024，215 個月）：
215 個月仍然不足以在 80% 檢定力下偵測月 0.8%，但至少進入了同一個數量級，
而且 3 年樣本連「無法判斷」都說不出口。

注意：若還要做多重檢定校正，α 會從 0.05 降到 0.05/32 ≈ 0.0016，
MDE 再上升約 30%。`power_table` 支援傳入校正後的 α。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS_PER_YEAR = 12


@dataclass
class PowerResult:
    """單一設定下的檢定力診斷。"""

    nobs: int
    sigma_m: float
    alpha: float
    target_power: float
    observed_mean_m: float | None
    mde_m: float
    mde_ann: float
    mde_sharpe_ann: float
    power_at_observed: float
    power_at_reference: float
    reference_mean_m: float
    required_n_for_observed: float
    required_n_for_reference: float

    @property
    def underpowered(self) -> bool:
        """對「文獻參考值」的檢定力是否低於目標。"""
        return bool(np.isfinite(self.power_at_reference) and self.power_at_reference < self.target_power)

    def as_dict(self, prefix: str = "pow_") -> dict:
        return {
            f"{prefix}nobs": self.nobs,
            f"{prefix}sigma_m": self.sigma_m,
            f"{prefix}alpha": self.alpha,
            f"{prefix}mde_m": self.mde_m,
            f"{prefix}mde_ann": self.mde_ann,
            f"{prefix}mde_sharpe_ann": self.mde_sharpe_ann,
            f"{prefix}power_at_observed": self.power_at_observed,
            f"{prefix}power_at_ref_{self.reference_mean_m:.4f}": self.power_at_reference,
            f"{prefix}required_n_for_ref": self.required_n_for_reference,
            f"{prefix}underpowered": self.underpowered,
        }


def minimum_detectable_effect(
    sigma: float,
    nobs: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """雙尾單樣本檢定的最小可偵測效果（與 sigma 同單位）。

        MDE = (z_{1-α/2} + z_{power}) · σ / √n
    """
    if nobs <= 1 or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    z_a = sps.norm.isf(alpha / 2.0 if two_sided else alpha)
    z_b = sps.norm.ppf(power)
    return float((z_a + z_b) * sigma / np.sqrt(nobs))


def power_at_effect(
    effect: float,
    sigma: float,
    nobs: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """給定真實效果大小時的檢定力。

    用非中心 t 分配而不是常態近似：n=23 時兩者差別不可忽略，而小樣本
    正是本研究的重點。
    """
    if nobs <= 1 or not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(effect):
        return float("nan")
    df = nobs - 1
    ncp = effect / sigma * np.sqrt(nobs)
    if two_sided:
        crit = sps.t.isf(alpha / 2.0, df)
        return float(sps.nct.sf(crit, df, ncp) + sps.nct.cdf(-crit, df, ncp))
    crit = sps.t.isf(alpha, df)
    return float(sps.nct.sf(crit, df, ncp))


def required_nobs(
    effect: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    max_n: int = 100_000,
) -> float:
    """達到目標檢定力所需的觀察期數。"""
    if not np.isfinite(effect) or effect == 0 or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    z_a = sps.norm.isf(alpha / 2.0 if two_sided else alpha)
    z_b = sps.norm.ppf(power)
    n0 = ((z_a + z_b) * sigma / abs(effect)) ** 2
    # 用常態解當起點，再往上微調到非中心 t 下真的達標。
    n = int(max(np.ceil(n0), 3))
    while n < max_n and power_at_effect(effect, sigma, n, alpha, two_sided) < power:
        n = int(np.ceil(n * 1.05)) + 1
    return float(min(n, max_n))


def diagnose(
    returns=None,
    sigma: float | None = None,
    nobs: int | None = None,
    observed_mean: float | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    reference_mean_m: float = 0.008,
) -> PowerResult:
    """對一個策略做完整的檢定力診斷。

    Parameters
    ----------
    returns          : 月報酬序列（給了就自動取 sigma / nobs / mean）。
    reference_mean_m : 文獻參考效果，預設月 0.8%（年化約 10%），
                       大致是學界動能溢酬估計的中位數。問題不是
                       「這份資料有沒有拒絕虛無」，而是「若真實效果
                       是文獻值，這份資料有多少機會抓到」。
    """
    if returns is not None:
        r = pd.Series(returns).dropna().to_numpy(dtype="float64")
        nobs = int(r.size)
        sigma = float(r.std(ddof=1)) if r.size > 1 else float("nan")
        observed_mean = float(r.mean()) if r.size else None
    if sigma is None or nobs is None:
        raise ValueError("必須提供 returns，或同時提供 sigma 與 nobs")

    mde = minimum_detectable_effect(sigma, nobs, alpha, target_power)
    return PowerResult(
        nobs=int(nobs),
        sigma_m=float(sigma),
        alpha=float(alpha),
        target_power=float(target_power),
        observed_mean_m=observed_mean,
        mde_m=mde,
        mde_ann=float((1.0 + mde) ** MONTHS_PER_YEAR - 1.0) if np.isfinite(mde) else float("nan"),
        mde_sharpe_ann=float(mde / sigma * np.sqrt(MONTHS_PER_YEAR)) if sigma > 0 else float("nan"),
        power_at_observed=(
            power_at_effect(observed_mean, sigma, nobs, alpha)
            if observed_mean is not None
            else float("nan")
        ),
        power_at_reference=power_at_effect(reference_mean_m, sigma, nobs, alpha),
        reference_mean_m=float(reference_mean_m),
        required_n_for_observed=(
            required_nobs(observed_mean, sigma, alpha, target_power)
            if observed_mean
            else float("nan")
        ),
        required_n_for_reference=required_nobs(reference_mean_m, sigma, alpha, target_power),
    )


def power_table(
    sigma: float,
    nobs_grid=(23, 34, 60, 120, 215, 360),
    effect_grid=(0.004, 0.008, 0.012, 0.020),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """檢定力矩陣：列 = 樣本月數，欄 = 真實月效果。

    這張表是整份報告最有說服力的一頁 —— 它讓「3 年樣本測不出東西」
    從一句主張變成一個可以查表的事實。
    """
    rows = []
    for n in nobs_grid:
        row = {"nobs_months": n, "years": round(n / 12.0, 1),
               "mde_m_at_80pct": minimum_detectable_effect(sigma, n, alpha, 0.80)}
        for e in effect_grid:
            row[f"power_at_{e * 100:.1f}pct_m"] = power_at_effect(e, sigma, n, alpha)
        rows.append(row)
    return pd.DataFrame(rows)
