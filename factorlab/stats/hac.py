"""
factorlab.stats.hac — 異質變異與自相關一致（HAC）標準誤。

為什麼需要
----------
原始碼用 `scipy.stats.ttest_1samp` 對 fwd_3m / fwd_6m / fwd_12m 做檢定。
那個函式假設觀察值 i.i.d.，變異數估計是 γ0/T。但每月取樣、持有 h 個月的
報酬序列，相鄰觀察值共用 h-1 個月的標的報酬，自相關是抽樣設計的結果。

真正的樣本平均數變異數是長期變異數除以 T：

    Var(x̄) = S / T,   S = γ0 + 2·Σ_{j>=1} w_j·γ_j

若底層月報酬近似 i.i.d.、h 月重疊，則 γ_j ≈ (1 - j/h)·γ0（j < h），代入得

    S ≈ γ0·[1 + 2·Σ_{j=1}^{h-1}(1 - j/h)] = γ0·h

所以 i.i.d. 假設下的 t 值被高估約 √h 倍：h=12 時是 3.46 倍。
`inflation_factor()` 直接把這個理論倍數算出來，報告裡會跟實際觀察到的
膨脹倍數並列 —— 兩者接近就說明「顯著」純粹來自重疊。

兩種核
------
- Bartlett（Newey-West 1987）：w_j = 1 - j/(L+1)，保證半正定，代價是
  對重疊報酬會略微低估 S（權重遞減，但真實 γ_j 的貢獻是 1-j/h）。
- Truncated（Hansen-Hodrick 1980）：w_j = 1，對 h 月重疊的理論結構是
  **恰好正確**的權重，但不保證半正定，小樣本可能算出負變異數。
  本模組偵測到負值時回報 nan 並在 `psd_failed` 標記，絕不回傳假數字。

對重疊報酬，正確的落後期數是 h-1（`lags_for_overlap`）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

Kernel = str


@dataclass
class HACResult:
    """HAC 修正後的平均數檢定結果。"""

    mean: float
    se: float
    tstat: float
    pvalue: float
    nobs: int
    lags: int
    kernel: Kernel
    iid_se: float
    iid_tstat: float
    psd_failed: bool = False

    @property
    def inflation(self) -> float:
        """天真 t 值相對 HAC t 值的膨脹倍數（= HAC se / iid se）。"""
        if not np.isfinite(self.se) or self.se <= 0 or not np.isfinite(self.iid_se):
            return float("nan")
        return self.se / self.iid_se

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}se": self.se,
            f"{prefix}t": self.tstat,
            f"{prefix}p": self.pvalue,
            f"{prefix}nobs": self.nobs,
            f"{prefix}lags": self.lags,
            f"{prefix}kernel": self.kernel,
            f"{prefix}t_iid": self.iid_tstat,
            f"{prefix}se_inflation": self.inflation,
        }


def kernel_weights(lags: int, kernel: Kernel = "bartlett") -> np.ndarray:
    """回傳 j = 1..lags 的核權重。"""
    if lags < 0:
        raise ValueError("lags 不可為負")
    j = np.arange(1, lags + 1, dtype="float64")
    if kernel == "bartlett":
        return 1.0 - j / (lags + 1.0)
    if kernel == "truncated":
        return np.ones_like(j)
    raise ValueError(f"未知的核 {kernel!r}，只接受 'bartlett' / 'truncated'")


def newey_west_bandwidth(nobs: int) -> int:
    """Newey-West (1994) 的經驗法則帶寬：floor(4·(T/100)^(2/9))。"""
    if nobs <= 1:
        return 0
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def andrews_bandwidth(x: np.ndarray) -> int:
    """Andrews (1991) 的 AR(1) plug-in 帶寬（Bartlett 核）。

    L* = 1.1447·(4·ρ²·T / (1-ρ²)²)^(1/3)，ρ 由 AR(1) 估得。
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    t = x.size
    if t < 3:
        return 0
    xc = x - x.mean()
    denom = (xc[:-1] ** 2).sum()
    rho = float((xc[1:] * xc[:-1]).sum() / denom) if denom > 0 else 0.0
    rho = float(np.clip(rho, -0.97, 0.97))
    alpha = 4.0 * rho**2 / (1.0 - rho**2) ** 2
    band = 1.1447 * (alpha * t) ** (1.0 / 3.0)
    return int(min(max(np.floor(band), 0), t - 1))


def lags_for_overlap(horizon: int) -> int:
    """h 月重疊報酬的正確落後期數：h-1。"""
    return max(int(horizon) - 1, 0)


def inflation_factor(horizon: int) -> float:
    """i.i.d. 底層報酬下，重疊 h 月取樣造成的 t 值理論膨脹倍數 √h。"""
    return float(np.sqrt(max(int(horizon), 1)))


def long_run_variance(
    x: np.ndarray,
    lags: int | None = None,
    kernel: Kernel = "bartlett",
) -> tuple[float, int, bool]:
    """估長期變異數 S = γ0 + 2·Σ w_j γ_j。

    Returns
    -------
    (S, lags_used, psd_failed)
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    t = x.size
    if t < 2:
        return float("nan"), 0, False

    if lags is None:
        lags = newey_west_bandwidth(t)
    lags = int(min(max(lags, 0), t - 1))

    xc = x - x.mean()
    gamma0 = float((xc**2).sum() / t)
    s = gamma0
    if lags > 0:
        w = kernel_weights(lags, kernel)
        for j in range(1, lags + 1):
            gamma_j = float((xc[j:] * xc[:-j]).sum() / t)
            s += 2.0 * w[j - 1] * gamma_j
    psd_failed = not (s > 0)
    return (s if s > 0 else float("nan")), lags, psd_failed


def hac_mean_test(
    x,
    lags: int | None = None,
    kernel: Kernel = "bartlett",
    auto: str = "nw",
) -> HACResult:
    """對序列平均數是否為 0 做 HAC 修正的檢定。

    Parameters
    ----------
    lags : None 時依 `auto` 自動選帶寬（"nw" = Newey-West 1994 法則，
           "andrews" = AR(1) plug-in）。
    """
    arr = np.asarray(pd.Series(x).to_numpy(), dtype="float64")
    arr = arr[np.isfinite(arr)]
    t = arr.size
    if t < 3:
        return HACResult(
            mean=float(arr.mean()) if t else float("nan"),
            se=float("nan"), tstat=float("nan"), pvalue=float("nan"),
            nobs=t, lags=0, kernel=kernel,
            iid_se=float("nan"), iid_tstat=float("nan"),
        )

    if lags is None:
        lags = andrews_bandwidth(arr) if auto == "andrews" else newey_west_bandwidth(t)

    mean = float(arr.mean())
    iid_se = float(arr.std(ddof=1) / np.sqrt(t))
    iid_t = mean / iid_se if iid_se > 0 else float("nan")

    s, lags_used, psd_failed = long_run_variance(arr, lags, kernel)
    if not np.isfinite(s):
        return HACResult(
            mean=mean, se=float("nan"), tstat=float("nan"), pvalue=float("nan"),
            nobs=t, lags=lags_used, kernel=kernel,
            iid_se=iid_se, iid_tstat=iid_t, psd_failed=True,
        )

    se = float(np.sqrt(s / t))
    tstat = mean / se if se > 0 else float("nan")
    # 用 t 分配（df = T-1）而不是常態：215 個月下差異很小，但小樣本
    # （3 年 = 34 個月）差異可觀，而本研究的重點正是小樣本沒有檢定力。
    pval = float(2.0 * sps.t.sf(abs(tstat), df=t - 1)) if np.isfinite(tstat) else float("nan")
    return HACResult(
        mean=mean, se=se, tstat=float(tstat), pvalue=pval,
        nobs=t, lags=lags_used, kernel=kernel,
        iid_se=iid_se, iid_tstat=float(iid_t), psd_failed=psd_failed,
    )


def hansen_hodrick_mean_test(x, horizon: int) -> HACResult:
    """Hansen-Hodrick：對 h 月重疊報酬用截斷核、落後 h-1 期。"""
    return hac_mean_test(x, lags=lags_for_overlap(horizon), kernel="truncated")


def newey_west_mean_test(x, horizon: int | None = None) -> HACResult:
    """Newey-West：Bartlett 核。給定 horizon 時用 h-1 落後期。"""
    lags = lags_for_overlap(horizon) if horizon is not None else None
    return hac_mean_test(x, lags=lags, kernel="bartlett")


def hac_cov(x_mat: np.ndarray, resid: np.ndarray, lags: int, kernel: Kernel = "bartlett") -> np.ndarray:
    """迴歸用的 HAC meat 矩陣 S = Σ_j w_j (Γ_j + Γ_j')，Γ_j = Σ_t u_t u_{t-j} x_t x_{t-j}'。"""
    x_mat = np.asarray(x_mat, dtype="float64")
    u = np.asarray(resid, dtype="float64").reshape(-1, 1)
    t = x_mat.shape[0]
    xu = x_mat * u
    s = xu.T @ xu
    if lags > 0:
        w = kernel_weights(lags, kernel)
        for j in range(1, min(lags, t - 1) + 1):
            g = xu[j:].T @ xu[:-j]
            s = s + w[j - 1] * (g + g.T)
    return s
