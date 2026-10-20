"""
factorlab.stats.multiple_testing — 多重檢定校正與 Sharpe 折價。

問題的規模
----------
這份研究掃了 4 個 lookback × 2 個 skip × 4 個 horizon = 32 個檢定。
在「所有策略真實報酬都是零」的全域虛無之下，32 個獨立 t 統計量的最大
絕對值期望值，常見的漸近近似是

    sqrt(2·ln M) = sqrt(2·ln 32) ≈ 2.63

但這個近似在 M 小的時候會**高估**（它省略了低階項）。M = 32 的精確值
用 Monte Carlo 算出來約 **2.35**，95% 分位約 3.0。`expected_max_abs_t`
同時回報兩者，報告也並列 —— 這裡刻意不拿好看的那個數字充當事實。

無論用哪個，結論都一樣：就算動能完全不存在，掃 32 個組合**預期**就會
看到一個 |t| 超過 2.3 的策略，而 5% 水準的門檻要到 3.0 附近。用 1.96
當門檻去宣告發現，等於保證自己會發現東西。原始碼沒有做任何校正，
所以它報出來的 p 值無論大小都無法支持「這個策略有效」的結論。

(32 個檢定其實高度相關 —— 共用 lookback 或 horizon 的策略持股大量重疊 ——
 相關會讓有效獨立檢定數下降，max|t| 的期望隨之降低。`expected_max_abs_t`
 允許指定等相關係數，把這個影響量化出來。)

本模組提供
----------
- Holm-Bonferroni：控制 FWER（族系錯誤率），最嚴格。
- Benjamini-Hochberg / Benjamini-Yekutieli：控制 FDR。BY 不假設獨立，
  是 Harvey-Liu-Zhu 建議用在因子研究上的版本。
- Harvey-Liu-Zhu (2016) haircut：把校正後的 p 值換回 t 值，再折減 Sharpe。
  HLZ 的結論是新因子的門檻應該是 t > 3.0，不是 1.96。
- Deflated Sharpe Ratio（Bailey & López de Prado 2014）：直接把「試了
  幾次」與「報酬的偏態峰態」一起吃進去，回答「這個 Sharpe 在考慮選擇
  偏誤後還剩多少機率是真的」。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def _as_array(pvalues) -> np.ndarray:
    p = np.asarray(pd.Series(pvalues).to_numpy(), dtype="float64")
    return np.clip(p, 1e-16, 1.0)


def bonferroni(pvalues) -> np.ndarray:
    """Bonferroni 校正後 p 值。"""
    p = _as_array(pvalues)
    return np.minimum(1.0, p * p.size)


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni 逐步下降法，控制 FWER，比 Bonferroni 有檢定力。"""
    p = _as_array(pvalues)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype="float64")
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj_sorted[rank] = min(1.0, running)
    out = np.empty(m, dtype="float64")
    out[order] = adj_sorted
    return out


def benjamini_hochberg(pvalues) -> np.ndarray:
    """BH 校正後 q 值，控制 FDR（假設檢定間獨立或正相關）。"""
    p = _as_array(pvalues)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype="float64")
    adj = p[order] * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m, dtype="float64")
    out[order] = np.minimum(1.0, adj)
    return out


def benjamini_yekutieli(pvalues) -> np.ndarray:
    """BY 校正後 q 值，控制 FDR 且**不假設獨立**。

    多了一個 c(m) = Σ_{i=1}^{m} 1/i 的懲罰項。因子研究裡的策略彼此高度
    相關，獨立性假設不成立，所以 Harvey-Liu-Zhu 建議用 BY 而不是 BH。
    """
    p = _as_array(pvalues)
    m = p.size
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranks = np.arange(1, m + 1, dtype="float64")
    adj = p[order] * m * cm / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m, dtype="float64")
    out[order] = np.minimum(1.0, adj)
    return out


def p_to_t(pvalues, df: int | None = None) -> np.ndarray:
    """把雙尾 p 值換回 t 值（HLZ haircut 的中間步驟）。"""
    p = _as_array(pvalues)
    p = np.clip(p, 1e-16, 1.0 - 1e-16)
    if df is None:
        return sps.norm.isf(p / 2.0)
    return sps.t.isf(p / 2.0, df=df)


def expected_max_abs_t(
    n_trials: int,
    rho: float = 0.0,
    reps: int = 20000,
    seed: int | None = 0,
) -> dict:
    """全域虛無下 max|t| 的期望值。

    `rho` = 檢定間的等相關係數。rho=0 時回傳解析近似 sqrt(2 ln M) 與
    Monte Carlo 值；rho>0 時只有 Monte Carlo（相關會讓有效獨立檢定數
    下降，max|t| 的期望隨之降低）。

    對本研究的 32 個策略，實際相關係數很高（0.7~0.95），所以 2.63 這個
    數字是保守上界 —— 但即使把 rho 設到 0.9，E[max|t|] 仍明顯高於 1.96。
    """
    m = int(n_trials)
    if m < 1:
        raise ValueError("n_trials 至少為 1")
    rng = np.random.default_rng(seed)
    if rho <= 0:
        z = rng.standard_normal((reps, m))
    else:
        rho = float(min(rho, 0.999))
        common = rng.standard_normal((reps, 1))
        idio = rng.standard_normal((reps, m))
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    max_abs = np.abs(z).max(axis=1)
    return {
        "n_trials": m,
        "rho": float(rho),
        "asymptotic_sqrt_2logM": float(np.sqrt(2.0 * np.log(m))) if rho <= 0 else float("nan"),
        "mc_expected_max_abs_t": float(max_abs.mean()),
        "mc_q95_max_abs_t": float(np.quantile(max_abs, 0.95)),
    }


@dataclass
class HaircutResult:
    """單一策略的 HLZ Sharpe 折價。"""

    method: str
    t_orig: float
    p_orig: float
    p_adj: float
    t_adj: float
    sharpe: float
    sharpe_haircut: float
    haircut_pct: float
    survives_hlz_threshold: bool

    def as_dict(self, prefix: str = "") -> dict:
        m = self.method
        return {
            f"{prefix}{m}_p_adj": self.p_adj,
            f"{prefix}{m}_t_adj": self.t_adj,
            f"{prefix}{m}_sharpe_haircut": self.sharpe_haircut,
            f"{prefix}{m}_haircut_pct": self.haircut_pct,
        }


def haircut_sharpe(
    tstats,
    sharpes,
    methods: tuple[str, ...] = ("bonferroni", "holm", "by"),
    hlz_threshold: float = 3.0,
) -> pd.DataFrame:
    """Harvey-Liu-Zhu (2016) 的 Sharpe 折價表。

    做法：觀察到的 t → 雙尾 p → 多重檢定校正後 p_adj → 換回 t_adj
    → 折價後 Sharpe = Sharpe × (t_adj / t)。直覺是「t 值縮水多少，
    Sharpe 就縮水多少」，因為兩者在同一樣本長度下成比例。

    Returns
    -------
    DataFrame，每列一個策略，欄位含各方法的 p_adj / t_adj / 折價後 Sharpe。
    """
    t_arr = np.asarray(pd.Series(tstats).to_numpy(), dtype="float64")
    s_arr = np.asarray(pd.Series(sharpes).to_numpy(), dtype="float64")
    if t_arr.shape != s_arr.shape:
        raise ValueError("tstats 與 sharpes 長度必須相同")

    p_orig = 2.0 * sps.norm.sf(np.abs(t_arr))
    adj = {
        "bonferroni": bonferroni(p_orig),
        "holm": holm(p_orig),
        "by": benjamini_yekutieli(p_orig),
        "bh": benjamini_hochberg(p_orig),
    }

    out = pd.DataFrame({"t_orig": t_arr, "p_orig": p_orig, "sharpe": s_arr})
    for m in methods:
        if m not in adj:
            raise ValueError(f"未知的校正方法 {m!r}")
        p_adj = adj[m]
        t_adj = p_to_t(p_adj)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.abs(t_arr) > 1e-12, t_adj / np.abs(t_arr), np.nan)
        hc_sharpe = s_arr * ratio
        out[f"{m}_p_adj"] = p_adj
        out[f"{m}_t_adj"] = t_adj
        out[f"{m}_sharpe_haircut"] = hc_sharpe
        out[f"{m}_haircut_pct"] = (1.0 - ratio) * 100.0
        out[f"{m}_significant_5pct"] = p_adj < 0.05

    out["survives_hlz_t3"] = np.abs(t_arr) > hlz_threshold
    return out


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。"""

    sharpe_ann: float
    sharpe_period: float
    expected_max_sharpe_period: float
    expected_max_sharpe_ann: float
    dsr: float
    nobs: int
    n_trials: int
    skew: float
    kurtosis: float
    variance_of_trial_sharpes: float

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.dsr) and self.dsr > 0.95)

    def as_dict(self, prefix: str = "dsr_") -> dict:
        return {
            f"{prefix}sharpe_ann": self.sharpe_ann,
            f"{prefix}expected_max_sharpe_ann": self.expected_max_sharpe_ann,
            f"{prefix}value": self.dsr,
            f"{prefix}significant_95": self.significant,
            f"{prefix}n_trials": self.n_trials,
            f"{prefix}nobs": self.nobs,
        }


def expected_max_sharpe(
    var_trial_sharpes: float,
    n_trials: int,
) -> float:
    """虛無下 N 次試驗的最大 Sharpe 期望值（期頻，非年化）。

        E[max SR] ≈ sqrt(V)·[(1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))]

    γ 是 Euler-Mascheroni 常數。V 是各次試驗 Sharpe 估計值的變異數 ——
    這是關鍵輸入：如果你掃的 32 個策略 Sharpe 散布很廣，那麼「最高的
    那個」本來就會很高，門檻自然要抬高。
    """
    n = int(max(n_trials, 2))
    v = float(var_trial_sharpes)
    if not np.isfinite(v) or v <= 0:
        return float("nan")
    term = (1.0 - EULER_GAMMA) * sps.norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * sps.norm.ppf(
        1.0 - 1.0 / (n * np.e)
    )
    return float(np.sqrt(v) * term)


def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    trial_sharpes=None,
    periods_per_year: int = 12,
) -> DeflatedSharpeResult:
    """計算 Deflated Sharpe Ratio。

    Parameters
    ----------
    returns      : 勝出策略的報酬序列（月頻）。
    n_trials     : 一共試了幾個策略（= StudySpec.n_trials）。
    trial_sharpes: 所有試驗的**年化** Sharpe；用來估 V。若為 None，
                   退而用一個保守預設（V 由樣本長度推得），並在結果裡標示。

    DSR 的意義是「在校正了選擇偏誤、樣本長度、偏態與峰態之後，真實
    Sharpe 大於零的機率」。慣例是 DSR > 0.95 才算通過。
    """
    r = pd.Series(returns).dropna().to_numpy(dtype="float64")
    t = r.size
    nan = float("nan")
    if t < 12:
        return DeflatedSharpeResult(nan, nan, nan, nan, nan, t, int(n_trials), nan, nan, nan)

    sd = r.std(ddof=1)
    if sd <= 0:
        return DeflatedSharpeResult(nan, nan, nan, nan, nan, t, int(n_trials), nan, nan, nan)

    sr_p = float(r.mean() / sd)                       # 期頻 Sharpe
    sr_ann = sr_p * np.sqrt(periods_per_year)
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))

    if trial_sharpes is not None:
        ts = np.asarray(pd.Series(trial_sharpes).dropna().to_numpy(), dtype="float64")
        ts_p = ts / np.sqrt(periods_per_year)         # 換回期頻才能跟 sr_p 比
        v = float(np.var(ts_p, ddof=1)) if ts_p.size > 1 else nan
    else:
        # 保守預設：虛無下期頻 Sharpe 估計的變異數約 1/T。
        v = 1.0 / t

    sr0 = expected_max_sharpe(v, n_trials)

    denom_sq = 1.0 - g3 * sr_p + 0.25 * (g4 - 1.0) * sr_p**2
    if not np.isfinite(denom_sq) or denom_sq <= 0:
        dsr = nan
    else:
        z = (sr_p - sr0) * np.sqrt(t - 1.0) / np.sqrt(denom_sq)
        dsr = float(sps.norm.cdf(z))

    return DeflatedSharpeResult(
        sharpe_ann=sr_ann,
        sharpe_period=sr_p,
        expected_max_sharpe_period=sr0,
        expected_max_sharpe_ann=sr0 * np.sqrt(periods_per_year) if np.isfinite(sr0) else nan,
        dsr=dsr,
        nobs=t,
        n_trials=int(n_trials),
        skew=g3,
        kurtosis=g4,
        variance_of_trial_sharpes=v,
    )
