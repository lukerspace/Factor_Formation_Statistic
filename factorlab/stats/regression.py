"""
factorlab.stats.regression — 時間序列迴歸、HAC 標準誤與 GRS 聯合檢定。

為什麼需要
----------
原始碼回報的是 Q10-Q1 的**原始平均數**，然後說它顯著或不顯著。原始平均數
不是 alpha。動能的高分位在多頭末段會系統性地偏向高 beta、小型、低流動性、
高波動的股票，Q10-Q1 的平均報酬裡有一部分只是這些已知風險的補償。
要主張「這是一個新因子」，必須拿掉已知因子後截距還在。

`factor_regression` 做這件事：回歸多空報酬在市場／規模／短期反轉／波動度
四個因子上，取截距當 alpha，標準誤用 HAC。

`grs_test` 做另一件事：原始碼只看 Q10 與 Q1 兩端，其餘八個分位的資訊全丟。
GRS (Gibbons-Ross-Shanken 1989) 同時檢定十個分位的 alpha 是否全為零 ——
這是「模型能不能解釋整條分位曲線」的聯合檢定，比單看兩端的差值嚴格得多，
也不會被兩端的極端值主導。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .hac import hac_cov, newey_west_bandwidth


@dataclass
class RegressionResult:
    """單一時間序列迴歸的結果（截距即 alpha）。"""

    names: list[str]
    params: np.ndarray
    se: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    nobs: int
    r2: float
    r2_adj: float
    lags: int
    resid: np.ndarray = field(repr=False, default=None)

    @property
    def alpha(self) -> float:
        return float(self.params[0])

    @property
    def alpha_t(self) -> float:
        return float(self.tstats[0])

    @property
    def alpha_p(self) -> float:
        return float(self.pvalues[0])

    def coef(self, name: str) -> float:
        return float(self.params[self.names.index(name)])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.params, "se": self.se, "t": self.tstats, "p": self.pvalues},
            index=self.names,
        )

    def as_dict(self, prefix: str = "") -> dict:
        out = {
            f"{prefix}alpha_m": self.alpha,
            f"{prefix}alpha_ann": (1.0 + self.alpha) ** 12 - 1.0,
            f"{prefix}alpha_t": self.alpha_t,
            f"{prefix}alpha_p": self.alpha_p,
            f"{prefix}r2": self.r2,
            f"{prefix}nobs": self.nobs,
            f"{prefix}hac_lags": self.lags,
        }
        for n, b, t in zip(self.names[1:], self.params[1:], self.tstats[1:]):
            out[f"{prefix}beta_{n}"] = float(b)
            out[f"{prefix}t_{n}"] = float(t)
        return out


def ols_hac(
    y,
    x,
    lags: int | None = None,
    kernel: str = "bartlett",
    add_const: bool = True,
) -> RegressionResult:
    """OLS + HAC（Newey-West）標準誤。

    `x` 可以是 DataFrame（用欄名）或 2 維陣列。截距永遠排在第一個。
    """
    y_s = pd.Series(y).rename("y")
    if isinstance(x, pd.Series):
        x_df = x.to_frame()
    elif isinstance(x, pd.DataFrame):
        x_df = x.copy()
    else:
        x_df = pd.DataFrame(np.asarray(x))
        x_df.columns = [f"x{i}" for i in range(x_df.shape[1])]

    # 明確排序：HAC 的落後結構完全依賴觀察值的時間順序，不能依賴
    # pandas 對 DatetimeIndex 的預設排序行為（4.0 起將改為不排序）。
    data = pd.concat([y_s, x_df], axis=1, sort=False).dropna().sort_index()
    if data.empty:
        raise ValueError("迴歸樣本為空（y 與 x 沒有重疊的有效觀察值）")

    y_v = data["y"].to_numpy(dtype="float64")
    x_v = data.drop(columns=["y"]).to_numpy(dtype="float64")
    names = list(data.columns[1:])

    if add_const:
        x_v = np.column_stack([np.ones(len(y_v)), x_v])
        names = ["const"] + names

    t, k = x_v.shape
    if t <= k:
        raise ValueError(f"觀察值 {t} 不足以估計 {k} 個參數")

    xtx_inv = np.linalg.pinv(x_v.T @ x_v)
    beta = xtx_inv @ (x_v.T @ y_v)
    resid = y_v - x_v @ beta

    if lags is None:
        lags = newey_west_bandwidth(t)
    lags = int(min(max(lags, 0), t - 1))

    meat = hac_cov(x_v, resid, lags, kernel)
    # 小樣本自由度修正 T/(T-k)，與 statsmodels 的 cov_type='HAC', use_correction 一致。
    cov = xtx_inv @ meat @ xtx_inv * (t / (t - k))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        tstats = np.where(se > 0, beta / se, np.nan)
    pvals = 2.0 * sps.t.sf(np.abs(tstats), df=t - k)

    tss = float(((y_v - y_v.mean()) ** 2).sum())
    rss = float((resid**2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (t - 1) / (t - k) if np.isfinite(r2) else float("nan")

    return RegressionResult(
        names=names, params=beta, se=se, tstats=tstats, pvalues=pvals,
        nobs=t, r2=r2, r2_adj=r2_adj, lags=lags, resid=resid,
    )


def factor_regression(
    portfolio_returns,
    factors: pd.DataFrame,
    lags: int | None = None,
    kernel: str = "bartlett",
) -> RegressionResult:
    """把投組報酬回歸在風險因子上，截距 = 風險調整後 alpha。"""
    return ols_hac(portfolio_returns, factors, lags=lags, kernel=kernel)


@dataclass
class GRSResult:
    """GRS 聯合檢定結果：H0 = 所有測試資產的 alpha 皆為零。"""

    statistic: float
    pvalue: float
    n_assets: int
    n_factors: int
    nobs: int
    alphas: np.ndarray
    mean_abs_alpha: float
    sr_squared_alpha: float

    def as_dict(self, prefix: str = "grs_") -> dict:
        return {
            f"{prefix}F": self.statistic,
            f"{prefix}p": self.pvalue,
            f"{prefix}n_assets": self.n_assets,
            f"{prefix}n_factors": self.n_factors,
            f"{prefix}nobs": self.nobs,
            f"{prefix}mean_abs_alpha_m": self.mean_abs_alpha,
            f"{prefix}sr2_alpha": self.sr_squared_alpha,
        }


def grs_test(test_assets: pd.DataFrame, factors: pd.DataFrame) -> GRSResult:
    """Gibbons-Ross-Shanken (1989) 聯合檢定。

    統計量（Fama-French 系列論文慣用的形式）：

        J = (T/N)·((T-N-K)/(T-K-1))·(α'Σ⁻¹α) / (1 + μ'Ω⁻¹μ) ~ F(N, T-N-K)

    其中 Σ = E'E/T（殘差共變異的 ML 估計），Ω = F_c'F_c/T，μ = 因子平均報酬。
    分母 (1 + μ'Ω⁻¹μ) 是因子本身的最大 Sharpe 平方，直覺是：因子組合本身
    越有效率，要再多解釋一點 alpha 就越難。

    注意：GRS 假設殘差為常態且同質變異，不含 HAC 修正。本研究用日曆時間
    非重疊月報酬做測試資產，所以自相關問題已在投組層面處理掉；但常態性
    仍可能不成立（見 moments 的 Jarque-Bera），因此報告把 GRS 當輔助證據，
    主結論仍以 HAC t 值與 bootstrap 為準。
    """
    data = pd.concat([test_assets, factors], axis=1, sort=False).dropna().sort_index()
    if data.empty:
        raise ValueError("GRS 樣本為空")

    n = test_assets.shape[1]
    k = factors.shape[1]
    r = data[test_assets.columns].to_numpy(dtype="float64")
    f = data[factors.columns].to_numpy(dtype="float64")
    t = r.shape[0]

    if t - n - k <= 0:
        raise ValueError(
            f"GRS 需要 T > N + K：目前 T={t}, N={n}, K={k}。"
            "這正是 3 年樣本（T≈34）無法做十分位 GRS 的原因。"
        )

    x = np.column_stack([np.ones(t), f])
    coef = np.linalg.pinv(x.T @ x) @ (x.T @ r)   # (1+K) × N
    alphas = coef[0]
    resid = r - x @ coef

    sigma = resid.T @ resid / t
    mu = f.mean(axis=0)
    fc = f - mu
    omega = fc.T @ fc / t

    sigma_inv = np.linalg.pinv(sigma)
    omega_inv = np.linalg.pinv(omega)

    a_term = float(alphas @ sigma_inv @ alphas)
    f_term = float(mu @ omega_inv @ mu)
    stat = (t / n) * ((t - n - k) / (t - k - 1)) * a_term / (1.0 + f_term)
    pval = float(sps.f.sf(stat, n, t - n - k))

    return GRSResult(
        statistic=float(stat),
        pvalue=pval,
        n_assets=n,
        n_factors=k,
        nobs=t,
        alphas=alphas,
        mean_abs_alpha=float(np.abs(alphas).mean()),
        sr_squared_alpha=a_term,
    )
