"""
factorlab.stats.famamacbeth — Fama-MacBeth (1973) 兩階段橫斷面迴歸。

分位投組只用到訊號的**排序**，等於把連續的訊號強度壓成 10 個桶子，
而且無法同時控制其他特徵。Fama-MacBeth 補上這兩塊：

1. 逐月跑橫斷面迴歸 r_{i,t+1} = γ_0t + γ_1t·signal_{i,t} + controls + ε
2. 對 γ_1t 的時間序列取平均並做 HAC 檢定

γ_1 的意義是「訊號每高一個單位，下個月報酬高多少」，而且是在控制了
規模代理、短期反轉、波動度之後的淨效果。這跟 Q10-Q1 的差值是兩種不同
的加權（FM 是最小平方加權，分位投組是等權），兩者都顯著才算穩健。

標準誤一律用 Newey-West：Fama-MacBeth 原始做法假設 γ_t 無自相關，
但月頻特徵（尤其波動度、規模）變動緩慢，γ_t 實務上有明顯自相關。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hac import hac_mean_test


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth 結果。"""

    names: list[str]
    means: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    ses: np.ndarray
    nobs_months: int
    avg_n_stocks: float
    avg_r2: float
    lags: int
    coefs: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"gamma": self.means, "se": self.ses, "t": self.tstats, "p": self.pvalues},
            index=self.names,
        )

    def as_dict(self, prefix: str = "fm_") -> dict:
        out = {
            f"{prefix}months": self.nobs_months,
            f"{prefix}avg_n_stocks": self.avg_n_stocks,
            f"{prefix}avg_r2": self.avg_r2,
            f"{prefix}hac_lags": self.lags,
        }
        for n, g, t, p in zip(self.names, self.means, self.tstats, self.pvalues):
            key = "signal" if n == "signal" else n
            out[f"{prefix}{key}_gamma"] = float(g)
            out[f"{prefix}{key}_t"] = float(t)
            out[f"{prefix}{key}_p"] = float(p)
        return out


def _standardize_row(v: np.ndarray) -> np.ndarray:
    """橫斷面 z 分數。讓不同月份、不同單位的特徵係數可以相加比較。"""
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    mu = v[ok].mean()
    sd = v[ok].std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(v, np.nan)
    out = np.full_like(v, np.nan)
    out[ok] = (v[ok] - mu) / sd
    return out


def fama_macbeth(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    controls: dict[str, pd.DataFrame] | None = None,
    min_stocks: int = 50,
    lags: int | None = None,
    standardize: bool = True,
) -> FamaMacBethResult:
    """兩階段橫斷面迴歸。

    Parameters
    ----------
    signal      : 月頻訊號寬表（index = 月末，columns = ticker）。
    fwd_returns : 與 signal 同形狀的「下期報酬」寬表，呼叫端負責對齊
                  （通常是 panel.mret.shift(-1)）。這裡刻意不自己 shift，
                  免得對齊責任分散在兩個地方。
    controls    : 名稱 → 同形狀寬表。會跟 signal 一起 z 分數化。
    """
    controls = controls or {}
    names = ["intercept", "signal"] + list(controls)

    months = signal.index.intersection(fwd_returns.index)
    cols = signal.columns.intersection(fwd_returns.columns)

    rows: list[np.ndarray] = []
    r2s: list[float] = []
    ns: list[int] = []
    idx: list[pd.Timestamp] = []

    sig = signal.loc[months, cols]
    fwd = fwd_returns.loc[months, cols]
    ctrl = {k: v.reindex(index=months, columns=cols) for k, v in controls.items()}

    for m in months:
        y = fwd.loc[m].to_numpy(dtype="float64")
        feats = [sig.loc[m].to_numpy(dtype="float64")]
        feats += [ctrl[k].loc[m].to_numpy(dtype="float64") for k in controls]
        if standardize:
            feats = [_standardize_row(f) for f in feats]

        x = np.column_stack(feats)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.sum() < max(min_stocks, x.shape[1] + 2):
            continue

        y_v = y[ok]
        x_v = np.column_stack([np.ones(ok.sum()), x[ok]])
        beta = np.linalg.pinv(x_v.T @ x_v) @ (x_v.T @ y_v)
        resid = y_v - x_v @ beta
        tss = float(((y_v - y_v.mean()) ** 2).sum())
        r2s.append(1.0 - float((resid**2).sum()) / tss if tss > 0 else np.nan)
        rows.append(beta)
        ns.append(int(ok.sum()))
        idx.append(m)

    if not rows:
        raise ValueError("Fama-MacBeth 沒有任何月份達到最低股票數要求")

    coefs = pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(idx), columns=names)

    means, ts, ps, ses = [], [], [], []
    lags_used = 0
    for n in names:
        res = hac_mean_test(coefs[n], lags=lags)
        means.append(res.mean)
        ts.append(res.tstat)
        ps.append(res.pvalue)
        ses.append(res.se)
        lags_used = res.lags

    return FamaMacBethResult(
        names=names,
        means=np.asarray(means),
        tstats=np.asarray(ts),
        pvalues=np.asarray(ps),
        ses=np.asarray(ses),
        nobs_months=len(coefs),
        avg_n_stocks=float(np.mean(ns)),
        avg_r2=float(np.nanmean(r2s)) if r2s else float("nan"),
        lags=lags_used,
        coefs=coefs,
    )
