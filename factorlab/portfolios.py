"""
factorlab.portfolios — Jegadeesh-Titman 日曆時間重疊投組。

這是整份重構最重要的一塊，因為它決定了後面所有 t 值的分母是否正確。

原始做法的問題
--------------
原始碼每個月形成一次投組，然後記錄 fwd_1m / fwd_3m / fwd_6m / fwd_12m，
再對這些 h 月報酬序列做 `ttest_1samp`。問題是 fwd_12m 每月取樣一次卻
覆蓋 12 個月，相鄰觀察值共用 11 個月的報酬 —— 序列自相關不是「資料
剛好有自相關」，而是抽樣設計造成的結構性重疊。樣本平均數的真實變異數是

    Var(x̄) = (1/T)·[γ0 + 2·Σ_{j=1}^{h-1} γ_j]

而 ttest_1samp 用的是 γ0/T。重疊的 h 月報酬在真實報酬近似 i.i.d. 時
γ_j ≈ (1 - j/h)·γ0，代入可得 Var 被低估約 h 倍，t 值因此被高估約 √h 倍。
fwd_12m 的 t 值要除以約 3.46 才是對的。

兩條正確的路，本模組走第二條當主結果
------------------------------------
(a) 保留重疊報酬，但用 Newey-West 或 Hansen-Hodrick(h-1) 修正標準誤。
    → 見 stats.hac，本模組仍提供 `overlapping_forward_returns` 讓報告
      可以並列「天真 t」與「HAC t」，直接把膨脹倍數攤出來。
(b) 改造投組本身，讓觀察值不重疊：每個月同時持有 h 個「分批投組」
    （sleeve），分別在 t-1, …, t-h 形成，各佔 1/h 權重。這樣每個月的
    組合報酬是一個真正的月頻、非重疊觀察值，平均數的變異數就是 γ0/T，
    普通 t 檢定成立（殘餘的自相關再用小落後期的 NW 兜一下即可）。
    這就是 Jegadeesh & Titman (1993) 的 overlapping portfolio 做法，
    也是本模組的主結果。

附帶的好處是 (b) 讓週轉率與交易成本有定義 —— 每個月只有 1/h 的部位
需要換手，重疊 h 月報酬那種算法根本無法回答「這要付多少成本」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PortfolioSpec


@dataclass
class QuantilePortfolios:
    """日曆時間分位投組的結果。

    Attributes
    ----------
    returns   : DataFrame，index = 月份，columns = 0..n-1（0 = 最低分位）。
                每一格是該月「h 個 sleeve 平均」後的組合月報酬（毛報酬）。
    long_short: Series，最高分位 - 最低分位的月報酬。
    turnover  : DataFrame，每個分位每月的單邊週轉率（0.5·Σ|Δw|）。
    counts    : DataFrame，每個分位每月的平均持股數。
    n_sleeves : Series，每月實際活著的 sleeve 數（期初會少於 h）。
    """

    returns: pd.DataFrame
    long_short: pd.Series
    turnover: pd.DataFrame
    counts: pd.DataFrame
    n_sleeves: pd.Series
    spec: PortfolioSpec
    meta: dict = field(default_factory=dict)

    @property
    def ls_turnover(self) -> pd.Series:
        """多空組合的單邊總週轉（兩條腿相加）。"""
        lo, hi = _legs(self.spec, self.returns.shape[1])
        return self.turnover[hi] + self.turnover[lo]


def _legs(spec: PortfolioSpec, n_quantiles: int) -> tuple[int, int]:
    lo = spec.short_quantile if spec.short_quantile is not None else 0
    hi = spec.long_quantile if spec.long_quantile is not None else n_quantiles - 1
    return lo, hi


class _Sleeve:
    """一個在某月形成、持有 h 個月的分批投組。

    權重矩陣 W 的形狀是 (n_tickers, n_quantiles)，每一欄自己加總為 1。
    """

    __slots__ = ("w", "age")

    def __init__(self, w: np.ndarray) -> None:
        self.w = w
        self.age = 0

    def step(self, ret: np.ndarray, monthly_rebalance: bool) -> tuple[np.ndarray, np.ndarray]:
        """推進一個月，回傳 (各分位報酬, 各分位有效持股數)。

        缺值處理：某檔股票當月沒有報酬（下市、停牌、資料缺口）時，把它
        從該月的加權平均中剔除並把權重重新歸一化，而不是當成 0% 報酬。
        當成 0 會系統性地把下市股票的報酬往上偏（真實下市多半是大跌），
        剔除則是把它當「無法觀察」。兩種都不完美，但剔除是實務上的標準
        做法，而且這裡的 universe 已經是流動性前 50%，下市污染很小。
        這個選擇會記在報告的 caveats 裡。
        """
        alive = np.isfinite(ret)[:, None] & (self.w > 0)
        r = np.where(np.isfinite(ret), ret, 0.0)[:, None]

        w_alive = np.where(alive, self.w, 0.0)
        denom = w_alive.sum(axis=0)
        numer = (w_alive * r).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            port_ret = np.where(denom > 0, numer / denom, np.nan)
        counts = alive.sum(axis=0).astype("float64")

        if monthly_rebalance:
            # 每月在存活的持股上重新等權。
            new_w = np.where(alive, 1.0, 0.0)
        else:
            # 買入持有：權重隨個股報酬漂移。
            new_w = w_alive * (1.0 + r)
        tot = new_w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            new_w = np.where(tot > 0, new_w / tot, 0.0)
        self.w = new_w
        self.age += 1
        return port_ret, counts


def _membership_weights(q_row: np.ndarray, n_quantiles: int) -> np.ndarray | None:
    """把某月的分位標籤轉成等權權重矩陣 (n_tickers, n_quantiles)。"""
    w = np.zeros((q_row.shape[0], n_quantiles), dtype="float64")
    ok = np.isfinite(q_row)
    if not ok.any():
        return None
    idx = q_row[ok].astype("int64")
    w[np.where(ok)[0], idx] = 1.0
    tot = w.sum(axis=0)
    if (tot == 0).any():
        return None
    return w / tot


def build_calendar_time(
    quantiles: pd.DataFrame,
    mret: pd.DataFrame,
    spec: PortfolioSpec,
) -> QuantilePortfolios:
    """建構日曆時間重疊投組。

    Parameters
    ----------
    quantiles : 月頻分位標籤（0..n-1，NaN = 不在 universe），index 為形成月末。
    mret      : 月報酬寬表，mret.loc[m] 是 m-1 月末到 m 月末的報酬。
    spec      : 投組規格（分位數、持有月數、權重方式）。

    Notes
    -----
    時序約定：在 m-1 月末用當時可觀察的資訊形成投組，賺取 m 月的報酬。
    所以 sleeve 在形成月的**下一個月**才開始貢獻報酬，不存在同月偷看。
    """
    h = int(spec.horizon)
    if h < 1:
        raise ValueError("horizon 至少為 1")
    n_q = int(spec.n_quantiles)

    quantiles = quantiles.reindex(columns=mret.columns)
    months = mret.index
    q_vals = quantiles.reindex(index=months).to_numpy(dtype="float64")
    r_vals = mret.to_numpy(dtype="float64")

    sleeves: list[_Sleeve] = []
    ret_rows: list[np.ndarray] = []
    cnt_rows: list[np.ndarray] = []
    turn_rows: list[np.ndarray] = []
    n_sleeve_list: list[int] = []

    # 上個月底實際持有的聚合權重（已漂移），用來算週轉。
    prev_agg: np.ndarray | None = None

    for m in range(len(months)):
        # 1) 用 m-1 月末的訊號開一個新 sleeve（m=0 時沒有前一個月）。
        if m >= 1:
            w0 = _membership_weights(q_vals[m - 1], n_q)
            if w0 is not None:
                sleeves.append(_Sleeve(w0))

        # 2) 汰除已持滿 h 個月的 sleeve。
        sleeves = [s for s in sleeves if s.age < h]

        if not sleeves:
            ret_rows.append(np.full(n_q, np.nan))
            cnt_rows.append(np.full(n_q, np.nan))
            turn_rows.append(np.full(n_q, np.nan))
            n_sleeve_list.append(0)
            prev_agg = None
            continue

        # 3) 月初的目標聚合權重 = 活著的 sleeve 權重平均（各 1/n_sleeves）。
        target_agg = np.mean([s.w for s in sleeves], axis=0)
        if prev_agg is None:
            turn = np.full(n_q, np.nan)
        else:
            turn = 0.5 * np.abs(target_agg - prev_agg).sum(axis=0)
        turn_rows.append(turn)
        n_sleeve_list.append(len(sleeves))

        # 4) 推進一個月，取各 sleeve 報酬的等權平均。
        rets = np.empty((len(sleeves), n_q))
        cnts = np.empty((len(sleeves), n_q))
        for i, s in enumerate(sleeves):
            rets[i], cnts[i] = s.step(r_vals[m], spec.monthly_rebalance_within_sleeve)
        with np.errstate(invalid="ignore"):
            ret_rows.append(np.nanmean(rets, axis=0))
            cnt_rows.append(np.nanmean(cnts, axis=0))

        # 5) 月末漂移後的聚合權重，作為下個月週轉的比較基準。
        prev_agg = np.mean([s.w for s in sleeves], axis=0)

    cols = list(range(n_q))
    returns = pd.DataFrame(np.vstack(ret_rows), index=months, columns=cols)
    counts = pd.DataFrame(np.vstack(cnt_rows), index=months, columns=cols)
    turnover = pd.DataFrame(np.vstack(turn_rows), index=months, columns=cols)
    n_sleeves = pd.Series(n_sleeve_list, index=months, name="n_sleeves")

    lo, hi = _legs(spec, n_q)
    long_short = (returns[hi] - returns[lo]).rename("long_short")

    # 期初暖機期（sleeve 還沒滿 h 個）剔除：那段時間的組合不是目標策略。
    warm = n_sleeves >= min(h, int(n_sleeves.max()) if len(n_sleeves) else h)
    meta = {
        "horizon": h,
        "n_quantiles": n_q,
        "long_leg": hi,
        "short_leg": lo,
        "warmup_months": int((~warm).sum()),
        "monthly_rebalance_within_sleeve": spec.monthly_rebalance_within_sleeve,
    }

    return QuantilePortfolios(
        returns=returns,
        long_short=long_short,
        turnover=turnover,
        counts=counts,
        n_sleeves=n_sleeves,
        spec=spec,
        meta=meta,
    )


def overlapping_forward_returns(
    quantiles: pd.DataFrame,
    mret: pd.DataFrame,
    horizon: int,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """原始碼那套「每月取樣、持有 h 個月」的重疊報酬，保留下來做對照。

    回傳 index = 形成月末、columns = 分位的 h 月累積報酬。這組數字**不可以**
    直接做 ttest_1samp（這正是原始碼的錯誤）；它存在的唯一目的是讓報告能
    並列「天真 t」與「NW / HH 修正後的 t」，把 √h 的膨脹量化給讀者看。
    """
    h = int(horizon)
    # 未來 h 個月的累積報酬：以 m 月末為起點，賺 m+1..m+h 的報酬。
    log1p = np.log1p(mret)
    fwd = log1p.rolling(h, min_periods=h).sum().shift(-h)
    fwd = np.expm1(fwd)

    out = {}
    q = quantiles.reindex(index=mret.index, columns=mret.columns)
    for qi in range(n_quantiles):
        mask = q == qi
        vals = fwd.where(mask)
        out[qi] = vals.mean(axis=1)
    res = pd.DataFrame(out, index=mret.index)
    return res.dropna(how="all")


def apply_costs(gross: pd.Series, turnover: pd.Series, one_way_bps: float) -> pd.Series:
    """把單邊週轉率換算成成本後扣掉，得到淨報酬。"""
    cost = turnover.reindex(gross.index).fillna(0.0) * (one_way_bps / 10_000.0)
    return (gross - cost).rename(f"{gross.name}_net")
