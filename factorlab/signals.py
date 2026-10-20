"""
factorlab.signals — 動能訊號與流動性篩選。

這個模組修掉了原始 us_factor_liqm.py 的兩個實作 bug：

bug 1：skip 被放在「持有期」而不是「形成期」。
    原本 `start_hold = rebalance_date + 1M if skip_holding_month`，
    等於把再平衡日往後推一個月 —— 那不是 skip，那是遲延進場（而且順手
    製造了一段看不見的 implementation lag）。Jegadeesh-Titman 的 12-2
    動能是跳過**形成期最後一個月**：訊號 = P[t-1]/P[t-12] - 1，
    持有從 t 開始。目的是避開一個月短期反轉與買賣價差的 bid-ask bounce，
    和進場時點無關。本模組的 skip 一律作用在形成期尾端。

bug 2：流動性篩選用 `form_start - 1M`。
    原本是用「形成期開始前一個月」的成交金額當篩選依據，lookback=12 時
    那是 13 個月前的流動性 —— 既不是當下可觀察到的交易條件，也讓不同
    lookback 的策略面對完全不同的 universe，使跨 lookback 比較失去意義。
    正確做法是用再平衡當月（t 月）的日均成交金額：那是下單當下真實
    能看到的流動性，而且它對所有 lookback 都是同一個 universe。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SignalSpec
from .data import Panel


@dataclass
class SignalPanel:
    """某個 (lookback, skip) 設定下的訊號面板。

    Attributes
    ----------
    signal   : 月頻寬表，index = 再平衡月末，columns = ticker。
               只在「通過流動性篩選 + 訊號可算」的格子上有值。
    universe : 布林寬表，該月通過流動性篩選的股票。
    raw      : 未經 universe 篩選的訊號（Fama-MacBeth 的對照用）。
    n_stocks : 每月進入分組的股票數。
    """

    signal: pd.DataFrame
    universe: pd.DataFrame
    raw: pd.DataFrame
    n_stocks: pd.Series
    spec: SignalSpec

    @property
    def valid_months(self) -> pd.DatetimeIndex:
        return self.n_stocks.index[self.n_stocks >= self.spec.min_stocks]


def momentum_signal(panel: Panel, lookback: int, skip: int = 0) -> pd.DataFrame:
    """形成期動能：P[t-skip] / P[t-lookback] - 1。

    用月末價的位移相除，而不是「在日頻資料裡切一段 slice 取 first/last」。
    原始碼的 first/last 做法有個隱性問題：slice 的兩端是用日曆日
    (`rebalance_date - DateOffset(months=lookback)`) 去切的，遇到假日
    就會滑到不同的交易日，不同 ticker 的形成期長度因此不一致。

    skip 必須小於 lookback，否則形成期長度 <= 0。
    """
    if lookback <= 0:
        raise ValueError(f"lookback 必須為正，收到 {lookback}")
    if skip < 0:
        raise ValueError(f"skip 不可為負，收到 {skip}")
    if skip >= lookback:
        raise ValueError(
            f"skip ({skip}) 必須小於 lookback ({lookback})，否則形成期長度為零或負。"
        )

    mclose = panel.mclose
    end = mclose.shift(skip)          # 形成期終點：t-skip
    start = mclose.shift(lookback)    # 形成期起點：t-lookback
    sig = end / start - 1.0
    # 形成期內若有任一端缺值，訊號無效（不 ffill —— 停牌期間的動能是假的）。
    return sig.where(np.isfinite(sig))


def liquidity_universe(
    panel: Panel,
    quantile: float = 0.5,
    min_stocks: int = 50,
) -> pd.DataFrame:
    """流動性 universe：再平衡當月日均成交金額 >= 橫斷面分位數。

    注意 quantile 是**橫斷面分位數**，逐月重算。用固定美元門檻會讓
    universe 在 2008 與 2024 之間差一個數量級。
    """
    if not 0.0 <= quantile < 1.0:
        raise ValueError(f"liquidity_quantile 必須在 [0, 1)，收到 {quantile}")

    adv = panel.adv
    cutoff = adv.quantile(quantile, axis=1)
    universe = adv.ge(cutoff, axis=0) & adv.notna()
    # 月份內若通過篩選的檔數太少（資料早期），整月作廢而不是硬跑。
    too_thin = universe.sum(axis=1) < min_stocks
    universe.loc[too_thin, :] = False
    return universe


def winsorize_cross_section(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """逐月橫斷面 winsorize。只用在 IC / Fama-MacBeth，不用在分位分組。

    分位分組本身對離群值免疫（它只看排序），所以對分組 winsorize 是多餘的；
    但 Fama-MacBeth 是 OLS，單一檔 +400% 的動能可以自己決定斜率的正負。
    """
    if pct <= 0:
        return df
    lo = df.quantile(pct, axis=1)
    hi = df.quantile(1.0 - pct, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def build_signal_panel(panel: Panel, spec: SignalSpec) -> SignalPanel:
    """組出某個 (lookback, skip) 設定下、已套用流動性篩選的訊號面板。"""
    raw = momentum_signal(panel, spec.lookback, spec.skip)
    universe = liquidity_universe(panel, spec.liquidity_quantile, spec.min_stocks)

    # 還要求「下個月有報酬可以算」，否則這檔股票進了投組卻沒有報酬序列。
    tradable = panel.mret.shift(-1).notna()
    eligible = universe & raw.notna() & tradable

    signal = raw.where(eligible)
    n_stocks = eligible.sum(axis=1)
    return SignalPanel(
        signal=signal,
        universe=universe,
        raw=raw,
        n_stocks=n_stocks,
        spec=spec,
    )


def assign_quantiles(signal: pd.DataFrame, n_quantiles: int = 10, min_stocks: int = 50) -> pd.DataFrame:
    """逐月把訊號切成 n 個分位（0 = 最低 = Q1，n-1 = 最高 = Qn）。

    用 rank(pct=True) 而不是 pd.qcut。差別不是「同值也能切成 n 組」——
    大量同值時，任何方法都不可能切出 n 個等大的組，這是資料的性質，
    不是演算法的選擇。差別在**失敗的方式**：

    - `pd.qcut(..., duplicates='drop')`（原始碼的做法）會丟掉重複的
      分界點，然後把剩下的分界點**重新貼上 Q1..Q10 的標籤**。於是某些
      月份標著 "Q10" 的那一組其實是第 7 分位，而 Q10-Q1 的價差被當成
      十分位價差混進時間序列裡 —— 靜默、且無法事後察覺。
    - rank 法下，分位索引永遠是訊號的單調函數：同值的股票拿到同一個
      分位（這是對的，相同訊號就該得到相同處置），標籤與分布位置的
      對應關係永不錯亂。代價是極端分位可能是空的。

    因此這裡再加一道防線：**若最低或最高分位是空的，整月作廢**。
    空的極端分位會讓 Q10-Q1 變成 NaN 而非錯誤的數字，但作廢是更明確的
    處理 —— 那個月根本沒有「十分位兩端」可言。真實資料（橫斷面上千檔
    連續的動能值）幾乎不會觸發這條，但零成交量造成動能恰為 0 的月份會。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少要 2")

    counts = signal.notna().sum(axis=1)
    usable = counts >= max(min_stocks, n_quantiles)

    ranks = signal.rank(axis=1, method="average", pct=True, na_option="keep")
    # pct 落在 (0, 1]；乘上 n 後用 ceil 分組，再夾到 [0, n-1]。
    q = np.ceil(ranks * n_quantiles) - 1.0
    q = q.clip(lower=0, upper=n_quantiles - 1)
    q = q.where(signal.notna())
    q.loc[~usable, :] = np.nan

    # 極端分位必須都有成員，否則多空價差沒有定義。
    has_bottom = (q == 0).any(axis=1)
    has_top = (q == n_quantiles - 1).any(axis=1)
    q.loc[~(has_bottom & has_top), :] = np.nan
    return q
