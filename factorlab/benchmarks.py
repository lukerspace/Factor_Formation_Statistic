"""
factorlab.benchmarks — 風險因子基準。

為什麼需要
----------
原始碼回報 Q10-Q1 的原始平均數並稱之為策略報酬。但動能的高分位在任何
時點都系統性地偏向某些已知特徵：剛漲過的股票通常 beta 較高、波動較大、
而且（在流動性篩選後仍然）偏小型。Q10-Q1 的平均報酬裡有一部分只是
這些已知風險的補償，不是新資訊。

要主張動能是獨立的現象，必須在控制已知因子後截距仍顯著。本模組用
**同一份價格面板**自建四個因子，不引入外部資料（引入 Fama-French
檔案會帶來另一個資料來源與對齊問題，而且 FF 的 universe 跟這裡的
流動性篩選後 universe 不一致）。

四個因子
--------
MKT  : 流動性 universe 的等權月報酬。用等權而非市值加權，是因為策略
       本身是等權的 —— 基準的加權方式必須跟被解釋的投組一致，否則
       截距會吸收掉「等權 vs 市值加權」的差異（這就是 size 效果）。
SIZE : 規模**代理**。本面板沒有歷史市值（價格 parquet 只有 OHLCV，
       而 FMP_stocks.csv 的市值是當期值，用它會造成前視偏誤）。
       因此用日均成交金額的對數當代理：成交金額與市值的橫斷面相關
       約 0.8~0.9。這是本研究最大的規格妥協，報告的 caveats 會明寫。
STREV: 短期反轉，上個月報酬的反向多空。這個因子特別重要 —— skip=0 的
       動能訊號的形成期包含最近一個月，因此會跟短期反轉直接衝突，
       控制它之後 skip=0 與 skip=1 的差異才解讀得出來。
VOL  : 低波動多空（做多低波動、做空高波動），用 60 日已實現波動。

所有因子都是日曆時間、非重疊的月頻序列，與被解釋的投組同頻。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel, realized_vol
from .signals import liquidity_universe

FACTOR_NAMES = ("MKT", "SIZE", "STREV", "VOL")


def _long_short_from_characteristic(
    char: pd.DataFrame,
    mret: pd.DataFrame,
    universe: pd.DataFrame,
    quantile: float = 0.3,
    long_high: bool = True,
    min_stocks: int = 50,
) -> pd.Series:
    """用某個月末特徵做三分位多空因子，持有一個月、每月再平衡。

    特徵在 m-1 月末觀察，報酬是 m 月的 —— 不存在同月偷看。
    """
    ch = char.where(universe)
    lo_cut = ch.quantile(quantile, axis=1)
    hi_cut = ch.quantile(1.0 - quantile, axis=1)

    low = ch.le(lo_cut, axis=0) & ch.notna()
    high = ch.ge(hi_cut, axis=0) & ch.notna()

    enough = (low.sum(axis=1) >= min_stocks // 3) & (high.sum(axis=1) >= min_stocks // 3)
    low = low.where(enough, False)
    high = high.where(enough, False)

    # 特徵 shift(1)：m 月的報酬由 m-1 月末的特徵決定。
    r_high = mret.where(high.shift(1).fillna(False)).mean(axis=1)
    r_low = mret.where(low.shift(1).fillna(False)).mean(axis=1)
    out = (r_high - r_low) if long_high else (r_low - r_high)
    return out


def build_factors(
    panel: Panel,
    liquidity_quantile: float = 0.5,
    min_stocks: int = 50,
    vol_window: int = 60,
) -> pd.DataFrame:
    """建出 MKT / SIZE / STREV / VOL 四因子的月報酬表。"""
    universe = liquidity_universe(panel, liquidity_quantile, min_stocks)
    mret = panel.mret

    # MKT：等權市場（universe 在 m-1 月末決定，賺 m 月報酬）。
    in_univ = universe.shift(1).fillna(False)
    mkt = mret.where(in_univ).mean(axis=1).rename("MKT")

    # SIZE：做多小（低成交金額）、做空大。符號與 FF 的 SMB 一致。
    log_adv = np.log(panel.adv.where(panel.adv > 0))
    size = _long_short_from_characteristic(
        log_adv, mret, universe, long_high=False, min_stocks=min_stocks
    ).rename("SIZE")

    # STREV：做多上個月跌的、做空上個月漲的。符號與短期反轉文獻一致。
    prev_ret = mret
    strev = _long_short_from_characteristic(
        prev_ret, mret, universe, long_high=False, min_stocks=min_stocks
    ).rename("STREV")

    # VOL：做多低波動、做空高波動（betting against volatility）。
    rv = realized_vol(panel, window=vol_window)
    vol = _long_short_from_characteristic(
        rv, mret, universe, long_high=False, min_stocks=min_stocks
    ).rename("VOL")

    factors = pd.concat([mkt, size, strev, vol], axis=1)
    return factors.loc[:, list(FACTOR_NAMES)]


def factor_summary(factors: pd.DataFrame) -> pd.DataFrame:
    """因子本身的描述統計 + HAC t 值。讀者得先相信基準本身是合理的。"""
    from .stats.hac import hac_mean_test

    rows = []
    for col in factors.columns:
        s = factors[col].dropna()
        res = hac_mean_test(s)
        sd = float(s.std(ddof=1))
        rows.append(
            {
                "factor": col,
                "nobs": int(len(s)),
                "mean_m": float(s.mean()),
                "mean_ann": float((1.0 + s.mean()) ** 12 - 1.0),
                "vol_ann": sd * np.sqrt(12.0),
                "t_hac": res.tstat,
                "p_hac": res.pvalue,
                "sharpe": float(s.mean() / sd * np.sqrt(12.0)) if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
