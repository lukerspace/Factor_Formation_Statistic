"""
factorlab.costs — 交易成本與容量。

為什麼放進檢定流程
------------------
原始碼完全沒有成本，而且因為它用「每月取樣的重疊 h 月報酬」，結構上
也無法算成本 —— 那種算法沒有「投組」這個物件，只有一堆報酬數字。
日曆時間重疊投組讓週轉率有了定義（見 portfolios），成本才算得出來。

這件事對結論的影響可能是決定性的：一個月換手一次的十分位多空策略，
單邊週轉接近 100%，雙邊 200%；在 6 bps 單邊（1 bps 手續費 + 5 bps
半價差）之下，年化成本約 2 × 12 × 6bps ≈ 1.44%。如果毛報酬的年化
alpha 只有 2~3%，扣掉成本後統計顯著性會直接消失。

持有期越長（h 越大），每月只有 1/h 的部位需要換手，成本線性下降 ——
這是「h=12 看起來比 h=1 好」的一個完全機械性的原因，跟動能的持續性
無關。報告會把毛報酬與淨報酬並列，讓這件事無法被混淆。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostSpec


@dataclass
class CostReport:
    """成本對策略的影響。"""

    avg_monthly_turnover_one_way: float
    annual_turnover_one_way: float
    annual_cost: float
    gross_mean_m: float
    net_mean_m: float
    gross_sharpe: float
    net_sharpe: float
    breakeven_one_way_bps: float
    one_way_bps: float

    def as_dict(self, prefix: str = "cost_") -> dict:
        return {
            f"{prefix}one_way_bps": self.one_way_bps,
            f"{prefix}monthly_turnover": self.avg_monthly_turnover_one_way,
            f"{prefix}annual_turnover": self.annual_turnover_one_way,
            f"{prefix}annual_drag": self.annual_cost,
            f"{prefix}gross_mean_m": self.gross_mean_m,
            f"{prefix}net_mean_m": self.net_mean_m,
            f"{prefix}gross_sharpe": self.gross_sharpe,
            f"{prefix}net_sharpe": self.net_sharpe,
            f"{prefix}breakeven_bps": self.breakeven_one_way_bps,
        }


# 低於這個標準差就當成零變異：常數序列在浮點誤差下會留下 1e-18 等級的
# 標準差，直接相除會吐出 1e15 這種看似驚人的 Sharpe。
_MIN_SD = 1e-12


def _sharpe(x: pd.Series) -> float:
    s = x.dropna()
    if s.empty:
        return float("nan")
    sd = float(s.std(ddof=1))
    if not np.isfinite(sd) or sd < _MIN_SD:
        return float("nan")
    return float(s.mean() / sd * np.sqrt(12.0))


def net_returns(gross: pd.Series, turnover: pd.Series, spec: CostSpec) -> pd.Series:
    """扣除成本後的月報酬。turnover 是單邊週轉率（兩條腿已相加）。"""
    if not spec.enabled:
        return gross.rename(f"{gross.name}_net")
    drag = turnover.reindex(gross.index).fillna(0.0) * (spec.one_way_bps / 10_000.0)
    return (gross - drag).rename(f"{gross.name}_net")


def breakeven_bps(gross: pd.Series, turnover: pd.Series) -> float:
    """讓平均淨報酬剛好為零的單邊成本（bps）。

    這是比「淨 Sharpe」更好用的一個數字：它把結論變成一句可以驗證的
    交易問題 —— 「你的實際單邊成本有沒有低於 X bps？」
    """
    g = gross.dropna()
    tn = turnover.reindex(g.index).fillna(0.0)
    denom = float(tn.mean())
    if denom <= 0:
        return float("nan")
    return float(g.mean() / denom * 10_000.0)


def analyze(gross: pd.Series, turnover: pd.Series, spec: CostSpec) -> CostReport:
    """完整成本分析。"""
    g = gross.dropna()
    tn = turnover.reindex(g.index).fillna(0.0)
    net = net_returns(g, tn, spec)
    monthly_to = float(tn.mean())
    return CostReport(
        avg_monthly_turnover_one_way=monthly_to,
        annual_turnover_one_way=monthly_to * 12.0,
        annual_cost=monthly_to * 12.0 * (spec.one_way_bps / 10_000.0),
        gross_mean_m=float(g.mean()),
        net_mean_m=float(net.mean()),
        gross_sharpe=_sharpe(g),
        net_sharpe=_sharpe(net),
        breakeven_one_way_bps=breakeven_bps(g, tn),
        one_way_bps=spec.one_way_bps,
    )
