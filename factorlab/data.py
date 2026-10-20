"""
factorlab.data — 價格面板的唯一入口。

兩件事：
1. 價格一律走中央 loader `aws_fetch_price_data.fetch_fmp_price.price`。
   本模組**不含任何 parquet 檔名或路徑字串**；要換資料來源只改中央 loader。
2. 把 loader 吐出的 MultiIndex 寬表整理成「日頻寬矩陣 + 月頻寬矩陣」。
   原始碼用 melt 成長表再 groupby，在 2,661 檔 × 4,530 天上又慢又容易
   在 reindex 時悄悄對齊錯；寬矩陣讓所有橫斷面運算都是一行 pandas。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataSpec

# 中央 loader 所在的程式碼根目錄。這裡找的是「套件根」，不是資料檔 ——
# 資料檔路徑的唯一知情者永遠是中央 loader 自己。
_KICKS_ENV = "FACTORLAB_KICKS_ROOT"
_KICKS_DEFAULT = Path.home() / "Desktop" / "KICKs"

# 套用國別篩選後至少要剩這麼多檔，否則視為 FMP_stocks.csv 與價格面板的
# ticker 命名不相容，寧可不篩也不要把 universe 砍到無法做橫斷面研究。
_MIN_TICKERS_AFTER_COUNTRY_FILTER = 200


def _import_central_loader():
    """匯入中央價格 loader，失敗時給出可操作的錯誤訊息。"""
    try:
        from aws_fetch_price_data.fetch_fmp_price import price as _price  # noqa: WPS433
        return _price
    except ImportError:
        pass

    root = Path(os.environ.get(_KICKS_ENV, _KICKS_DEFAULT)).expanduser()
    if (root / "aws_fetch_price_data" / "fetch_fmp_price" / "price.py").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from aws_fetch_price_data.fetch_fmp_price import price as _price  # noqa: WPS433
        return _price

    raise ImportError(
        "找不到中央價格 loader aws_fetch_price_data.fetch_fmp_price.price。\n"
        f"請把 KICKs 專案根目錄加進 PYTHONPATH，或設環境變數 {_KICKS_ENV}=<KICKs 根目錄>。\n"
        "本專案刻意不內建 parquet 路徑：美股價格的唯一出口是那個中央 loader。"
    )


@dataclass
class Panel:
    """對齊好的價格面板。所有矩陣都是 index=日期 / columns=ticker 的寬表。

    Attributes
    ----------
    close, volume : 日頻收盤價與成交量（已清理）
    dret          : 日頻簡單報酬
    mclose        : 月末收盤價（index 為月末交易日）
    mret          : 月報酬，mret.loc[m] = mclose[m]/mclose[m-1] - 1
    dollar_vol    : 日頻成交金額 close * volume
    adv           : 每個月的日均成交金額（index 同 mclose）
    valid         : 月頻布林表，該 ticker 當月是否有足夠交易日
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    dret: pd.DataFrame
    mclose: pd.DataFrame
    mret: pd.DataFrame
    dollar_vol: pd.DataFrame
    adv: pd.DataFrame
    valid: pd.DataFrame
    spec: DataSpec

    @property
    def months(self) -> pd.DatetimeIndex:
        return self.mret.index

    @property
    def n_months(self) -> int:
        return len(self.mret)

    def describe(self) -> dict:
        return {
            "history": self.spec.history,
            "n_days": int(len(self.close)),
            "n_months": int(self.n_months),
            "n_tickers": int(self.close.shape[1]),
            "date_start": str(self.close.index.min().date()),
            "date_end": str(self.close.index.max().date()),
            "avg_valid_tickers_per_month": float(self.valid.sum(axis=1).mean()),
        }


def _extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """從 MultiIndex 欄位取出某個 OHLCV 欄位，兩種 level 排列都接。"""
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("中央 loader 回傳的不是 MultiIndex 欄位，無法判斷 OHLCV 層級。")
    lvl0 = raw.columns.get_level_values(0)
    if field in set(lvl0):
        out = raw[field]
    else:
        lvl1 = raw.columns.get_level_values(1)
        if field not in set(lvl1):
            raise ValueError(f"面板裡找不到欄位 {field!r}。")
        out = raw.swaplevel(axis=1)[field]
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()


def load_panel(spec: DataSpec | None = None) -> Panel:
    """載入並清理價格面板。"""
    spec = spec or DataSpec()
    loader = _import_central_loader()

    raw = loader.load_us_price_data(history=spec.history)
    close = _extract_field(raw, "Close").astype("float64")
    volume = _extract_field(raw, "Volume").astype("float64")
    volume = volume.reindex(columns=close.columns)

    if spec.start is not None:
        close = close.loc[close.index >= pd.Timestamp(spec.start)]
    if spec.end is not None:
        close = close.loc[close.index <= pd.Timestamp(spec.end)]
    volume = volume.reindex(index=close.index)

    # 國別/幣別篩選：非美國公司一律排除（ADR 幣別問題，見中央 loader 說明）。
    if spec.usd_only:
        try:
            usd = loader.get_usd_only_tickers()
        except FileNotFoundError:
            usd = None
        if usd:
            keep = [c for c in close.columns if c in usd]
            if len(keep) >= _MIN_TICKERS_AFTER_COUNTRY_FILTER:
                close = close[keep]
                volume = volume[keep]

    # 非正價格視為缺值；低價股剔除（避免 1 美元以下的報價跳動主導橫斷面排序）。
    close = close.where(close > 0)
    close = close.where(close >= spec.min_price)
    volume = volume.where(volume >= 0)

    dret = close.pct_change(fill_method=None)
    # 日報酬離群值：|r| > max_abs_daily_return 幾乎都是未調整的拆股/反向分割。
    dret = dret.where(dret.abs() <= spec.max_abs_daily_return)

    dollar_vol = close * volume

    grouper = close.index.to_period("M")
    # 月末交易日（而非日曆月底）—— 再平衡只能發生在有交易的日子。
    month_end_dates = pd.DatetimeIndex(
        pd.Series(close.index, index=grouper).groupby(level=0).max().values
    ).sort_values()

    n_days = close.notna().groupby(grouper).sum()
    n_days.index = month_end_dates
    valid = n_days >= spec.min_trading_days_per_month

    mclose = close.groupby(grouper).last()
    mclose.index = month_end_dates
    mclose = mclose.where(valid)

    adv = dollar_vol.groupby(grouper).mean()
    adv.index = month_end_dates
    adv = adv.where(valid)

    # 月報酬由月末價算，但只在「連續兩個月都有效」時才成立；
    # 中間斷月（停牌、資料缺口）不可以用 ffill 假裝有報酬。
    mret = mclose / mclose.shift(1) - 1.0
    mret = mret.where(valid & valid.shift(1).fillna(False))
    mret = mret.where(mret.abs() <= 3.0)

    return Panel(
        close=close,
        volume=volume,
        dret=dret,
        mclose=mclose,
        mret=mret,
        dollar_vol=dollar_vol,
        adv=adv,
        valid=valid,
        spec=spec,
    )


def realized_vol(panel: Panel, window: int = 60, min_periods: int = 40) -> pd.DataFrame:
    """月末觀察到的 window 日已實現波動（年化），對齊到月頻。

    給 benchmarks 的波動度因子與 signals 的控制變數用。用日報酬算、
    在月末取樣，所以不含任何未來資訊。
    """
    rv = panel.dret.rolling(window, min_periods=min_periods).std() * np.sqrt(252.0)
    out = rv.reindex(panel.mclose.index)
    return out.where(panel.valid)
