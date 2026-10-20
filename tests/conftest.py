"""測試共用的合成資料。刻意不碰真實 parquet：測試必須離線、秒級完成。"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def months() -> pd.DatetimeIndex:
    return pd.date_range("2007-01-31", periods=180, freq="ME")


def make_overlapping(x: np.ndarray, h: int) -> np.ndarray:
    """把 i.i.d. 序列變成 h 期重疊的滾動和 —— 模擬 fwd_h 月報酬的抽樣結構。"""
    if h <= 1:
        return x.copy()
    n = x.size - h + 1
    return np.array([x[t : t + h].sum() for t in range(n)])


@pytest.fixture
def two_ticker_panel(months):
    """兩檔股票、報酬固定：A 每月 +2%，B 每月 -2%。投組報酬完全可預測。"""
    mret = pd.DataFrame(
        {"A": np.full(len(months), 0.02), "B": np.full(len(months), -0.02)},
        index=months,
    )
    # A 永遠在高分位(1)、B 永遠在低分位(0)
    quantiles = pd.DataFrame(
        {"A": np.ones(len(months)), "B": np.zeros(len(months))}, index=months
    )
    return mret, quantiles
