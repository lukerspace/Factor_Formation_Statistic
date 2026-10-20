"""
factorlab.stats — 檢驗統計量。

這個子套件存在的理由，就是原始 us_factor_liqm.py 最後那段
`scipy.stats.ttest_1samp(x, 0)`。那一行有五個致命問題：

1. 重疊報酬 → t 值被高估約 √h 倍                     → hac
2. 沒有風險調整，原始平均數不是 alpha                 → regression, benchmarks
3. 沒有多重檢定校正，32 個檢定本來就會跑出顯著         → multiple_testing, spa
4. 只看兩端、沒檢驗單調性                             → monotonicity, regression.grs_test
5. 樣本只有 23–34 個月，「不顯著」= 測不出來而非沒效果  → power

每個模組的 docstring 都寫明它修的是哪一個問題、以及為什麼那個修法是對的。
"""

from . import (
    bootstrap,
    famamacbeth,
    hac,
    ic,
    moments,
    monotonicity,
    multiple_testing,
    power,
    regression,
    spa,
)

__all__ = [
    "bootstrap",
    "famamacbeth",
    "hac",
    "ic",
    "moments",
    "monotonicity",
    "multiple_testing",
    "power",
    "regression",
    "spa",
]
