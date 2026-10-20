"""
factorlab — 多重回看期動能因子的研究級檢驗框架。

這個套件是 `us_factor_liqm.py`（187 行單檔）的重構。重構的動機不是
「把長檔案拆小」，而是原始檔案的**檢驗統計量全部是錯的**，而要修對
就必須先讓資料、訊號、投組、基準、成本各自成為可以單獨測試的物件。

原始檔案的五個致命問題，與各自的解法：

| # | 問題 | 解法所在 |
|---|---|---|
| 1 | 重疊報酬做 `ttest_1samp`，t 值高估約 √h 倍 | `portfolios`（日曆時間重疊投組）、`stats.hac`（NW / HH） |
| 2 | 回報原始平均數而非風險調整後 alpha | `benchmarks`、`stats.regression` |
| 3 | 32 個檢定沒有多重檢定校正 | `stats.multiple_testing`、`stats.spa` |
| 4 | 只看 Q10−Q1 兩端，沒檢驗單調性 | `stats.monotonicity`、`stats.regression.grs_test` |
| 5 | 樣本 23–34 個月，「不顯著」被誤讀成「沒效果」 | `stats.power` |

另加兩個實作 bug：skip 被放在持有期而非形成期、流動性篩選用錯時點，
都在 `signals` 修正並在 docstring 說明。

典型用法：

    from factorlab import StudySpec, run_study, write_all

    result = run_study(StudySpec())          # 預設 history="long"
    write_all(result, "output")

或用命令列：

    python -m factorlab.cli run --history long

美股價格一律透過中央 loader
`aws_fetch_price_data.fetch_fmp_price.price` 取得；本套件不含任何
parquet 路徑字串。
"""

from .config import (
    CostSpec,
    DataSpec,
    PortfolioSpec,
    SignalSpec,
    StatsSpec,
    StudySpec,
    strategy_tag,
)
from .data import Panel, load_panel
from .report import write_all, write_excel, write_markdown
from .study import StrategyResult, StudyResult, run_study

__version__ = "1.0.0"

__all__ = [
    "CostSpec",
    "DataSpec",
    "Panel",
    "PortfolioSpec",
    "SignalSpec",
    "StatsSpec",
    "StrategyResult",
    "StudyResult",
    "StudySpec",
    "load_panel",
    "run_study",
    "strategy_tag",
    "write_all",
    "write_excel",
    "write_markdown",
    "__version__",
]
