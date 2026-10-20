"""
factorlab.config — 研究規格（specification）的唯一定義處。

設計原則：所有「研究者的選擇」都必須是顯式、可序列化的 dataclass 欄位，
不准散落在函式預設值或 module 常數裡。理由很實際 —— 這份研究的核心結論是
「多重檢定會把雜訊變成顯著」，而多重檢定的分母（試了幾組參數）只有在
規格被集中宣告時才算得準。StudySpec.n_trials 就是那個分母。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence
import itertools
import json

History = Literal["3y", "long"]
Kernel = Literal["bartlett", "truncated"]


@dataclass(frozen=True)
class DataSpec:
    """價格面板的取得與清理規格。"""

    # "long" = 2007–2024（215 個月），"3y" = 近 3 年（~34 個月）。
    # 預設 long：3 年樣本對本檔案裡的任何檢定都沒有檢定力（見 stats.power）。
    history: History = "long"
    start: str | None = None
    end: str | None = None
    # 單月最少要有幾個交易日才算這檔股票當月有效（避免剛上市/停牌月汙染）。
    min_trading_days_per_month: int = 15
    # 日報酬絕對值超過此值視為資料錯誤（反向分割未調整等），該日報酬設為 NaN。
    # 注意這個門檻是**不對稱**的：報酬下界是 -100%，所以 1.0 只擋得住上漲
    # 方向的錯誤。10 拆 1 的正向分割呈現 -90%，不會被擋掉。刻意不收緊負向
    # 門檻 —— 那會同時丟掉真實的單日崩跌，而那是動能報酬的一部分。
    max_abs_daily_return: float = 1.0
    # 只保留 country == 'US' 的公司（避免 ADR 幣別問題，見中央 loader docstring）。
    usd_only: bool = True
    min_price: float = 1.0


@dataclass(frozen=True)
class SignalSpec:
    """動能訊號的形成期規格。"""

    lookback: int = 12
    # 形成期要跳過的月數。注意：skip 作用在「形成期的尾端」，不是持有期。
    # Jegadeesh-Titman 的 12-2 動能 = lookback 12、skip 1，訊號為
    # P[t-1]/P[t-12]-1，目的是避開短期反轉與買賣價差跳動。
    skip: int = 1
    # 流動性篩選：以「再平衡當月」的日均成交金額做橫斷面分位數篩選。
    liquidity_quantile: float = 0.5
    min_stocks: int = 50
    # 訊號是否先做橫斷面 winsorize（只影響 Fama-MacBeth 與 IC，不影響分位分組）。
    winsorize: float = 0.01


@dataclass(frozen=True)
class PortfolioSpec:
    """投組建構規格。"""

    n_quantiles: int = 10
    # 持有月數。h > 1 時一律用 Jegadeesh-Titman 日曆時間重疊投組，
    # 不是對重疊的 h 月報酬做 t 檢定。
    horizon: int = 1
    weighting: Literal["equal"] = "equal"
    # 持有期內是否每月重新等權（True）或買入持有、讓權重隨報酬漂移（False）。
    # 預設 False：較低週轉、較貼近可實作的做法。
    monthly_rebalance_within_sleeve: bool = False
    long_quantile: int | None = None   # None → 最高分位
    short_quantile: int | None = None  # None → 最低分位


@dataclass(frozen=True)
class CostSpec:
    """交易成本規格（單邊，bps）。"""

    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    enabled: bool = True

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps


@dataclass(frozen=True)
class StatsSpec:
    """檢定統計量的規格。"""

    # HAC 落後期數；None → Newey-West 自動帶寬（Newey-West 1994 / Andrews）。
    hac_lags: int | None = None
    hac_kernel: Kernel = "bartlett"
    bootstrap_reps: int = 2000
    # 定態 bootstrap 的平均區塊長度（Politis-Romano 的 1/p）。
    bootstrap_block: float = 6.0
    seed: int = 20261020
    alpha: float = 0.05
    # Harvey-Liu-Zhu (2016) 建議的新因子門檻。
    hlz_tstat_threshold: float = 3.0
    power_target: float = 0.80


@dataclass(frozen=True)
class StudySpec:
    """一次完整研究的規格：參數網格 + 統計設定。"""

    data: DataSpec = field(default_factory=DataSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)
    costs: CostSpec = field(default_factory=CostSpec)
    lookbacks: Sequence[int] = (1, 3, 6, 12)
    skips: Sequence[int] = (0, 1)
    horizons: Sequence[int] = (1, 3, 6, 12)
    n_quantiles: int = 10
    liquidity_quantile: float = 0.5
    min_stocks: int = 50

    @property
    def n_trials(self) -> int:
        """多重檢定的分母：這份研究一共掃了幾個 (lookback, skip, horizon) 組合。

        這個數字直接餵進 multiple_testing 與 Deflated Sharpe。全域虛無之下
        E[max|t|] ≈ sqrt(2 log n_trials)，所以 n_trials 被低報的話，
        校正後的門檻就會偏鬆 —— 這是整份研究最容易自欺的一個數字。
        """
        return len(self.lookbacks) * len(self.skips) * len(self.horizons)

    def grid(self) -> list[tuple[int, int, int]]:
        """回傳 (lookback, skip, horizon) 的完整網格。"""
        return list(itertools.product(self.lookbacks, self.skips, self.horizons))

    def signal_grid(self) -> list[tuple[int, int]]:
        """回傳 (lookback, skip) 的網格 —— 訊號只需算一次，與 horizon 無關。"""
        return list(itertools.product(self.lookbacks, self.skips))

    def signal_spec(self, lookback: int, skip: int) -> SignalSpec:
        return SignalSpec(
            lookback=lookback,
            skip=skip,
            liquidity_quantile=self.liquidity_quantile,
            min_stocks=self.min_stocks,
        )

    def portfolio_spec(self, horizon: int) -> PortfolioSpec:
        return PortfolioSpec(n_quantiles=self.n_quantiles, horizon=horizon)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


def strategy_tag(lookback: int, skip: int, horizon: int) -> str:
    """策略標籤。沿用 `12-2` 這種學界慣例讓 skip 的語意不會被誤讀成持有期。"""
    form = f"{lookback}-{skip}" if skip else f"{lookback}"
    return f"mom{form}_h{horizon}"
