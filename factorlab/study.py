"""
factorlab.study — 流程編排。

把 data → signals → portfolios → benchmarks → costs → stats 串成一次
完整研究，並且**在正確的層級**套用每一種校正：

- 策略層（每個 lookback × skip × horizon）
    日曆時間非重疊多空報酬 → HAC t 值、風險調整 alpha、單調性、
    GRS、成本、檢定力。另外並列「原始碼那套重疊報酬的天真 t」
    與其 NW / HH 修正值，把 √h 膨脹攤開。
- 訊號層（每個 lookback × skip）
    IC 與 Fama-MacBeth —— 這兩個不依賴 horizon，用到全部股票而非兩端。
- 研究層（所有策略一起）
    多重檢定校正、HLZ haircut、Deflated Sharpe、Hansen SPA。
    這一層才是回答「這份研究整體上發現了什麼」的地方；
    單看任何一個策略的 p 值都是在挑選後的樣本上做推論。

執行順序有意義：先算完所有策略，再做研究層校正。不可以邊跑邊挑
「看起來不錯的」然後只對它做校正 —— 那會讓分母失真。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import warnings

import numpy as np
import pandas as pd

from .benchmarks import build_factors, factor_summary
from .config import PortfolioSpec, StudySpec, strategy_tag
from .costs import CostReport, analyze as analyze_costs, net_returns
from .data import Panel, load_panel, realized_vol
from .portfolios import (
    QuantilePortfolios,
    build_calendar_time,
    overlapping_forward_returns,
)
from .signals import (
    SignalPanel,
    assign_quantiles,
    build_signal_panel,
    winsorize_cross_section,
)
from .stats.bootstrap import BootstrapCI, bootstrap_mean, bootstrap_sharpe
from .stats.famamacbeth import FamaMacBethResult, fama_macbeth
from .stats.hac import (
    HACResult,
    hac_mean_test,
    hansen_hodrick_mean_test,
    inflation_factor,
    newey_west_mean_test,
)
from .stats.ic import ICResult, information_coefficient
from .stats.moments import Moments, describe
from .stats.monotonicity import MRResult, mr_test, quantile_profile
from .stats.multiple_testing import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_max_abs_t,
    haircut_sharpe,
)
from .stats.power import PowerResult, diagnose as power_diagnose, power_table
from .stats.regression import GRSResult, RegressionResult, factor_regression, grs_test
from .stats.spa import SPAResult, spa_test


@dataclass
class SignalLevelResult:
    """訊號層（與 horizon 無關）的結果。"""

    lookback: int
    skip: int
    ic_spearman: ICResult | None
    ic_pearson: ICResult | None
    fama_macbeth: FamaMacBethResult | None
    avg_n_stocks: float
    n_months: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {
            "lookback": self.lookback,
            "skip": self.skip,
            "n_months": self.n_months,
            "avg_n_stocks": self.avg_n_stocks,
        }
        if self.ic_spearman is not None:
            out.update(self.ic_spearman.as_dict("ic_rank_"))
        if self.ic_pearson is not None:
            out.update({"ic_pearson_mean": self.ic_pearson.mean_ic,
                        "ic_pearson_t": self.ic_pearson.tstat})
        if self.fama_macbeth is not None:
            out.update(self.fama_macbeth.as_dict("fm_"))
        return out


@dataclass
class StrategyResult:
    """單一 (lookback, skip, horizon) 策略的完整結果。"""

    lookback: int
    skip: int
    horizon: int
    tag: str
    portfolios: QuantilePortfolios
    ls_gross: pd.Series
    ls_net: pd.Series
    moments_gross: Moments
    moments_net: Moments
    hac: HACResult
    alpha: RegressionResult | None
    grs: GRSResult | None
    mr: MRResult | None
    cost: CostReport
    power: PowerResult
    boot_mean: BootstrapCI
    boot_sharpe: BootstrapCI
    naive_overlap: HACResult | None
    nw_overlap: HACResult | None
    hh_overlap: HACResult | None
    profile: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def summary_row(self) -> dict:
        """攤平成一列，給 Excel 的 summary 工作表用。"""
        row = {
            "strategy": self.tag,
            "lookback": self.lookback,
            "skip": self.skip,
            "horizon": self.horizon,
            "n_months": self.hac.nobs,
            "avg_n_long": float(self.portfolios.counts.iloc[:, -1].mean()),
            "avg_n_short": float(self.portfolios.counts.iloc[:, 0].mean()),
            # --- 主結果：日曆時間非重疊多空 ---
            "ls_mean_m": self.moments_gross.mean_m,
            "ls_mean_ann": self.moments_gross.mean_ann,
            "ls_vol_ann": self.moments_gross.vol_ann,
            "ls_t_hac": self.hac.tstat,
            "ls_p_hac": self.hac.pvalue,
            "ls_t_iid": self.hac.iid_tstat,
            "ls_hac_lags": self.hac.lags,
            "ls_sharpe": self.moments_gross.sharpe,
            "ls_sharpe_t": self.moments_gross.sharpe_t,
            "ls_skew": self.moments_gross.skew,
            "ls_exkurt": self.moments_gross.excess_kurtosis,
            "ls_max_dd": self.moments_gross.max_drawdown,
            "ls_hit_rate": self.moments_gross.hit_rate,
            # --- 風險調整 ---
            "alpha_m": self.alpha.alpha if self.alpha else np.nan,
            "alpha_ann": ((1 + self.alpha.alpha) ** 12 - 1) if self.alpha else np.nan,
            "alpha_t": self.alpha.alpha_t if self.alpha else np.nan,
            "alpha_p": self.alpha.alpha_p if self.alpha else np.nan,
            "r2_factors": self.alpha.r2 if self.alpha else np.nan,
            # --- 單調性與聯合檢定 ---
            "mr_J": self.mr.statistic if self.mr else np.nan,
            "mr_p": self.mr.pvalue if self.mr else np.nan,
            "mr_n_pos_deltas": self.mr.n_positive if self.mr else np.nan,
            "mr_spearman": self.mr.spearman if self.mr else np.nan,
            "grs_F": self.grs.statistic if self.grs else np.nan,
            "grs_p": self.grs.pvalue if self.grs else np.nan,
            # --- 成本 ---
            "turnover_m": self.cost.avg_monthly_turnover_one_way,
            "net_mean_m": self.cost.net_mean_m,
            "net_sharpe": self.cost.net_sharpe,
            "breakeven_bps": self.cost.breakeven_one_way_bps,
            # --- bootstrap ---
            "boot_p_mean": self.boot_mean.pvalue,
            "boot_sharpe_lo": self.boot_sharpe.lower,
            "boot_sharpe_hi": self.boot_sharpe.upper,
            # --- 檢定力 ---
            "mde_m_80pct": self.power.mde_m,
            "power_at_ref_0.8pct": self.power.power_at_reference,
            "required_n_for_ref": self.power.required_n_for_reference,
            "underpowered": self.power.underpowered,
        }
        # --- 重疊報酬的對照（原始碼的做法 vs 修正後）---
        if self.naive_overlap is not None:
            row.update(
                {
                    "overlap_mean_h": self.naive_overlap.mean,
                    "overlap_t_naive": self.naive_overlap.iid_tstat,
                    "overlap_t_nw": self.nw_overlap.tstat if self.nw_overlap else np.nan,
                    "overlap_t_hh": self.hh_overlap.tstat if self.hh_overlap else np.nan,
                    "overlap_se_inflation_nw": self.nw_overlap.inflation if self.nw_overlap else np.nan,
                    "overlap_theory_inflation_sqrt_h": inflation_factor(self.horizon),
                }
            )
        return row


@dataclass
class StudyResult:
    """一次完整研究的結果。"""

    spec: StudySpec
    panel_info: dict
    factors: pd.DataFrame
    factor_stats: pd.DataFrame
    strategies: list[StrategyResult]
    signal_results: list[SignalLevelResult]
    summary: pd.DataFrame
    signal_summary: pd.DataFrame
    haircuts: pd.DataFrame
    spa: SPAResult | None
    dsr: DeflatedSharpeResult | None
    max_t_null: dict
    power_grid: pd.DataFrame
    ls_matrix: pd.DataFrame
    runtime_sec: float
    warnings_log: list[str] = field(default_factory=list)

    @property
    def best(self) -> StrategyResult | None:
        """樣本內 HAC t 值最高的策略。

        刻意叫 "best" 而不是 "winner"：它是**挑選後**的結果，
        它的 p 值不能直接解讀，必須看 haircut / DSR / SPA。
        """
        if not self.strategies:
            return None
        valid = [s for s in self.strategies if np.isfinite(s.hac.tstat)]
        if not valid:
            return None
        return max(valid, key=lambda s: s.hac.tstat)


def _build_strategy(
    panel: Panel,
    sig_panel: SignalPanel,
    quantiles: pd.DataFrame,
    factors: pd.DataFrame,
    spec: StudySpec,
    lookback: int,
    skip: int,
    horizon: int,
    warn_log: list[str],
) -> StrategyResult:
    """算出單一策略的所有統計量。"""
    tag = strategy_tag(lookback, skip, horizon)
    pspec = PortfolioSpec(n_quantiles=spec.n_quantiles, horizon=horizon)
    ports = build_calendar_time(quantiles, panel.mret, pspec)

    ls_gross = ports.long_short.dropna().rename(tag)
    turnover = ports.ls_turnover
    cost = analyze_costs(ls_gross, turnover, spec.costs)
    ls_net = net_returns(ls_gross, turnover, spec.costs)

    st = spec.stats
    hac = hac_mean_test(ls_gross, lags=st.hac_lags, kernel=st.hac_kernel)
    m_gross = describe(ls_gross)
    m_net = describe(ls_net)

    notes: list[str] = []

    # --- 風險調整 alpha ---
    alpha_res = None
    try:
        alpha_res = factor_regression(ls_gross, factors, lags=st.hac_lags, kernel=st.hac_kernel)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"alpha 迴歸失敗：{exc}")
        warn_log.append(f"[{tag}] alpha 迴歸失敗：{exc}")

    # --- GRS：十個分位的 alpha 聯合為零？ ---
    grs_res = None
    try:
        qrets = ports.returns.dropna(how="any")
        qrets.columns = [f"Q{i + 1}" for i in qrets.columns]
        grs_res = grs_test(qrets, factors.reindex(qrets.index).dropna())
    except Exception as exc:  # noqa: BLE001
        notes.append(f"GRS 失敗：{exc}")
        warn_log.append(f"[{tag}] GRS 失敗：{exc}")

    # --- 單調性 ---
    mr_res = None
    try:
        mr_res = mr_test(
            ports.returns, direction="up",
            reps=st.bootstrap_reps, block=st.bootstrap_block, seed=st.seed,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"MR 檢定失敗：{exc}")
        warn_log.append(f"[{tag}] MR 檢定失敗：{exc}")

    # --- bootstrap ---
    boot_m = bootstrap_mean(ls_gross, st.bootstrap_reps, st.bootstrap_block, st.alpha, st.seed)
    boot_s = bootstrap_sharpe(ls_gross, st.bootstrap_reps, st.bootstrap_block, st.alpha, 12, st.seed)

    # --- 檢定力 ---
    pw = power_diagnose(ls_gross, alpha=st.alpha, target_power=st.power_target)

    # --- 原始碼那套重疊報酬的對照 ---
    naive = nw_ov = hh_ov = None
    if horizon > 1:
        try:
            ov = overlapping_forward_returns(quantiles, panel.mret, horizon, spec.n_quantiles)
            ov_ls = (ov[spec.n_quantiles - 1] - ov[0]).dropna()
            if len(ov_ls) >= 10:
                naive = hac_mean_test(ov_ls, lags=0)
                nw_ov = newey_west_mean_test(ov_ls, horizon=horizon)
                hh_ov = hansen_hodrick_mean_test(ov_ls, horizon=horizon)
                if hh_ov.psd_failed:
                    notes.append("Hansen-Hodrick 截斷核算出非正的長期變異數（小樣本常見），該值回報為 nan")
        except Exception as exc:  # noqa: BLE001
            warn_log.append(f"[{tag}] 重疊報酬對照失敗：{exc}")
    else:
        # h=1 時日曆時間與重疊報酬是同一回事，天真 t 就是 iid t。
        naive = hac_mean_test(ls_gross, lags=0)
        nw_ov = hac
        hh_ov = None

    profile = quantile_profile(ports.returns)

    return StrategyResult(
        lookback=lookback, skip=skip, horizon=horizon, tag=tag,
        portfolios=ports, ls_gross=ls_gross, ls_net=ls_net,
        moments_gross=m_gross, moments_net=m_net, hac=hac,
        alpha=alpha_res, grs=grs_res, mr=mr_res, cost=cost, power=pw,
        boot_mean=boot_m, boot_sharpe=boot_s,
        naive_overlap=naive, nw_overlap=nw_ov, hh_overlap=hh_ov,
        profile=profile, notes=notes,
    )


def _build_signal_level(
    panel: Panel,
    sig_panel: SignalPanel,
    lookback: int,
    skip: int,
    spec: StudySpec,
    warn_log: list[str],
) -> SignalLevelResult:
    """IC 與 Fama-MacBeth：用全部股票而非只看兩端。"""
    fwd = panel.mret.shift(-1)
    sig = sig_panel.signal
    notes: list[str] = []

    ic_s = ic_p = fm = None
    try:
        ic_s = information_coefficient(sig, fwd, "spearman", spec.min_stocks)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Spearman IC 失敗：{exc}")
        warn_log.append(f"[mom{lookback}-{skip}] Spearman IC 失敗：{exc}")
    try:
        ic_p = information_coefficient(sig, fwd, "pearson", spec.min_stocks)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Pearson IC 失敗：{exc}")

    try:
        wins = winsorize_cross_section(sig, sig_panel.spec.winsorize)
        controls = {
            "log_adv": np.log(panel.adv.where(panel.adv > 0)),
            "strev": panel.mret,
            "rvol": realized_vol(panel),
        }
        fm = fama_macbeth(wins, fwd, controls, min_stocks=spec.min_stocks)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Fama-MacBeth 失敗：{exc}")
        warn_log.append(f"[mom{lookback}-{skip}] Fama-MacBeth 失敗：{exc}")

    n = sig_panel.n_stocks
    return SignalLevelResult(
        lookback=lookback, skip=skip,
        ic_spearman=ic_s, ic_pearson=ic_p, fama_macbeth=fm,
        avg_n_stocks=float(n[n > 0].mean()) if (n > 0).any() else float("nan"),
        n_months=int((n >= spec.min_stocks).sum()),
        notes=notes,
    )


def run_study(
    spec: StudySpec | None = None,
    panel: Panel | None = None,
    verbose: bool = True,
) -> StudyResult:
    """跑完整研究。"""
    spec = spec or StudySpec()
    t0 = time.time()
    warn_log: list[str] = []

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"[1/5] 載入面板 history={spec.data.history} ...")
    panel = panel or load_panel(spec.data)
    info = panel.describe()
    log(
        f"      {info['n_days']} 個交易日 / {info['n_months']} 個月 / "
        f"{info['n_tickers']} 檔，{info['date_start']} → {info['date_end']}"
    )

    log("[2/5] 建立風險因子基準 ...")
    factors = build_factors(panel, spec.liquidity_quantile, spec.min_stocks)
    fstats = factor_summary(factors)

    log(f"[3/5] 掃 {spec.n_trials} 個策略（{len(spec.signal_grid())} 組訊號 × {len(spec.horizons)} 個持有期）...")
    strategies: list[StrategyResult] = []
    signal_results: list[SignalLevelResult] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lookback, skip in spec.signal_grid():
            if skip >= lookback:
                # 形成期長度 = lookback - skip <= 0，這個組合在定義上不存在
                # （例如 lookback=1, skip=1）。這不是失敗，是網格的邊界；
                # 記錄下來讓多重檢定的分母說得清楚為什麼是 28 而不是 32。
                warn_log.append(
                    f"跳過 lookback={lookback}, skip={skip}：形成期長度為 "
                    f"{lookback - skip} 個月，該組合無定義（非錯誤）。"
                )
                continue
            sspec = spec.signal_spec(lookback, skip)
            try:
                sig_panel = build_signal_panel(panel, sspec)
            except Exception as exc:  # noqa: BLE001
                warn_log.append(f"訊號 mom{lookback}-{skip} 建立失敗：{exc}")
                continue

            quantiles = assign_quantiles(sig_panel.signal, spec.n_quantiles, spec.min_stocks)
            signal_results.append(
                _build_signal_level(panel, sig_panel, lookback, skip, spec, warn_log)
            )

            for horizon in spec.horizons:
                try:
                    res = _build_strategy(
                        panel, sig_panel, quantiles, factors, spec,
                        lookback, skip, horizon, warn_log,
                    )
                except Exception as exc:  # noqa: BLE001
                    warn_log.append(f"策略 {strategy_tag(lookback, skip, horizon)} 失敗：{exc}")
                    continue
                strategies.append(res)
                log(
                    f"      {res.tag:>16s}  n={res.hac.nobs:>3d}  "
                    f"mean={res.moments_gross.mean_m * 100:6.3f}%/m  "
                    f"t_iid={res.hac.iid_tstat:6.2f}  t_HAC={res.hac.tstat:6.2f}  "
                    f"alpha_t={(res.alpha.alpha_t if res.alpha else float('nan')):6.2f}  "
                    f"MR_p={(res.mr.pvalue if res.mr else float('nan')):.3f}"
                )

    if not strategies:
        raise RuntimeError("沒有任何策略成功完成，無法產生研究結果")

    log("[4/5] 研究層校正：多重檢定 / HLZ haircut / Deflated Sharpe / SPA ...")
    summary = pd.DataFrame([s.summary_row() for s in strategies])
    signal_summary = pd.DataFrame([s.as_dict() for s in signal_results])

    # 多重檢定：分母是實際掃過的策略數，不是「我後來覺得有道理的那幾個」。
    n_trials = len(strategies)
    haircuts = haircut_sharpe(
        summary["ls_t_hac"].to_numpy(),
        summary["ls_sharpe"].to_numpy(),
        hlz_threshold=spec.stats.hlz_tstat_threshold,
    )
    haircuts.insert(0, "strategy", summary["strategy"].to_numpy())

    ls_matrix = pd.concat([s.ls_gross.rename(s.tag) for s in strategies], axis=1)

    # 策略間平均相關係數 → 餵進 expected_max_abs_t，量化「32 個檢定其實
    # 不是 32 個獨立檢定」這件事。
    corr = ls_matrix.corr()
    off = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    rho_bar = float(np.nanmean(off)) if off.size else 0.0
    max_t_null = {
        "independent": expected_max_abs_t(n_trials, rho=0.0, seed=spec.stats.seed),
        "observed_avg_corr": expected_max_abs_t(
            n_trials, rho=max(rho_bar, 0.0), seed=spec.stats.seed
        ),
        "avg_pairwise_corr": rho_bar,
    }

    spa_res = None
    try:
        spa_res = spa_test(
            ls_matrix, reps=spec.stats.bootstrap_reps,
            block=spec.stats.bootstrap_block, seed=spec.stats.seed,
        )
    except Exception as exc:  # noqa: BLE001
        warn_log.append(f"SPA 檢定失敗：{exc}")

    dsr_res = None
    best = None
    valid = [s for s in strategies if np.isfinite(s.hac.tstat)]
    if valid:
        best = max(valid, key=lambda s: s.hac.tstat)
        try:
            dsr_res = deflated_sharpe_ratio(
                best.ls_gross, n_trials=n_trials,
                trial_sharpes=summary["ls_sharpe"].to_numpy(),
            )
        except Exception as exc:  # noqa: BLE001
            warn_log.append(f"Deflated Sharpe 失敗：{exc}")

    # 檢定力表：用所有策略月標準差的中位數當代表值。
    sigma_rep = float(np.nanmedian(summary["ls_vol_ann"] / np.sqrt(12.0)))
    grid = power_table(sigma_rep, alpha=spec.stats.alpha)

    log("[5/5] 完成。")
    return StudyResult(
        spec=spec,
        panel_info=info,
        factors=factors,
        factor_stats=fstats,
        strategies=strategies,
        signal_results=signal_results,
        summary=summary,
        signal_summary=signal_summary,
        haircuts=haircuts,
        spa=spa_res,
        dsr=dsr_res,
        max_t_null=max_t_null,
        power_grid=grid,
        ls_matrix=ls_matrix,
        runtime_sec=time.time() - t0,
        warnings_log=warn_log,
    )
