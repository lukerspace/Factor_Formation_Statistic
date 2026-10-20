"""
端到端整合測試。

用合成面板跑完整流程，驗證兩件事：
1. 注入真實動能效果時，流程抓得到（檢定力的 sanity check）。
2. 面板純粹是隨機漫步時，研究層校正把它擋下來（型一錯誤的 sanity check）。

第 2 點比第 1 點重要 —— 一個會在隨機資料上宣稱發現的框架毫無用處。
"""

import numpy as np
import pandas as pd
import pytest

from factorlab.config import DataSpec, StatsSpec, StudySpec
from factorlab.data import Panel
from factorlab.report import write_all
from factorlab.study import run_study


def _synthetic_panel(n_months=180, n_stocks=120, momentum_strength=0.0, seed=0) -> Panel:
    """造一個日頻面板，可選擇注入「上期動能預測下期報酬」的效果。

    注入方式必須在**日頻、逐月迭代**進行：對每個日曆月 m，先用截至
    m-1 月底的價格算出過去 12 個月報酬的橫斷面 z 分數，再把
    momentum_strength × z 平均分攤到 m 月的每一個交易日上。

    早期版本的做法是先造月頻報酬、再用 `np.repeat(mret/21, 21)` 攤成
    日頻 —— 那是錯的：`bdate_range` 的日曆月有 20~23 個交易日，21 天
    的區塊與月界線會逐漸漂移，於是相鄰兩個日曆月的報酬共用同一個
    區塊，製造出真實的正自相關。那個「純噪音」面板實際上含有可觀的
    動能，跑出年化 Sharpe 1.69，把型一錯誤測試整個推翻。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2008-01-01", periods=n_months * 21)
    cols = [f"T{i}" for i in range(n_stocks)]

    periods = days.to_period("M")
    month_keys = list(dict.fromkeys(periods))
    day_pos = {k: np.where(periods == k)[0] for k in month_keys}
    # 每月最後一個交易日的位置，用來取「月底價格」。
    month_end_pos = {k: day_pos[k][-1] for k in month_keys}

    monthly_sd = 0.09
    logret = np.zeros((len(days), n_stocks))
    cum = np.zeros((len(days), n_stocks))   # 累積對數報酬

    for j, key in enumerate(month_keys):
        pos = day_pos[key]
        n_d = len(pos)
        noise = rng.standard_normal((n_d, n_stocks)) * (monthly_sd / np.sqrt(n_d))

        drift = 0.0
        if momentum_strength and j >= 13:
            end_prev = month_end_pos[month_keys[j - 1]]
            end_base = month_end_pos[month_keys[j - 13]]
            past = cum[end_prev] - cum[end_base]      # 過去 12 個月對數報酬
            sd = past.std(ddof=1)
            z = (past - past.mean()) / sd if sd > 1e-12 else np.zeros_like(past)
            drift = momentum_strength * z / n_d

        logret[pos] = noise + drift
        base = cum[pos[0] - 1] if pos[0] > 0 else np.zeros(n_stocks)
        cum[pos] = base + np.cumsum(logret[pos], axis=0)

    close = pd.DataFrame(100.0 * np.exp(cum), index=days, columns=cols)
    dret = pd.DataFrame(np.expm1(logret), index=days, columns=cols)
    volume = pd.DataFrame(
        rng.lognormal(12, 0.8, (len(days), n_stocks)), index=days, columns=cols
    )
    dollar_vol = close * volume

    grouper = close.index.to_period("M")
    month_ends = pd.DatetimeIndex(
        pd.Series(close.index, index=grouper).groupby(level=0).max().values
    ).sort_values()
    valid = pd.DataFrame(True, index=month_ends, columns=cols)
    mclose = close.groupby(grouper).last()
    mclose.index = month_ends
    adv = dollar_vol.groupby(grouper).mean()
    adv.index = month_ends
    mret_df = mclose / mclose.shift(1) - 1.0

    return Panel(
        close=close, volume=volume, dret=dret, mclose=mclose, mret=mret_df,
        dollar_vol=dollar_vol, adv=adv, valid=valid, spec=DataSpec(),
    )


def _small_spec() -> StudySpec:
    return StudySpec(
        lookbacks=(6, 12),
        skips=(0, 1),
        horizons=(1, 3),
        min_stocks=30,
        stats=StatsSpec(bootstrap_reps=300, seed=7),
    )


@pytest.fixture(scope="module")
def noise_result():
    panel = _synthetic_panel(momentum_strength=0.0, seed=11)
    return run_study(_small_spec(), panel=panel, verbose=False)


@pytest.fixture(scope="module")
def signal_result():
    panel = _synthetic_panel(momentum_strength=0.030, seed=12)
    return run_study(_small_spec(), panel=panel, verbose=False)


# ------------------------------------------------- 結構


def test_study_runs_all_grid_combinations(noise_result):
    assert len(noise_result.strategies) == 8
    assert len(noise_result.signal_results) == 4
    assert len(noise_result.summary) == 8


def test_summary_has_all_inference_layers(noise_result):
    cols = set(noise_result.summary.columns)
    for required in (
        "ls_t_hac", "ls_t_iid",          # HAC vs 天真
        "alpha_t", "alpha_p", "r2_factors",  # 風險調整
        "mr_p", "grs_p",                  # 單調性 / 聯合檢定
        "net_sharpe", "breakeven_bps",    # 成本
        "mde_m_80pct", "underpowered",    # 檢定力
        "boot_p_mean",                    # bootstrap
    ):
        assert required in cols, required


def test_research_level_objects_exist(noise_result):
    assert noise_result.spa is not None
    assert noise_result.dsr is not None
    assert not noise_result.haircuts.empty
    assert "avg_pairwise_corr" in noise_result.max_t_null
    assert not noise_result.power_grid.empty
    assert noise_result.factors.shape[1] == 4


def test_hac_differs_from_iid_for_multi_month_horizons(noise_result):
    """h > 1 的策略，HAC 與天真 t 必須不同，否則修正沒有生效。"""
    multi = noise_result.summary[noise_result.summary["horizon"] > 1]
    assert len(multi) > 0
    assert not np.allclose(multi["ls_t_hac"], multi["ls_t_iid"])


def test_overlap_inflation_is_recorded(noise_result):
    """重疊報酬的對照欄位必須有值，且實測膨脹大於 1。"""
    multi = noise_result.summary[noise_result.summary["horizon"] > 1]
    infl = multi["overlap_se_inflation_nw"].dropna()
    assert len(infl) > 0
    assert (infl > 1.0).all()


def test_turnover_decreases_with_horizon(noise_result):
    s = noise_result.summary
    t1 = s.loc[s["horizon"] == 1, "turnover_m"].mean()
    t3 = s.loc[s["horizon"] == 3, "turnover_m"].mean()
    assert t3 < t1


def test_no_warnings_logged_on_clean_synthetic_data(noise_result):
    assert noise_result.warnings_log == [], noise_result.warnings_log


# ------------------------------------------------- 型一錯誤（最重要）


def test_pure_noise_is_not_declared_significant(noise_result):
    """隨機漫步面板上，研究層校正必須擋下來。"""
    assert noise_result.spa.p_consistent > 0.05, "SPA 在純噪音上不該拒絕虛無"
    assert int(noise_result.haircuts["by_significant_5pct"].sum()) == 0
    assert int(noise_result.haircuts["holm_significant_5pct"].sum()) == 0


def test_pure_noise_verdict_is_negative(noise_result):
    from factorlab.report import _verdict

    verdict, reasons = _verdict(noise_result)
    assert verdict == "無法在校正後主張發現"
    assert len(reasons) > 0


def test_pure_noise_deflated_sharpe_below_threshold(noise_result):
    assert noise_result.dsr.dsr < 0.95


# ------------------------------------------------- 檢定力


def test_injected_momentum_is_detected(signal_result):
    """注入真實動能時，至少要有策略在 HAC 檢定下顯著。"""
    s = signal_result.summary
    assert (s["ls_t_hac"] > 2.0).any()
    assert (s["ls_mean_m"] > 0).all()


def test_injected_momentum_survives_multiple_testing(signal_result):
    assert int(signal_result.haircuts["by_significant_5pct"].sum()) > 0
    assert signal_result.spa.p_consistent < 0.05


def test_injected_momentum_is_monotone(signal_result):
    """DGP 的效果對動能 z 分數是線性的，分位曲線應該單調。"""
    s = signal_result.summary
    assert (s["mr_spearman"] > 0.8).all()


def test_signal_level_ic_is_positive_when_momentum_injected(signal_result):
    ss = signal_result.signal_summary
    assert (ss["ic_rank_mean"] > 0).all()
    assert (ss["ic_rank_t"] > 2).any()


def test_longer_lookback_matches_injected_dgp(signal_result):
    """注入的效果用 12 個月累積報酬定義，12M 訊號的 IC 應高於 6M。"""
    ss = signal_result.signal_summary.set_index(["lookback", "skip"])
    assert ss.loc[(12, 0), "ic_rank_mean"] > ss.loc[(6, 0), "ic_rank_mean"]


def test_best_is_selected_by_hac_tstat(signal_result):
    best = signal_result.best
    assert best is not None
    assert best.hac.tstat == pytest.approx(signal_result.summary["ls_t_hac"].max())


# ------------------------------------------------- 報告輸出


def test_write_all_produces_readable_files(noise_result, tmp_path):
    paths = write_all(noise_result, tmp_path, stem="test_study")
    assert paths["excel"].exists() and paths["excel"].stat().st_size > 5000
    assert paths["markdown"].exists()
    assert paths["summary_csv"].exists()

    md = paths["markdown"].read_text(encoding="utf-8")
    for heading in ("## 結論", "## 一、原始檢定錯在哪裡", "## 七、檢定力",
                    "## 九、保留意見"):
        assert heading in md, heading

    sheets = pd.read_excel(paths["excel"], sheet_name=None)
    for name in ("01_research_level", "02_strategy_summary", "03_overlap_inflation",
                 "04_multiple_testing", "08_power", "12_spec_and_caveats"):
        assert name in sheets, name
    assert not sheets["01_research_level"].empty


def test_markdown_states_power_limitation(noise_result, tmp_path):
    """報告必須明確寫出「不顯著 ≠ 沒效果」，這是原始研究最大的誤讀。"""
    paths = write_all(noise_result, tmp_path, stem="t2")
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "測不出來" in md or "不可以被讀成" in md


def test_spec_json_is_embedded_in_excel(noise_result, tmp_path):
    paths = write_all(noise_result, tmp_path, stem="t3")
    spec_sheet = pd.read_excel(paths["excel"], sheet_name="12_spec_and_caveats")
    assert (spec_sheet["section"] == "caveats").any()
    assert (spec_sheet["section"] == "spec").any()
