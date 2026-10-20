"""
factorlab.report — Excel 多工作表與 markdown 研究結論。

報告的設計原則：**讓讀者無法只看一個數字就下結論。**

原始輸出是一張單一工作表的 Excel，裡面是 Q10-Q1 的報酬百分比，加一個
CSV 寫著 t 值與 p 值。那種格式邀請讀者去挑最大的 t 值。本模組的作法相反：
每一個策略旁邊都並列「天真 t / HAC t / alpha t / 校正後 t」，讓膨脹與
折價在同一列上無法迴避；研究層的結論（SPA、DSR、E[max|t|]）放在
最前面的工作表，而不是附錄。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .stats.multiple_testing import EULER_GAMMA  # noqa: F401  (保持 API 可見)
from .study import StudyResult

PCT_COLS_HINT = ("mean", "alpha", "net", "mde", "overlap_mean", "turnover", "vol", "dd")


def _fmt_pct(x: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _fmt(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


# ---------------------------------------------------------------- Excel


def _autofit(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns):
        width = max(len(str(col)), int(df[col].astype(str).str.len().max() if len(df) else 0))
        ws.set_column(i, i, min(max(width + 2, 10), 42))
    ws.freeze_panes(1, 1)


def write_excel(result: StudyResult, path: str | Path) -> Path:
    """寫出多工作表 Excel。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pd.DataFrame] = {}

    # 01 研究層結論 —— 刻意放第一張。
    sheets["01_research_level"] = _research_level_frame(result)

    # 02 策略總表
    summary = result.summary.copy()
    sheets["02_strategy_summary"] = summary.sort_values("ls_t_hac", ascending=False)

    # 03 重疊報酬膨脹對照（原始碼的錯誤規模）
    sheets["03_overlap_inflation"] = _overlap_frame(result)

    # 04 多重檢定與 haircut
    sheets["04_multiple_testing"] = result.haircuts

    # 05 分位曲線
    prof = []
    for s in result.strategies:
        p = s.profile.copy()
        p.insert(0, "strategy", s.tag)
        prof.append(p)
    sheets["05_quantile_profiles"] = pd.concat(prof, ignore_index=True)

    # 06 訊號層（IC / Fama-MacBeth）
    sheets["06_signal_level"] = result.signal_summary

    # 07 風險因子基準
    sheets["07_factors"] = result.factor_stats

    # 08 檢定力
    sheets["08_power"] = result.power_grid

    # 09 策略相關矩陣
    corr = result.ls_matrix.corr().round(3)
    sheets["09_correlations"] = corr.reset_index().rename(columns={"index": "strategy"})

    # 10 月報酬序列（多空，毛）
    monthly = result.ls_matrix.copy()
    monthly.index.name = "month"
    sheets["10_monthly_ls_returns"] = monthly.reset_index()

    # 11 因子月報酬
    f = result.factors.copy()
    f.index.name = "month"
    sheets["11_factor_returns"] = f.reset_index()

    # 12 規格與警告
    sheets["12_spec_and_caveats"] = _spec_frame(result)

    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        for name, df in sheets.items():
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_bool_dtype(out[c]):
                    out[c] = out[c].astype(object)
            out.to_excel(writer, sheet_name=name[:31], index=False)
            _autofit(writer, name[:31], out)
    return path


def _research_level_frame(result: StudyResult) -> pd.DataFrame:
    """研究層結論表：回答「這份研究整體上發現了什麼」。"""
    rows: list[dict] = []

    def add(item: str, value, note: str = "") -> None:
        rows.append({"item": item, "value": value, "note": note})

    info = result.panel_info
    add("資料集", info["history"], "long = 2007–2024；3y = 近三年")
    add("樣本期間", f"{info['date_start']} → {info['date_end']}", "")
    add("月數 T", info["n_months"], "檢定力的主要決定因素")
    add("平均每月可選股票數", round(info["avg_valid_tickers_per_month"], 1), "")
    add("掃過的策略數 M", len(result.strategies), "多重檢定的分母")

    mt = result.max_t_null
    add(
        "全域虛無下 E[max|t|]（獨立）",
        round(mt["independent"]["mc_expected_max_abs_t"], 3),
        f"解析近似 sqrt(2 ln M) = {mt['independent']['asymptotic_sqrt_2logM']:.3f}",
    )
    add("策略間平均相關係數", round(mt["avg_pairwise_corr"], 3), "越高代表有效獨立檢定數越少")
    add(
        "全域虛無下 E[max|t|]（按實際相關）",
        round(mt["observed_avg_corr"]["mc_expected_max_abs_t"], 3),
        "這是本研究真正該比較的門檻，而不是 1.96",
    )
    add(
        "全域虛無下 max|t| 的 95% 分位",
        round(mt["observed_avg_corr"]["mc_q95_max_abs_t"], 3),
        "單一策略要超過這個值才算在 5% 水準上有發現",
    )
    add("HLZ 建議門檻", result.spec.stats.hlz_tstat_threshold, "Harvey-Liu-Zhu (2016) 對新因子的建議")

    best = result.best
    if best is not None:
        add("樣本內最佳策略", best.tag, "挑選後的結果，其 p 值不可直接解讀")
        add("最佳策略 t（iid，原始碼做法）", round(best.hac.iid_tstat, 3), "")
        add("最佳策略 t（HAC 修正）", round(best.hac.tstat, 3), "")
        add(
            "最佳策略 alpha t（四因子）",
            round(best.alpha.alpha_t, 3) if best.alpha else "n/a",
            "風險調整後才是 alpha",
        )
        add("最佳策略年化 Sharpe（毛）", round(best.moments_gross.sharpe, 3), "")
        add("最佳策略年化 Sharpe（淨）", round(best.cost.net_sharpe, 3), "扣交易成本")

    if result.dsr is not None:
        d = result.dsr
        add("Deflated Sharpe Ratio", round(d.dsr, 4) if np.isfinite(d.dsr) else "n/a",
            "慣例門檻 > 0.95")
        add("虛無下期望最大 Sharpe（年化）", round(d.expected_max_sharpe_ann, 3),
            "試了 M 次本來就會出現的最高 Sharpe")

    if result.spa is not None:
        s = result.spa
        add("Hansen SPA p（一致版，主結果）", round(s.p_consistent, 4), "H0：沒有任何策略優於零")
        add("Hansen SPA p（下界，最寬鬆）", round(s.p_lower, 4), "連這個都不顯著就沒有解讀空間")
        add("White Reality Check p（上界）", round(s.p_upper, 4), "最保守")

    sig = result.summary
    add("HAC t > 1.96 的策略數", int((sig["ls_t_hac"].abs() > 1.96).sum()), "未校正")
    add("HAC t > 3.0 的策略數", int((sig["ls_t_hac"].abs() > 3.0).sum()), "HLZ 門檻")
    add("Holm 校正後 5% 顯著的策略數", int(result.haircuts["holm_significant_5pct"].sum()), "")
    add("BY 校正後 5% 顯著的策略數", int(result.haircuts["by_significant_5pct"].sum()), "")
    add("MR 單調性檢定 p < 0.05 的策略數", int((sig["mr_p"] < 0.05).sum()), "整條分位曲線單調遞增")
    add("GRS 拒絕（p < 0.05）的策略數", int((sig["grs_p"] < 0.05).sum()),
        "拒絕代表四因子無法解釋十分位的 alpha")
    add("檢定力不足（對月 0.8% 效果）的策略數", int(sig["underpowered"].sum()), "")
    add("執行時間（秒）", round(result.runtime_sec, 1), "")
    return pd.DataFrame(rows)


def _overlap_frame(result: StudyResult) -> pd.DataFrame:
    """重疊報酬的 t 值膨脹對照表 —— 原始碼錯誤的量化。"""
    rows = []
    for s in result.strategies:
        if s.naive_overlap is None:
            continue
        rows.append(
            {
                "strategy": s.tag,
                "horizon": s.horizon,
                "nobs_overlapping": s.naive_overlap.nobs,
                "mean_h_return": s.naive_overlap.mean,
                "t_naive_ttest_1samp": s.naive_overlap.iid_tstat,
                "t_newey_west": s.nw_overlap.tstat if s.nw_overlap else np.nan,
                "t_hansen_hodrick": s.hh_overlap.tstat if s.hh_overlap else np.nan,
                "se_inflation_observed": s.nw_overlap.inflation if s.nw_overlap else np.nan,
                "se_inflation_theory_sqrt_h": np.sqrt(s.horizon),
                "nobs_calendar_time": s.hac.nobs,
                "t_calendar_time_hac": s.hac.tstat,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "strategy"])


def _spec_frame(result: StudyResult) -> pd.DataFrame:
    rows = [{"section": "spec", "key": "json", "value": result.spec.to_json()}]
    for i, w in enumerate(result.warnings_log):
        rows.append({"section": "warnings", "key": f"w{i + 1}", "value": w})
    for cav in CAVEATS:
        rows.append({"section": "caveats", "key": cav[0], "value": cav[1]})
    return pd.DataFrame(rows)


CAVEATS: list[tuple[str, str]] = [
    (
        "規模因子是代理",
        "價格 parquet 只有 OHLCV，沒有歷史市值；FMP_stocks.csv 的市值是當期值，"
        "用它會造成前視偏誤。因此 SIZE 因子用日均成交金額的對數當代理（與市值的"
        "橫斷面相關約 0.8~0.9）。這是本研究最大的規格妥協：如果動能的報酬主要是"
        "規模補償，這個代理可能無法完全吸收掉它。",
    ),
    (
        "下市與存續偏誤",
        "持有期內若某檔股票當月沒有報酬（下市、停牌、資料缺口），該月把它從投組"
        "剔除並重新歸一化，而不是當成 0% 或 -100%。剔除會讓真實下市（多半大跌）"
        "的損失被低估。universe 已限制在流動性前 50%，污染較小，但方向是對策略"
        "有利的，所以本研究的 alpha 估計應視為上界。",
    ),
    (
        "universe 的時點",
        "Russell 3000 的成分股清單是當期的，parquet 收錄的是「現在還在清單上」"
        "的公司的歷史價格。這會造成存續偏誤：2007 年就已消失的公司不在樣本裡。"
        "這個偏誤無法用本地資料修正，需要 point-in-time 的成分股歷史。",
    ),
    (
        "GRS 的常態假設",
        "GRS 假設殘差常態且同質變異，不含 HAC 修正。本研究的測試資產是日曆時間"
        "非重疊月報酬，自相關已在投組層面處理，但常態性通常不成立（見 Jarque-Bera）。"
        "因此 GRS 僅作輔助證據，主結論以 HAC t 值與 bootstrap 為準。",
    ),
    (
        "日報酬離群值過濾是不對稱的",
        "|r| > 100% 的日報酬被視為資料錯誤而剔除，但報酬下界是 -100%，"
        "所以這個門檻只擋得住反向分割（上漲方向）。10 拆 1 的正向分割會呈現 "
        "-90%，不會被擋掉。刻意不收緊負向門檻，因為那會同時丟掉真實的單日崩跌"
        "（生技試驗失敗、財報造假），而那正是動能策略報酬分布左尾的一部分。"
        "若中央 parquet 的 Close 未做分割調整，動能訊號會在分割月出現假的大跌，"
        "請看 Excel 的執行期警告與 `extreme_daily_returns` 診斷。",
    ),
    (
        "成本模型是靜態的",
        "單邊 6 bps（1 bps 手續費 + 5 bps 半價差）對大型股在近年是合理的，但"
        "2008–2009 的實際價差遠高於此，而十分位的兩端本來就偏向流動性較差的股票。"
        "淨報酬因此也偏樂觀。breakeven_bps 是比淨 Sharpe 更穩健的解讀方式。",
    ),
    (
        "參數網格仍是人選的",
        "多重檢定校正的分母是實際掃過的 32 個組合，但 lookback ∈ {1,3,6,12}、"
        "horizon ∈ {1,3,6,12} 本身是依文獻慣例選的 —— 真正的「研究者自由度」"
        "比 32 更大（流動性門檻、分位數、權重方式、winsorize 比例都可以調）。"
        "因此即使通過 Holm/BY，也只能說「在這個網格內沒有被多重檢定解釋掉」。",
    ),
]


# ---------------------------------------------------------------- Markdown


def _verdict(result: StudyResult) -> tuple[str, list[str]]:
    """根據實際數字產生研究結論。這裡刻意寫成規則，不讓人工挑字。"""
    reasons: list[str] = []
    sig = result.summary

    thresh = result.max_t_null["observed_avg_corr"]["mc_q95_max_abs_t"]
    n_over = int((sig["ls_t_hac"].abs() > thresh).sum())
    n_holm = int(result.haircuts["holm_significant_5pct"].sum())
    n_by = int(result.haircuts["by_significant_5pct"].sum())
    n_alpha = int((sig["alpha_p"] < 0.05).sum())
    n_mr = int((sig["mr_p"] < 0.05).sum())
    n_underpowered = int(sig["underpowered"].sum())
    spa_p = result.spa.p_consistent if result.spa else float("nan")
    dsr = result.dsr.dsr if result.dsr else float("nan")

    survived = (
        n_by > 0
        and n_alpha > 0
        and np.isfinite(spa_p)
        and spa_p < 0.05
    )

    if survived:
        verdict = "在校正後仍有證據"
        reasons.append(
            f"{n_by} 個策略通過 BY 校正、{n_alpha} 個策略的四因子 alpha 在 5% 水準顯著，"
            f"且 Hansen SPA 的一致 p 值為 {spa_p:.4f}（< 0.05），"
            "表示「最好的策略只是運氣」這個虛無被拒絕。"
        )
        if np.isfinite(dsr):
            reasons.append(
                f"Deflated Sharpe = {dsr:.3f}"
                + ("（> 0.95，通過）" if dsr > 0.95 else "（<= 0.95，未通過，仍需保留）")
            )
        if n_mr == 0:
            reasons.append(
                "但沒有任何策略通過 Patton-Timmermann 單調性檢定 —— 多空價差顯著"
                "卻不單調，意味著報酬集中在兩端，更像是極端分位的特質而非一個"
                "貫穿橫斷面的因子。"
            )
    else:
        verdict = "無法在校正後主張發現"
        if n_over == 0:
            reasons.append(
                f"沒有任何策略的 HAC t 值超過全域虛無下 max|t| 的 95% 分位"
                f"（{thresh:.2f}）—— 也就是說，觀察到的最大 t 值落在「就算動能"
                "完全不存在、掃這麼多組合也會看到」的範圍內。"
            )
        else:
            reasons.append(
                f"有 {n_over} 個策略的 HAC t 超過虛無分配的 95% 分位，但"
                f"Holm 校正後顯著的有 {n_holm} 個、BY 校正後 {n_by} 個，"
                f"四因子 alpha 顯著的有 {n_alpha} 個。"
            )
        if np.isfinite(spa_p):
            reasons.append(
                f"Hansen SPA 的一致 p 值為 {spa_p:.4f}"
                + ("，無法拒絕「沒有任何策略優於零」。" if spa_p >= 0.05 else "。")
            )
        if np.isfinite(dsr):
            reasons.append(f"Deflated Sharpe = {dsr:.3f}（門檻 0.95）。")

    if n_underpowered > 0:
        reasons.append(
            f"{n_underpowered}/{len(sig)} 個策略對月 0.8%（年化約 10%，文獻動能溢酬的"
            "量級）的真實效果檢定力低於 80%。**所以「不顯著」不可以被讀成「動能不存在」**，"
            "只能讀成「這份樣本在這個規格下分辨不出來」。"
        )
    return verdict, reasons


def write_markdown(result: StudyResult, path: str | Path) -> Path:
    """寫出 markdown 研究結論。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    info = result.panel_info
    sig = result.summary.sort_values("ls_t_hac", ascending=False)
    best = result.best
    verdict, reasons = _verdict(result)
    mt = result.max_t_null

    lines: list[str] = []
    a = lines.append

    a("# 多重回看期動能因子：檢驗統計量重做")
    a("")
    a(f"- 資料集：`history={info['history']}`，{info['date_start']} → {info['date_end']}")
    a(f"- 樣本：{info['n_months']} 個月、{info['n_days']} 個交易日、{info['n_tickers']} 檔股票")
    a(f"- 平均每月可選股票數：{info['avg_valid_tickers_per_month']:.0f}")
    a(f"- 掃過的策略組合：{len(result.strategies)} 個（lookback × skip × horizon）")
    a(f"- 產生時間：{pd.Timestamp.now():%Y-%m-%d %H:%M}，耗時 {result.runtime_sec:.0f} 秒")
    a("")
    a("---")
    a("")
    a("## 結論")
    a("")
    a(f"**{verdict}。**")
    a("")
    for r in reasons:
        a(f"- {r}")
    a("")
    a("---")
    a("")
    a("## 一、原始檢定錯在哪裡，錯多少")
    a("")
    a("原始 `us_factor_liqm.py` 的最後一段是 `scipy.stats.ttest_1samp(x, 0)`，")
    a("對每月取樣、持有 h 個月的重疊報酬做檢定。下表把膨脹量化出來：")
    a("")
    ov = _overlap_frame(result)
    if not ov.empty:
        show = ov[ov["horizon"] > 1].copy()
        if not show.empty:
            a("| 策略 | h | 重疊樣本 n | 天真 t | NW t | HH t | 實測 SE 膨脹 | 理論 √h |")
            a("|---|---|---|---|---|---|---|---|")
            for _, r in show.iterrows():
                a(
                    f"| `{r['strategy']}` | {int(r['horizon'])} | {int(r['nobs_overlapping'])} "
                    f"| {_fmt(r['t_naive_ttest_1samp'])} | {_fmt(r['t_newey_west'])} "
                    f"| {_fmt(r['t_hansen_hodrick'])} | {_fmt(r['se_inflation_observed'])} "
                    f"| {_fmt(r['se_inflation_theory_sqrt_h'])} |"
                )
            a("")
            med = show["se_inflation_observed"].median()
            a(
                f"實測的標準誤膨脹中位數是 {_fmt(med)}，與理論值 √h 同量級 —— "
                "證實重疊報酬的 t 值膨脹是抽樣設計造成的，不是資料特性。"
            )
            a("")
    a("本研究的主結果改用 Jegadeesh-Titman 日曆時間重疊投組：每月同時持有 h 個")
    a("分批投組（各 1/h 權重），組合月報酬因此是非重疊觀察值，平均數的變異數")
    a("回到 γ₀/T，普通 t 檢定成立（仍保留小落後期 HAC 處理殘餘自相關）。")
    a("")
    a("## 二、研究層的門檻：1.96 是錯的門檻")
    a("")
    a(f"- 掃過 M = {len(result.strategies)} 個策略")
    a(f"- 全域虛無、檢定獨立時 E[max\\|t\\|] = {mt['independent']['mc_expected_max_abs_t']:.2f}"
      f"（解析近似 √(2 ln M) = {mt['independent']['asymptotic_sqrt_2logM']:.2f}）")
    a(f"- 策略間實際平均相關係數 = {mt['avg_pairwise_corr']:.3f}")
    a(f"- 按實際相關校正後 E[max\\|t\\|] = {mt['observed_avg_corr']['mc_expected_max_abs_t']:.2f}，"
      f"95% 分位 = {mt['observed_avg_corr']['mc_q95_max_abs_t']:.2f}")
    a("")
    a("換句話說：**單一策略的 HAC t 值要超過"
      f" {mt['observed_avg_corr']['mc_q95_max_abs_t']:.2f} 才算在 5% 水準上有發現**，"
      "不是 1.96。Harvey-Liu-Zhu (2016) 對新因子的建議門檻是 3.0，與此一致。")
    a("")
    a("## 三、前十名策略（按 HAC t 排序）")
    a("")
    a("| 策略 | n | 月均 | t(iid) | t(HAC) | α(四因子) | t(α) | MR p | GRS p | 淨 Sharpe | 損益兩平 bps |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in sig.head(10).iterrows():
        a(
            f"| `{r['strategy']}` | {int(r['n_months'])} | {_fmt_pct(r['ls_mean_m'])} "
            f"| {_fmt(r['ls_t_iid'])} | {_fmt(r['ls_t_hac'])} | {_fmt_pct(r['alpha_m'])} "
            f"| {_fmt(r['alpha_t'])} | {_fmt(r['mr_p'], 3)} | {_fmt(r['grs_p'], 3)} "
            f"| {_fmt(r['net_sharpe'])} | {_fmt(r['breakeven_bps'], 1)} |"
        )
    a("")
    if best is not None:
        a(f"樣本內最佳是 `{best.tag}`。它是**挑選後**的結果，所以下面三個校正才是它的真實成績：")
        a("")
        hc = result.haircuts.set_index("strategy").loc[best.tag]
        a(f"- Holm 校正後 p = {hc['holm_p_adj']:.4f}，折價後 Sharpe = {_fmt(hc['holm_sharpe_haircut'])}"
          f"（折減 {_fmt(hc['holm_haircut_pct'], 1)}%）")
        a(f"- BY 校正後 p = {hc['by_p_adj']:.4f}，折價後 Sharpe = {_fmt(hc['by_sharpe_haircut'])}")
        if result.dsr is not None:
            a(f"- Deflated Sharpe = {_fmt(result.dsr.dsr, 4)}（虛無下期望最大年化 Sharpe = "
              f"{_fmt(result.dsr.expected_max_sharpe_ann)}）")
        if result.spa is not None:
            a(f"- Hansen SPA：p_lower = {result.spa.p_lower:.4f}、"
              f"p_consistent = {result.spa.p_consistent:.4f}、p_upper = {result.spa.p_upper:.4f}")
        a("")

    a("## 四、單調性：兩端顯著不等於有因子")
    a("")
    n_mr = int((sig["mr_p"] < 0.05).sum())
    n_allpos = int((sig["mr_n_pos_deltas"] == result.spec.n_quantiles - 1).sum())
    a(f"- {n_mr}/{len(sig)} 個策略在 5% 水準通過 Patton-Timmermann MR 檢定")
    a(f"- {n_allpos}/{len(sig)} 個策略的 9 個相鄰分位差全為正")
    a("")
    a("MR 檢定的統計量是 `min_i Δ_i`，只有每一個相鄰分位差都為正時才大。")
    a("它比 Q10−Q1 嚴格得多：Q10−Q1 可以只靠 Q1 被幾檔崩跌股拉下去就顯著，")
    a("中間八個分位完全可以是平的。通過 MR 才能說「訊號在整個橫斷面都有排序能力」。")
    a("")
    if best is not None:
        a(f"`{best.tag}` 的分位曲線：")
        a("")
        a("| 分位 | 月均 | 年化 | t(HAC) | Sharpe |")
        a("|---|---|---|---|---|")
        for _, r in best.profile.iterrows():
            a(f"| Q{int(r['quantile']) + 1} | {_fmt_pct(r['mean_m'])} | {_fmt_pct(r['mean_ann'])} "
              f"| {_fmt(r['t_hac'])} | {_fmt(r['sharpe'])} |")
        a("")

    a("## 五、訊號層：IC 與 Fama-MacBeth")
    a("")
    a("分位投組只用訊號分布的兩端，中間 80% 的股票對結論毫無貢獻。")
    a("IC 與 Fama-MacBeth 用全部股票，統計效率高得多。")
    a("")
    ss = result.signal_summary
    if not ss.empty and "ic_rank_mean" in ss.columns:
        a("| lookback | skip | 月數 | 平均檔數 | 等級 IC | IC IR | IC t(HAC) | FM γ(訊號) | FM t |")
        a("|---|---|---|---|---|---|---|---|---|")
        for _, r in ss.iterrows():
            a(
                f"| {int(r['lookback'])} | {int(r['skip'])} | {int(r['n_months'])} "
                f"| {r['avg_n_stocks']:.0f} | {_fmt(r['ic_rank_mean'], 4)} "
                f"| {_fmt(r.get('ic_rank_ir', np.nan), 3)} | {_fmt(r.get('ic_rank_t', np.nan))} "
                f"| {_fmt(r.get('fm_signal_gamma', np.nan), 4)} "
                f"| {_fmt(r.get('fm_signal_t', np.nan))} |"
            )
        a("")
        a("Fama-MacBeth 的 γ 是在控制了規模代理（log 日均成交金額）、短期反轉")
        a("與 60 日已實現波動之後，訊號每高一個橫斷面標準差對下月報酬的淨影響。")
        a("")

    a("## 六、交易成本")
    a("")
    a("| 策略 | 月週轉（單邊） | 毛 Sharpe | 淨 Sharpe | 損益兩平單邊 bps |")
    a("|---|---|---|---|---|")
    for _, r in sig.head(8).iterrows():
        a(f"| `{r['strategy']}` | {_fmt_pct(r['turnover_m'], 1)} | {_fmt(r['ls_sharpe'])} "
          f"| {_fmt(r['net_sharpe'])} | {_fmt(r['breakeven_bps'], 1)} |")
    a("")
    a("持有期越長，每月只有 1/h 的部位需要換手，成本線性下降 —— 這是")
    a("「h 大看起來比較好」的一個純機械原因，與動能的持續性無關。")
    a("損益兩平 bps 把結論變成一個可驗證的交易問題：你的實際單邊成本有沒有低於這個數？")
    a("")

    a("## 七、檢定力：為什麼 3 年樣本不該用")
    a("")
    sigma_rep = float(np.nanmedian(sig["ls_vol_ann"] / np.sqrt(12.0)))
    a(f"以多空月標準差中位數 σ = {_fmt_pct(sigma_rep)} 計算（α = {result.spec.stats.alpha}）：")
    a("")
    pg = result.power_grid
    eff_cols = [c for c in pg.columns if c.startswith("power_at_")]
    a("| 樣本月數 | 年 | 80% 檢定力下的 MDE | " + " | ".join(
        c.replace("power_at_", "檢定力@").replace("pct_m", "%/月") for c in eff_cols) + " |")
    a("|---|---|---|" + "---|" * len(eff_cols))
    for _, r in pg.iterrows():
        a(
            f"| {int(r['nobs_months'])} | {r['years']} | {_fmt_pct(r['mde_m_at_80pct'])} | "
            + " | ".join(_fmt(r[c], 3) for c in eff_cols)
            + " |"
        )
    a("")
    a("讀法：文獻的動能溢酬大約是月 0.7~1.0%。看 `檢定力@0.8%/月` 那一欄 ——")
    a("34 個月（3 年）的檢定力遠低於 0.80，意思是**即使動能真的存在且幅度與文獻一致，")
    a("3 年樣本也有很大機率抓不到**。原始輸出的 `p = 0.2654` 因此不是「沒有動能」")
    a("的證據，而是「這份樣本沒有能力回答這個問題」的證據。")
    a("")

    a("## 八、實作層面修掉的兩個 bug")
    a("")
    a("1. **skip 放錯位置。** 原始碼的 `skip_holding_month` 把再平衡日往後推一個月，")
    a("   那是遲延進場，不是 skip。Jegadeesh-Titman 的 12-2 是跳過**形成期最後一個月**：")
    a("   訊號 = P[t-1]/P[t-12] − 1，持有仍從 t 開始，目的是避開一個月短期反轉與")
    a("   bid-ask bounce。修正後 skip 一律作用在形成期尾端。")
    a("2. **流動性篩選的時點。** 原始碼用 `form_start - 1M`，lookback=12 時那是 13 個月前")
    a("   的流動性，既非下單當下可觀察，也讓不同 lookback 面對不同 universe，")
    a("   使跨 lookback 比較失去意義。修正後一律用**再平衡當月**的日均成交金額。")
    a("")
    a("另外，分位分組從 `pd.qcut(..., duplicates='drop')` 改成 `rank(pct=True)`：")
    a("`duplicates='drop'` 在同值過多時會**悄悄回傳少於 10 組**，那些月份的")
    a("「Q10−Q1」其實不是十分位的兩端。")
    a("")

    a("## 九、保留意見（caveats）")
    a("")
    for title, body in CAVEATS:
        a(f"- **{title}**：{body}")
    a("")

    if result.warnings_log:
        a("## 十、執行期警告")
        a("")
        for w in result.warnings_log:
            a(f"- {w}")
        a("")

    a("---")
    a("")
    a("## 重現方式")
    a("")
    a("```bash")
    a("python -m factorlab.cli run --history long")
    a("```")
    a("")
    a("完整規格（JSON）見 Excel 的 `12_spec_and_caveats` 工作表。")
    a("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all(result: StudyResult, outdir: str | Path = ".", stem: str = "momentum_study") -> dict:
    """一次寫出 Excel 與 markdown，回傳檔案路徑。"""
    outdir = Path(outdir)
    xlsx = write_excel(result, outdir / f"{stem}.xlsx")
    md = write_markdown(result, outdir / f"{stem}.md")
    csv = outdir / f"{stem}_summary.csv"
    result.summary.to_csv(csv, index=False)
    return {"excel": xlsx, "markdown": md, "summary_csv": csv}
