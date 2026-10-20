"""
factorlab.cli — 命令列介面。

    python -m factorlab.cli run --history long
    python -m factorlab.cli run --history 3y --out out_3y   # 示範檢定力不足
    python -m factorlab.cli power --sigma 0.0558 --nobs 23  # 重算原始輸出的檢定力
    python -m factorlab.cli info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataSpec, StatsSpec, CostSpec, StudySpec
from .data import load_panel
from .report import write_all
from .stats.power import diagnose, power_table
from .study import run_study


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(text).replace(" ", "").split(",") if x != "")


def _cmd_run(args: argparse.Namespace) -> int:
    spec = StudySpec(
        data=DataSpec(
            history=args.history,
            start=args.start,
            end=args.end,
            usd_only=not args.allow_non_us,
        ),
        stats=StatsSpec(
            bootstrap_reps=args.reps,
            seed=args.seed,
            alpha=args.alpha,
        ),
        costs=CostSpec(
            half_spread_bps=args.half_spread_bps,
            commission_bps=args.commission_bps,
            enabled=not args.no_costs,
        ),
        lookbacks=_parse_ints(args.lookbacks),
        skips=_parse_ints(args.skips),
        horizons=_parse_ints(args.horizons),
        n_quantiles=args.quantiles,
        liquidity_quantile=args.liquidity_quantile,
        min_stocks=args.min_stocks,
    )

    result = run_study(spec, verbose=not args.quiet)
    paths = write_all(result, args.out, args.stem)

    print()
    print("=" * 78)
    print("輸出檔案")
    for k, v in paths.items():
        print(f"  {k:12s} {v}")
    print("=" * 78)

    sig = result.summary
    thresh = result.max_t_null["observed_avg_corr"]["mc_q95_max_abs_t"]
    print(f"掃過 {len(result.strategies)} 個策略；"
          f"虛無下 max|t| 的 95% 分位 = {thresh:.2f}（不是 1.96）")
    print(f"HAC t 超過該門檻的策略數：{int((sig['ls_t_hac'].abs() > thresh).sum())}")
    print(f"BY 校正後 5% 顯著：{int(result.haircuts['by_significant_5pct'].sum())}")
    if result.spa is not None:
        print(f"Hansen SPA p（一致版）：{result.spa.p_consistent:.4f}")
    if result.dsr is not None:
        print(f"Deflated Sharpe：{result.dsr.dsr:.4f}（門檻 0.95）")
    best = result.best
    if best is not None:
        print(f"樣本內最佳：{best.tag}  t_HAC={best.hac.tstat:.2f}  "
              f"（同一序列的 t_iid={best.hac.iid_tstat:.2f}）")
    print(f"檢定力不足的策略：{int(sig['underpowered'].sum())}/{len(sig)}")
    print()
    print("結論摘要請看 markdown 的「結論」一節。")
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    res = diagnose(
        sigma=args.sigma,
        nobs=args.nobs,
        observed_mean=args.observed_mean,
        alpha=args.alpha,
        reference_mean_m=args.reference,
    )
    print(f"樣本月數 n = {res.nobs}，月標準差 σ = {res.sigma_m * 100:.3f}%，α = {res.alpha}")
    print(f"80% 檢定力下的最小可偵測效果：{res.mde_m * 100:.3f}%/月（年化 {res.mde_ann * 100:.1f}%）")
    print(f"等價的年化 Sharpe 門檻：{res.mde_sharpe_ann:.2f}")
    if res.observed_mean_m is not None:
        print(f"觀察到的月均 {res.observed_mean_m * 100:.3f}% 之檢定力：{res.power_at_observed:.3f}")
    print(f"對文獻參考值 {res.reference_mean_m * 100:.1f}%/月的檢定力：{res.power_at_reference:.3f}")
    print(f"要在 80% 檢定力下偵測 {res.reference_mean_m * 100:.1f}%/月，需要 "
          f"{res.required_n_for_reference:.0f} 個月（約 {res.required_n_for_reference / 12:.0f} 年）")
    print()
    print(power_table(args.sigma, alpha=args.alpha).to_string(index=False))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    for history in ("3y", "long"):
        try:
            panel = load_panel(DataSpec(history=history))
        except Exception as exc:  # noqa: BLE001
            print(f"history={history}: 無法載入（{exc}）")
            continue
        info = panel.describe()
        print(f"history={history}: {info['n_months']} 個月 / {info['n_days']} 天 / "
              f"{info['n_tickers']} 檔，{info['date_start']} → {info['date_end']}，"
              f"平均每月可選 {info['avg_valid_tickers_per_month']:.0f} 檔")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="factorlab",
        description="多重回看期動能因子的研究級檢驗（修正重疊報酬、風險調整、多重檢定、單調性、檢定力）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="跑完整研究並輸出 Excel + markdown")
    r.add_argument("--history", default="long", choices=("3y", "long"),
                   help="價格資料集。預設 long（2007–2024，215 個月）；3y 樣本沒有檢定力")
    r.add_argument("--start", default=None, help="樣本起始日 YYYY-MM-DD")
    r.add_argument("--end", default=None, help="樣本結束日 YYYY-MM-DD")
    r.add_argument("--lookbacks", default="1,3,6,12", help="形成期月數，逗號分隔")
    r.add_argument("--skips", default="0,1", help="形成期尾端跳過的月數，逗號分隔")
    r.add_argument("--horizons", default="1,3,6,12", help="持有月數，逗號分隔")
    r.add_argument("--quantiles", type=int, default=10, help="分位數（預設 10）")
    r.add_argument("--liquidity-quantile", type=float, default=0.5,
                   dest="liquidity_quantile", help="流動性橫斷面分位門檻（預設 0.5）")
    r.add_argument("--min-stocks", type=int, default=50, dest="min_stocks")
    r.add_argument("--reps", type=int, default=2000, help="bootstrap 重抽次數")
    r.add_argument("--seed", type=int, default=20261020)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--half-spread-bps", type=float, default=5.0, dest="half_spread_bps")
    r.add_argument("--commission-bps", type=float, default=1.0, dest="commission_bps")
    r.add_argument("--no-costs", action="store_true", help="關閉交易成本")
    r.add_argument("--allow-non-us", action="store_true",
                   help="允許非美國公司（預設排除，避免 ADR 幣別問題）")
    r.add_argument("--out", default="output", help="輸出目錄")
    r.add_argument("--stem", default="momentum_study", help="輸出檔名主幹")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)

    w = sub.add_parser("power", help="檢定力與最小可偵測效果")
    w.add_argument("--sigma", type=float, required=True, help="月報酬標準差（小數，例 0.0558）")
    w.add_argument("--nobs", type=int, required=True, help="樣本月數")
    w.add_argument("--observed-mean", type=float, default=None, dest="observed_mean",
                   help="觀察到的月均報酬（小數）")
    w.add_argument("--alpha", type=float, default=0.05)
    w.add_argument("--reference", type=float, default=0.008,
                   help="文獻參考效果，月報酬小數（預設 0.008 = 年化約 10%%）")
    w.set_defaults(func=_cmd_power)

    i = sub.add_parser("info", help="列出可用資料集的涵蓋範圍")
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
