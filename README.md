# Multi-Lookback Momentum Factor — Research-Grade Statistical Inference

多重回看期動能因子（1M / 3M / 6M / 12M）的研究級檢驗框架。

重點**不是**跑出一個動能策略，而是正確地檢定它到底有沒有效。原始版本是一支 187 行的
`us_factor_liqm.py`，投組建構大致合理，但最後一行 `stats.ttest_1samp(x, 0)` 讓整份研究的
結論無法成立。`factorlab` 是為了修掉那一行背後的五個問題而生的。

## 原始檢定的五個問題

| # | 問題 | 為什麼致命 | 解法 |
|---|---|---|---|
| 1 | 重疊報酬做 `ttest_1samp` | 每月取樣卻持有 h 個月，自相關是抽樣設計造成的、不是資料特性；i.i.d. 假設讓 t 值高估約 √h 倍 | `portfolios`（JT 日曆時間重疊投組）+ `stats.hac` |
| 2 | 沒有風險調整 | Q10−Q1 的原始平均數不是 alpha；動能高分位系統性偏向高 beta、小型、高波動的股票 | `benchmarks` 四因子 + `stats.regression` |
| 3 | 沒有多重檢定校正 | 4 lookback × 2 skip × 4 horizon = 32 個檢定，全域虛無下 5% 門檻約 3.0；用 1.96 等於保證自己會「發現」東西 | `stats.multiple_testing` + `stats.spa` |
| 4 | 只看兩端、沒檢驗單調性 | Q10−Q1 顯著不代表訊號有排序能力，中間八個分位可以完全是平的 | `stats.monotonicity` + `stats.regression.grs_test` |
| 5 | 樣本只有 23–34 個月 | `p = 0.2654` 不代表「沒有動能」，代表「測不出來」 | `stats.power` |

另外修掉三個實作問題：`skip` 放在持有期而非形成期尾端、流動性篩選用了 13 個月前的資料、
分位分組的 `qcut(duplicates='drop')` 在同值過多時會悄悄回傳少於 10 組，使那些月份的
「Q10−Q1」其實不是十分位的兩端。詳見 [`RESULTS.md`](RESULTS.md) 第八節。

## 結論

**無法在校正後主張發現**（2007–2024，掃過 28 個策略）：最佳策略 HAC t = 1.08，低於全域
虛無下 max|t| 的 95% 分位 2.77；Hansen SPA 一致 p 值 = 0.704；Deflated Sharpe = 0.631
（門檻 0.95）；Holm / BY 校正後 5% 顯著的策略數 = 0。

但 22/28 個策略對月 0.8% 的真實效果檢定力低於 80% —— **「沒顯著」不等於「沒有動能」**，
這份樣本在這個規格下分辨不出文獻量級的動能溢酬。

完整統計量、前十名策略、重疊膨脹實測見 [`RESULTS.md`](RESULTS.md)。

## 安裝與執行

價格資料一律走中央 loader，本套件不含任何 parquet 路徑字串：

```python
from aws_fetch_price_data.fetch_fmp_price.price import load_us_price_data
```

所以需要讓 Python 找到 KICKs 專案根目錄：

```bash
export PYTHONPATH=/path/to/KICKs          # 或
export FACTORLAB_KICKS_ROOT=/path/to/KICKs
```

```bash
python -m factorlab.cli run --history long --out output    # 完整研究（預設 2007–2024）
python -m factorlab.cli run --history 3y --out output_3y   # 示範 3 年樣本為什麼不該用
python -m factorlab.cli run --lookbacks 6,12 --skips 1 --horizons 1,6
python -m factorlab.cli power --sigma 0.05576 --nobs 23    # 重算原始輸出的檢定力
python -m factorlab.cli info                               # 列出資料集涵蓋範圍
```

程式化使用：

```python
from factorlab import StudySpec, run_study, write_all

result = run_study(StudySpec())
write_all(result, "output")
print(result.best.tag, result.best.hac.tstat)
```

## 輸出

- `momentum_study.md` — 研究結論。判定是**規則產生**的，不是人工挑字
- `momentum_study.xlsx` — 12 個工作表，刻意把**研究層結論放第一張**（虛無下的 max|t|
  門檻、SPA、DSR、各層校正後的顯著策略數），其後依序是策略統計量、重疊膨脹、多重檢定、
  分位曲線、IC 與 Fama-MacBeth、風險因子、檢定力矩陣、策略相關矩陣、月報酬序列、
  完整規格與保留意見
- `momentum_study_summary.csv` — 策略總表（扁平）

## 架構

```
factorlab/
  config.py      研究規格（dataclass）。n_trials = 多重檢定的分母
  data.py        價格面板唯一入口。只走中央 loader，無路徑字串
  signals.py     動能訊號 + 流動性篩選
  portfolios.py  Jegadeesh-Titman 日曆時間重疊投組（主結果）
  benchmarks.py  MKT / SIZE / STREV / VOL 四因子
  costs.py       週轉率、交易成本、損益兩平 bps
  study.py       流程編排：在正確的層級套用每一種校正
  report.py      Excel 多工作表 + markdown
  cli.py         命令列
  stats/         hac · bootstrap · moments · regression · famamacbeth
                 ic · monotonicity · multiple_testing · spa · power
```

校正要套在正確的層級：**策略層**（HAC t、風險調整 alpha、單調性、GRS、成本、檢定力）、
**訊號層**（IC 與 Fama-MacBeth，與 horizon 無關、用全部股票）、**研究層**（多重檢定、
HLZ haircut、Deflated Sharpe、SPA）。執行順序有意義 —— **先跑完所有策略，再做研究層
校正**；不可以邊跑邊挑「看起來不錯的」然後只對它做校正，那會讓分母失真。

## 測試

```bash
python -m pytest tests/ -q      # 217 passed
```

離線、秒級，不碰真實 parquet。重點不是覆蓋率，而是**用已知答案驗證統計量**：對 i.i.d.
序列造出 h 期重疊和，驗證 Hansen-Hodrick 回復出理論上的 √h 膨脹（h=12 → 3.46×）；
200 個純噪音策略中挑出最佳者，DSR 必須擋下來；純隨機漫步面板上研究層校正必須擋下來，
而注入已知強度的動能時流程必須抓得到。

## 保留意見

完整清單在 Excel 的 `12_spec_and_caveats` 與 [`RESULTS.md`](RESULTS.md) 第九節。最重要的三項：

1. **SIZE 是代理。** 價格資料只有 OHLCV，沒有歷史市值（當期市值會造成前視偏誤），
   因此用日均成交金額的對數代理。這是最大的規格妥協。
2. **存續偏誤。** 成分股清單是當期的，2007 年就已消失的公司不在樣本裡；需要
   point-in-time 的成分股歷史才能修正。
3. **下市處理的方向對策略有利。** 持有期內缺報酬的個股被剔除並重新歸一化，而非當成
   −100%，所以 alpha 估計應視為上界。

## 遺留檔案

`us_factor_liqm.py` 是被取代的原始版本，保留作為對照基準，**不應執行**。
`Russell3000_price_data_3Y.parquet` 是歷史遺留的資料副本，`factorlab` 不讀它。

## 參考文獻

Jegadeesh & Titman (1993) · Newey & West (1987, 1994) · Hansen & Hodrick (1980) ·
Gibbons, Ross & Shanken (1989) · Fama & MacBeth (1973) · Politis & Romano (1994) ·
Lo (2002) / Mertens (2002) · White (2000) · Hansen (2005) · Patton & Timmermann (2010) ·
Bailey & López de Prado (2014) · Harvey, Liu & Zhu (2016)
