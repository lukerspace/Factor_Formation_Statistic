# Multi-Lookback Momentum Factor — Research-Grade Statistical Inference

多重回看期動能因子（1M / 3M / 6M / 12M）的研究級檢驗框架。

這個 repo 的重點**不是**「跑出一個動能策略」，而是**正確地檢定它到底有沒有效**。
原始版本是一支 187 行的 `us_factor_liqm.py`，它的投組建構大致合理，但最後那一行

```python
t, p = stats.ttest_1samp(x, 0)
```

讓整份研究的結論無法成立。`factorlab` 是為了修掉那一行背後的五個問題而生的。

---

## 原始檢定的五個致命問題

| # | 問題 | 為什麼致命 | 解法 |
|---|---|---|---|
| 1 | **重疊報酬**做 `ttest_1samp` | `fwd_12m` 每月取樣卻持有 12 個月，相鄰觀察值共用 11 個月的報酬。自相關是抽樣設計造成的，不是資料特性。真實 `Var(x̄) = S/T` 而非 `γ₀/T`，i.i.d. 假設讓 t 值高估約 **√h 倍**（h=12 時 3.46 倍） | `portfolios`（Jegadeesh-Titman 日曆時間重疊投組）+ `stats.hac`（Newey-West / Hansen-Hodrick） |
| 2 | **沒有風險調整** | Q10−Q1 的原始平均數不是 alpha。動能高分位系統性偏向高 beta、小型、高波動的股票，價差裡有一部分只是已知風險的補償 | `benchmarks`（MKT/SIZE/STREV/VOL 四因子）+ `stats.regression` |
| 3 | **沒有多重檢定校正** | 4 lookback × 2 skip × 4 horizon = 32 個檢定。全域虛無下 E[max\|t\|] ≈ 2.3（漸近近似 √(2 ln 32) ≈ 2.63 會高估），5% 門檻約 3.0。用 1.96 等於保證自己會「發現」東西 | `stats.multiple_testing`（Holm / BY / HLZ haircut / Deflated Sharpe）+ `stats.spa`（Hansen SPA） |
| 4 | **只看兩端、沒檢驗單調性** | Q10−Q1 顯著不代表訊號有排序能力 —— 只要 Q1 被幾檔崩跌股拉下去就會顯著，中間八個分位可以完全是平的 | `stats.monotonicity`（Patton-Timmermann MR）+ `stats.regression.grs_test`（十分位 alpha 聯合檢定） |
| 5 | **樣本只有 23–34 個月** | `p = 0.2654` 不代表「沒有動能」，代表「測不出來」。n=23、σ=5.58% 時，80% 檢定力下的最小可偵測效果是月 **3.26%**（年化 47%），而文獻動能溢酬約月 0.8% | `stats.power`（MDE / 檢定力 / 所需樣本數） |

### 另外兩個實作 bug

1. **`skip` 放錯位置。** 原始碼的 `skip_holding_month` 把再平衡日往後推一個月 —— 那是
   遲延進場，不是 skip。Jegadeesh-Titman 的 12-2 動能是跳過**形成期最後一個月**：
   訊號 = `P[t-1]/P[t-12] − 1`，持有仍從 t 開始，目的是避開一個月短期反轉與
   bid-ask bounce。修正見 `signals.momentum_signal`。
2. **流動性篩選的時點。** 原始碼用 `form_start - 1M`，`lookback=12` 時那是 **13 個月前**
   的流動性，既非下單當下可觀察，也讓不同 lookback 面對不同 universe，使跨 lookback
   比較失去意義。修正後一律用**再平衡當月**的日均成交金額。見 `signals.liquidity_universe`。

還有一個靜默失敗：分位分組從 `pd.qcut(..., duplicates='drop')` 改成 `rank(pct=True)`。
`duplicates='drop'` 會丟掉重複的分界點，然後把剩下的分界點**重新貼上 Q1..Q10 的標籤** ——
某些月份標著 `Q10` 的其實是第 7 分位，而那個價差被當成十分位價差混進時間序列。

---

## 實證結果（2007–2024，215 個月）

```
樣本：4530 個交易日 / 2094 檔 / 平均每月可選 1585 檔
掃過 28 個策略（32 個組合中 4 個 lookback=1&skip=1 形成期長度為零，依設計跳過）
```

**結論：無法在校正後主張發現。**

- 沒有任何策略的 HAC t 值超過全域虛無下 max\|t\| 的 95% 分位（**2.77**）
- 樣本內最佳 `mom1_h12`：t_HAC = **1.08**（同序列 t_iid = 1.13）
- Hansen SPA 一致 p 值 = **0.704** — 無法拒絕「沒有任何策略優於零」
- Deflated Sharpe = **0.631**（門檻 0.95）
- Holm / BY 校正後 5% 顯著的策略數：**0**
- **22/28 個策略對月 0.8% 的真實效果檢定力低於 80%**

最後一條是整份研究最重要的保留意見：**「沒顯著」不等於「沒有動能」。**
這份樣本在這個規格下分辨不出文獻量級的動能溢酬。

重疊報酬的膨脹也被量化出來了 —— 實測標準誤膨脹中位數 1.62，h=12 的策略達 2.1~2.4：

| 策略 | h | 天真 t | NW t | HH t | 實測膨脹 | 理論 √h |
|---|---|---|---|---|---|---|
| `mom12-1_h12` | 12 | −1.70 | −0.72 | −0.67 | 2.36 | 3.46 |
| `mom12_h12` | 12 | −1.66 | −0.71 | −0.65 | 2.33 | 3.46 |
| `mom6-1_h12` | 12 | −0.92 | −0.43 | −0.40 | 2.14 | 3.46 |

實測值低於理論 √h 有兩個原因，都在程式碼裡記著：Bartlett 核的權重與重疊結構的
真實權重不一致（換來半正定保證），以及分位成員每月換手會稀釋重疊造成的自相關。

---

## 安裝與執行

價格資料一律走中央 loader，**本套件不含任何 parquet 路徑字串**：

```python
from aws_fetch_price_data.fetch_fmp_price.price import load_us_price_data
```

所以需要讓 Python 找到 KICKs 專案根目錄：

```bash
export PYTHONPATH=/path/to/KICKs          # 或
export FACTORLAB_KICKS_ROOT=/path/to/KICKs
```

```bash
# 完整研究（預設 history="long" = 2007–2024，215 個月）
python -m factorlab.cli run --history long --out output

# 示範 3 年樣本為什麼不該用
python -m factorlab.cli run --history 3y --out output_3y

# 自訂網格
python -m factorlab.cli run --lookbacks 6,12 --skips 1 --horizons 1,6

# 重算原始輸出的檢定力（n=23、σ=5.576%）
python -m factorlab.cli power --sigma 0.05576 --nobs 23 --observed-mean 0.01329

# 列出可用資料集的涵蓋範圍
python -m factorlab.cli info
```

程式化使用：

```python
from factorlab import StudySpec, run_study, write_all

result = run_study(StudySpec())      # 預設 long
write_all(result, "output")

print(result.best.tag, result.best.hac.tstat)
print(result.spa.p_consistent, result.dsr.dsr)
```

---

## 輸出

| 檔案 | 內容 |
|---|---|
| `momentum_study.md` | 研究結論。結論一節的判定是**規則產生**的，不是人工挑字 |
| `momentum_study.xlsx` | 12 個工作表，見下 |
| `momentum_study_summary.csv` | 策略總表（扁平） |

Excel 工作表刻意把**研究層結論放第一張**，而不是附錄：

| 工作表 | 內容 |
|---|---|
| `01_research_level` | 虛無下的 max\|t\| 門檻、SPA、DSR、各層校正後的顯著策略數 |
| `02_strategy_summary` | 每個策略的完整統計量 |
| `03_overlap_inflation` | 天真 t / NW t / HH t 並列 + 實測與理論膨脹倍數 |
| `04_multiple_testing` | Holm / BY / Bonferroni 校正後 p 值與 HLZ Sharpe 折價 |
| `05_quantile_profiles` | 十個分位的報酬曲線 |
| `06_signal_level` | IC 與 Fama-MacBeth（用全部股票，不只兩端） |
| `07_factors` | 四個風險因子本身的統計量 |
| `08_power` | 檢定力矩陣：樣本月數 × 真實效果大小 |
| `09_correlations` | 策略間相關矩陣（多重檢定的有效獨立數） |
| `10/11_*_returns` | 月報酬序列 |
| `12_spec_and_caveats` | 完整規格 JSON、執行期警告、保留意見 |

---

## 架構

```
factorlab/
  config.py          研究規格（dataclass）。n_trials = 多重檢定的分母
  data.py            價格面板唯一入口。只走中央 loader，無路徑字串
  signals.py         動能訊號 + 流動性篩選（修掉兩個實作 bug）
  portfolios.py      Jegadeesh-Titman 日曆時間重疊投組（主結果）
  benchmarks.py      MKT / SIZE / STREV / VOL 四因子
  costs.py           週轉率、交易成本、損益兩平 bps
  study.py           流程編排：在正確的層級套用每一種校正
  report.py          Excel 多工作表 + markdown
  cli.py             命令列
  stats/
    hac.py              Newey-West / Hansen-Hodrick（問題 1）
    bootstrap.py        Politis-Romano 定態 bootstrap
    moments.py          Sharpe 與其標準誤（Lo 2002 偏態峰態修正）
    regression.py       HAC 迴歸 + GRS 聯合檢定（問題 2、4）
    famamacbeth.py      兩階段橫斷面迴歸
    ic.py               資訊係數（用全部股票而非兩端）
    monotonicity.py     Patton-Timmermann MR 檢定（問題 4）
    multiple_testing.py Holm / BY / HLZ haircut / Deflated Sharpe（問題 3）
    spa.py              Hansen SPA / White Reality Check（問題 3）
    power.py            檢定力與最小可偵測效果（問題 5）
```

### 為什麼「校正要套在正確的層級」

- **策略層**：HAC t、風險調整 alpha、單調性、GRS、成本、檢定力
- **訊號層**：IC 與 Fama-MacBeth（與 horizon 無關，用全部股票）
- **研究層**：多重檢定、HLZ haircut、Deflated Sharpe、SPA

執行順序有意義：**先跑完所有策略，再做研究層校正**。不可以邊跑邊挑
「看起來不錯的」然後只對它做校正 —— 那會讓分母失真。

---

## 測試

```bash
python -m pytest tests/ -q      # 217 passed
```

測試不碰真實 parquet（離線、秒級）。重點不是覆蓋率，而是**用已知答案驗證統計量**：

- `test_hac.py` — 對 i.i.d. 序列造出 h 期重疊和，驗證 Hansen-Hodrick 回復出
  理論上的 **√h 膨脹**（h=12 → 3.46×）。這是整個 repo 最重要的一組測試
- `test_signals.py` — 兩個實作 bug 的回歸測試：`12-2` 的訊號必須是
  `1.1^11 − 1` 而非 `1.1^12 − 1`；流動性篩選必須用再平衡當月
- `test_portfolios.py` — 日曆時間投組的無前視性、週轉率 ∝ 1/h、
  缺值剔除而非當成 0%
- `test_monotonicity.py` — 「只有兩端有差、中間是平的」時 MR 檢定**不該**拒絕
- `test_multiple_testing.py` — 200 個純噪音策略中挑出最佳者，DSR 必須擋下來
- `test_spa.py` — 所有策略都虧錢時三個 p 值都必須是 1
- `test_power.py` — 重現原始輸出的 n=23、σ=5.576%，驗證 MDE > 月 3%
- `test_study.py` — 端到端。**純隨機漫步面板上，研究層校正必須擋下來**；
  注入已知強度的動能時，流程必須抓得到

---

## 保留意見

完整清單在 Excel 的 `12_spec_and_caveats`，摘要：

1. **SIZE 是代理。** 價格 parquet 只有 OHLCV，沒有歷史市值；`FMP_stocks.csv` 的市值是
   當期值，用它會造成前視偏誤。因此用日均成交金額的對數當代理。這是最大的規格妥協。
2. **存續偏誤。** Russell 3000 成分股清單是當期的，2007 年就已消失的公司不在樣本裡。
   需要 point-in-time 成分股歷史才能修正。
3. **下市處理。** 持有期內缺報酬的個股被剔除並重新歸一化，而非當成 −100%。
   方向對策略有利，所以 alpha 估計應視為上界。
4. **日報酬離群值過濾是不對稱的。** `|r| > 100%` 的門檻只擋得住反向分割；
   正向分割呈現 −90%，不會被擋掉（實測顯示本資料集已做分割調整：
   710 萬個觀察值中僅 55 個落在 −50% 附近，遠低於 18 年該有的分割次數）。
5. **GRS 假設殘差常態。** 僅作輔助證據，主結論以 HAC t 與 bootstrap 為準。
6. **成本模型是靜態的。** 單邊 6 bps 對近年大型股合理，但 2008–09 的實際價差遠高於此。
   `breakeven_bps` 是比淨 Sharpe 更穩健的解讀方式。
7. **參數網格仍是人選的。** 校正的分母是實際掃過的 28 個組合，但
   `lookback ∈ {1,3,6,12}` 本身依文獻慣例選定 —— 真正的研究者自由度比 28 更大。

---

## 遺留檔案

`us_factor_liqm.py` 是被取代的原始版本，**保留作為對照基準，不應執行**。
它硬編了 parquet 檔名（違反中央 loader 規則）、skip 放在持有期、流動性篩選用錯時點，
並用 `ttest_1samp` 對重疊報酬做檢定。`factorlab` 的每一個設計決定都是針對它而來。

repo 內的 `Russell3000_price_data_3Y.parquet`（75MB）是歷史遺留的資料副本。
`factorlab` 不讀它 —— 價格一律走中央 loader。

---

## 參考文獻

- Jegadeesh & Titman (1993) — 動能與日曆時間重疊投組
- Newey & West (1987, 1994) — HAC 標準誤與自動帶寬
- Hansen & Hodrick (1980) — 重疊報酬的截斷核
- Gibbons, Ross & Shanken (1989) — alpha 聯合檢定
- Fama & MacBeth (1973) — 兩階段橫斷面迴歸
- Politis & Romano (1994) — 定態 bootstrap
- Lo (2002) / Mertens (2002) — Sharpe 的標準誤
- White (2000) — Reality Check
- Hansen (2005) — Superior Predictive Ability
- Patton & Timmermann (2010) — 單調性檢定
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio
- Harvey, Liu & Zhu (2016) — `…and the Cross-Section of Expected Returns`（t > 3.0 門檻）
