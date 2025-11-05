## Multi-Lookback Momentum & Liquidity Factor Factory (1M / 3M / 6M / 12M)

**Research-Grade Factor Backtesting Framework** | Python + Pandas + Parquet | One-File Output (Excel / CSV)

---

## Overview
This project builds a **multi-horizon momentum factor engine** integrated with a **liquidity screening layer**.  
It automates monthly cross-sectional formation, decile grouping, and forward-return computation with flexible skip periods.  
Designed for **quantitative equity research** and portfolio prototyping.

---

## Core Features
- **Multiple formation lookbacks** → 1M / 3M / 6M / 12M  
- Optional **skip-1-month holding** to avoid look-ahead bias  
- **Liquidity filter**: selects stocks with prior 1-month average dollar volume ≥ market median  
- **Automatic decile grouping** (Q1–Q10) and **spread computation** (Q10–Q1, Q9–Q2)  
- Generates a **single Excel sheet** (`Momentum_MultiLookback_Liquidity_SingleSheet_.xlsx`)  
- Includes **t-test / Sharpe ratio statistics** for forward 1-month returns

---

## Data Input
- `Russell3000_price_data_3Y.parquet`  
  Required columns:  
  - `Close` (daily closing price)  
  - `Volume` (daily trading volume)

The script automatically calculates **amount = Close × Volume**, reindexes by `date` and `ticker`,  
and aligns both price and liquidity data for consistent monthly rebalancing.

---

## Output Files
| File | Description |
|------|--------------|
| `Momentum_MultiLookback_Liquidity_SingleSheet_.xlsx` | Core output with all deciles & spreads (1M / 3M / 6M / 12M) |
| `Momentum_Strategy_fw1m_stats_by_decile.csv` | Statistical summary (mean, std, t-stat, p-value, annualized Sharpe) |

---

## Run Example
```bash
python us_factor_liqm.py
