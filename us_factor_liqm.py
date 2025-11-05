import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = "Russell3000_price_data_3Y.parquet"
SAVE_FILE = "Momentum_MultiLookback_Liquidity_SingleSheet_.xlsx"
LOOKBACKS = [1, 3, 6, 12]
SKIP_OPTIONS = [False, True]
HORIZONS = [1, 3, 6, 12]
MIN_STOCKS = 50

print("讀取資料...")
price = pd.read_parquet(DATA_PATH)["Close"]
volume = pd.read_parquet(DATA_PATH)["Volume"]

def standardize_df(df, value_name):
    if df.index.name == 'Date' or 'Date' in df.index.names:
        df = df.reset_index()
        df.rename(columns={'Date': 'date'}, inplace=True)
    if 'date' not in df.columns:
        df = df.melt(id_vars=df.columns[0], var_name='ticker', value_name=value_name)
    elif 'ticker' not in df.columns:
        df = df.melt(id_vars='date', var_name='ticker', value_name=value_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date', value_name]).sort_values(['ticker', 'date']).reset_index(drop=True)

df_price = standardize_df(price, 'Close')
df_vol = standardize_df(volume, 'Volume')
df = pd.merge(df_price, df_vol, on=['date', 'ticker'], how='inner')
df['amount'] = df['Close'] * df['Volume']

print("建立 lookup...")
price_lookup = df.set_index(['date', 'ticker'])[['Close', 'amount']].sort_index()
idx = pd.IndexSlice
month_ends = df.groupby(df['date'].dt.to_period('M'))['date'].max().sort_values().tolist()

def build_factor_panel(lookback, skip_holding_month=False):
    out = []
    tag = f"{lookback}M" + ("_skip1M" if skip_holding_month else "")
    for rebalance_date in month_ends:
        form_start = rebalance_date - pd.DateOffset(months=lookback)
        form_end = rebalance_date
        try:
            form_slice = price_lookup.loc[idx[form_start:form_end], :]
        except KeyError:
            continue
        if form_slice.empty:
            continue

        liq_start = form_start - pd.DateOffset(months=1)
        try:
            liq_slice = price_lookup.loc[idx[liq_start:form_start], 'amount']
        except KeyError:
            continue
        if liq_slice.empty:
            continue
        avg_amt = liq_slice.groupby(level='ticker').mean()
        cutoff = avg_amt.median()
        liquid_tickers = avg_amt[avg_amt >= cutoff].index
        if len(liquid_tickers) < MIN_STOCKS:
            continue

        form_slice = form_slice.loc[pd.IndexSlice[:, liquid_tickers], :]
        first = form_slice.groupby(level='ticker')['Close'].first()
        last = form_slice.groupby(level='ticker')['Close'].last()
        momentum = (last / first - 1).dropna()
        if len(momentum) < MIN_STOCKS:
            continue

        tmp = pd.DataFrame({
            'momentum': momentum,
            'rebalance_date': rebalance_date,
            'lookback': f"{lookback}M",
            'lookback_interval': f"{form_start:%Y-%m-%d} → {form_end:%Y-%m-%d}"
        })
        tmp['decile'] = pd.qcut(tmp['momentum'], 10, labels=[f'Q{i}' for i in range(1, 11)], duplicates='drop')

        start_hold = rebalance_date + pd.DateOffset(months=1) if skip_holding_month else rebalance_date
        actual_start = df[df['date'] > start_hold]['date'].min()
        if pd.isna(actual_start):
            continue
        hold_start = df[df['date'] == actual_start].set_index('ticker')['Close']
        hold_start = hold_start.reindex(tmp.index).dropna()
        if hold_start.empty:
            continue

        for h in HORIZONS:
            tmp[f"fwd_{h}m"] = np.nan

        for h in HORIZONS:
            end_fwd = actual_start + pd.DateOffset(months=h)
            if end_fwd > df['date'].max():
                continue
            try:
                fwd = price_lookup.loc[idx[actual_start:end_fwd], 'Close']
            except KeyError:
                continue
            if fwd.empty:
                continue
            last_fwd = fwd.groupby(level='ticker').last()
            tmp[f"fwd_{h}m"] = last_fwd.reindex(tmp.index) / hold_start - 1

        qavg = tmp.groupby(['rebalance_date', 'lookback', 'lookback_interval', 'decile'])[
            [f"fwd_{h}m" for h in HORIZONS]
        ].mean().reset_index()
        qavg['fwd_start_date'] = actual_start.strftime("%Y-%m-%d")
        qavg['strategy'] = tag
        out.append(qavg)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# 跑全部策略
all_panels = []
for lb in LOOKBACKS:
    for skip in SKIP_OPTIONS:
        panel = build_factor_panel(lb, skip)
        all_panels.append(panel)
all_data = pd.concat(all_panels, ignore_index=True) if all_panels else pd.DataFrame()

# spread 計算
def calc_spreads(df):
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=['rebalance_date', 'lookback', 'lookback_interval', 'fwd_start_date', 'strategy'],
        columns='decile',
        values=[f"fwd_{h}m" for h in HORIZONS]
    )
    out = []
    def make_spread(high, low, name):
        s = pd.DataFrame({
            'rebalance_date': pivot.index.get_level_values(0),
            'lookback': pivot.index.get_level_values(1),
            'lookback_interval': pivot.index.get_level_values(2),
            'fwd_start_date': pivot.index.get_level_values(3),
            'strategy': pivot.index.get_level_values(4),
            'decile_or_spread': name
        })
        for h in HORIZONS:
            col = f"fwd_{h}m"
            hi = pivot.get((col, high))
            lo = pivot.get((col, low))
            s[col] = hi.values - lo.values if (hi is not None and lo is not None) else np.nan
        return s
    out.append(make_spread('Q10', 'Q1', 'Q10-Q1'))
    out.append(make_spread('Q9', 'Q2', 'Q9-Q2'))
    return pd.concat(out, ignore_index=True)

spreads = [calc_spreads(all_data[all_data['strategy'] == s]) for s in all_data['strategy'].unique()]
final = pd.concat(spreads, ignore_index=True).sort_values(['lookback', 'rebalance_date'])
for h in HORIZONS:
    final[f"fwd_{h}m"] = (final[f"fwd_{h}m"] * 100).round(2)

cols = ['rebalance_date', 'fwd_start_date', 'lookback', 'lookback_interval', 'strategy',
        'decile_or_spread'] + [f"fwd_{h}m" for h in HORIZONS]
final = final[cols]

final.to_excel(SAVE_FILE, index=False)
print("done →", SAVE_FILE)

# t 檢定
df = pd.read_excel(SAVE_FILE).dropna(subset=["fwd_1m"])
out = []
for (s, decile), g in df.groupby(["strategy", "decile_or_spread"]):
    x = g["fwd_1m"].dropna()
    if len(x) < 5:
        continue
    mean = x.mean()
    std = x.std(ddof=1)
    t, p = stats.ttest_1samp(x, 0)
    sharpe = mean / std * np.sqrt(12) if std > 0 else np.nan
    out.append({
        "strategy": s,
        "decile_or_spread": decile,
        "n_obs": len(x),
        "mean_1m(%)": round(mean, 3),
        "std_1m(%)": round(std, 3),
        "t_stat": round(t, 3),
        "p_value": round(p, 4),
        "sharpe(annualized)": round(sharpe, 3)
    })

summary = pd.DataFrame(out).sort_values(["strategy", "decile_or_spread"])
summary.to_csv("Momentum_Strategy_fw1m_stats_by_decile.csv", index=False)
print(summary)
print("已輸出 CSV")
