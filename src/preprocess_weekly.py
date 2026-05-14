"""
preprocess_weekly.py — Build unified_weekly.csv from all raw data sources.

Extends the monthly pipeline (02_data_cleaning.ipynb) to weekly frequency:
  - Daily market signals  → resample to W-FRI (Friday close)
  - Monthly FRED data     → forward-fill to W-FRI
  - Annual macro data     → broadcast to every week in that year
  - USD/THB gap 2000-2003 → filled from FRED EXTHUS monthly series

Output: data/processed/unified_weekly.csv
  Training window (2000-09 → 2019-12) ≈ 1,000 weeks  (vs 180 monthly)
  Test window     (2020-01 → 2025-12) ≈ 310  weeks
"""

from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RAW_DIR  = Path('data/raw')
PROC_DIR = Path('data/processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Gold/Oil available from Aug 2000 — start weekly data from Sep 2000
CLIP_START = pd.Timestamp('2000-09-01')
CLIP_END   = pd.Timestamp('2025-12-31')
WEEK_FREQ  = 'W-FRI'   # week ending Friday (last trading day)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_weekly_price(series: pd.Series) -> pd.Series:
    """Daily price series → weekly Friday close (ffill weekends/holidays ≤5d)."""
    biz = pd.bdate_range(series.index.min(), series.index.max())
    return series.reindex(biz).ffill(limit=5).resample(WEEK_FREQ).last()


def _monthly_to_weekly(series: pd.Series, ffill_limit: int = 5) -> pd.Series:
    """Monthly series → weekly by forward-filling each monthly value."""
    monthly = series.resample('ME').last().ffill(limit=2)
    week_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq=WEEK_FREQ)
    return monthly.reindex(week_idx, method='ffill').ffill(limit=ffill_limit)


def _annual_to_weekly(df_annual: pd.DataFrame, week_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Annual int-indexed DataFrame → broadcast each year's value to every week."""
    out = pd.DataFrame(index=week_idx, dtype=float)
    for col in df_annual.columns:
        mapping = {w: df_annual.loc[w.year, col]
                   for w in week_idx
                   if w.year in df_annual.index and not pd.isna(df_annual.loc[w.year, col])}
        out[col] = pd.Series(mapping)
    return out


def _winsorize(ret_df: pd.DataFrame, lo=0.01, hi=0.99) -> pd.DataFrame:
    """Clip extreme returns at lo/hi quantiles (per column)."""
    out = ret_df.copy()
    for col in out.columns:
        s = out[col].dropna()
        if len(s) < 20:
            continue
        out[col] = out[col].clip(s.quantile(lo), s.quantile(hi))
    return out


# ── Step 1: Market prices → weekly ────────────────────────────────────────────

MARKET_NAMES = ['SET_index', 'vix', 'gold', 'oil', 'sp500',
                'us_10yr_treasury', 'nasdaq']

print('Step 1: Loading daily market signals → weekly ...')
price_dict = {}
for name in MARKET_NAMES:
    df = pd.read_csv(RAW_DIR / f'{name}_market_signals.csv', parse_dates=['date'])
    s  = df.set_index('date')['close'].rename(name).sort_index()
    price_dict[name] = _to_weekly_price(s)
    print(f'  {name:<22} {price_dict[name].index[0].date()} → '
          f'{price_dict[name].index[-1].date()}  ({len(price_dict[name])} weeks)')

# USD/THB: yfinance from 2003-12, FRED EXTHUS monthly for the 2000-2003 gap
yf_raw = pd.read_csv(RAW_DIR / 'USD_THB_market_signals.csv', parse_dates=['date'])
yf_w   = _to_weekly_price(yf_raw.set_index('date')['close'].rename('USD_THB').sort_index())

fred_thb = pd.read_csv(RAW_DIR / 'fred_usd_thb_monthly.csv', parse_dates=['date'])
fred_thb_w = _monthly_to_weekly(fred_thb.set_index('date')['value'].rename('USD_THB').sort_index())

# Merge: FRED fills 2000-2003 gap, yfinance takes over after
usd_thb_w = fred_thb_w.combine_first(yf_w).rename('USD_THB')
price_dict['USD_THB'] = usd_thb_w
print(f'  {"USD_THB (combined)":<22} {usd_thb_w.index[0].date()} → '
      f'{usd_thb_w.index[-1].date()}  ({len(usd_thb_w)} weeks)')

market_price_w = pd.concat(price_dict, axis=1).sort_index()


# ── Step 2: Weekly returns + winsorize ────────────────────────────────────────

print('\nStep 2: Computing weekly returns (winsorized 1%/99%) ...')
market_ret_w = _winsorize(market_price_w.pct_change())
market_ret_w.columns = [f'{c}_ret_w' for c in market_ret_w.columns]
market_price_w.columns = [f'{c}_price' for c in market_price_w.columns]
print(f'  Ret shape: {market_ret_w.shape}   Price shape: {market_price_w.shape}')


# ── Step 3: FRED monthly → weekly ─────────────────────────────────────────────

FRED_NAMES = [
    'th_exchange_rate_real', 'th_us_imports', 'th_property_prices',
    'th_uncertainty', 'us_fed_funds_rate', 'us_cpi_monthly',
    'us_unemployment', 'us_industrial_prod', 'global_uncertainty',
    'us_consumer_sentiment', 'us_govt_spending',
]

print('\nStep 3: Loading FRED monthly → weekly (forward-fill) ...')
fred_dict = {}
for name in FRED_NAMES:
    path = RAW_DIR / f'fred_{name}.csv'
    if not path.exists():
        print(f'  [skip] {name} — file not found')
        continue
    df = pd.read_csv(path, parse_dates=['date'])
    s  = df.set_index('date')['value'].rename(name).sort_index()
    fred_dict[name] = _monthly_to_weekly(s)
    print(f'  {name:<30} {fred_dict[name].index[0].date()} → '
          f'{fred_dict[name].index[-1].date()}')

fred_w = pd.concat(fred_dict, axis=1).sort_index()


# ── Step 4: Annual macro → weekly ─────────────────────────────────────────────

def _load_macro(filename, col):
    df = pd.read_csv(RAW_DIR / filename)
    df['year']  = pd.to_numeric(df['year'],  errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna(subset=['year', 'value']).set_index('year')['value'].rename(col)

print('\nStep 4: Loading annual macro → weekly broadcast ...')
macro = pd.concat([
    _load_macro('consumption_pct_gdp_TH.csv',     'consumption'),
    _load_macro('lending_rate_TH.csv',             'lending_rate'),
    _load_macro('inflation_TH.csv',                'inflation'),
    _load_macro('thailand_unemployment_rate.csv',  'unemployment'),
    _load_macro('gross_capital_formation_TH.csv',  'business_invest'),
    _load_macro('exports_pct_gdp_TH.csv',          'exports_pct_gdp'),
    _load_macro('imports_pct_gdp_TH.csv',          'imports_pct_gdp'),
    _load_macro('imf_gdp_growth_TH.csv',           'gdp_growth'),
    _load_macro('govt_expenditure_pct_gdp_TH.csv', 'govt_expenditure'),
    _load_macro('govt_debt_pct_gdp_TH.csv',        'govt_debt'),
], axis=1)
macro = macro[(macro.index >= 2000) & (macro.index <= 2025)]
macro = macro.interpolate(method='linear', limit=2, limit_direction='forward')
print(f'  Annual macro: {macro.shape}  years {int(macro.index.min())}–{int(macro.index.max())}')


# ── Step 5: Merge all → clip → save ──────────────────────────────────────────

print('\nStep 5: Merging all sources ...')

# Build common weekly index
week_idx = pd.date_range(start=CLIP_START, end=CLIP_END, freq=WEEK_FREQ)

macro_w = _annual_to_weekly(macro, week_idx)
macro_w.columns = [f'{c}_annual' for c in macro_w.columns]

unified_w = (
    market_ret_w
    .join(market_price_w, how='outer')
    .join(fred_w,         how='outer')
    .join(macro_w,        how='outer')
    .sort_index()
)

# Clip to target window
unified_w = unified_w[(unified_w.index >= CLIP_START) & (unified_w.index <= CLIP_END)]

# Rename ret columns to match existing model convention but with _w suffix
# (model notebook will detect _ret_w suffix)
print(f'\n  Unified weekly: {unified_w.shape}')
print(f'  Date range   : {unified_w.index[0].date()} → {unified_w.index[-1].date()}')

train = unified_w[unified_w.index <= '2019-12-31']
test  = unified_w[unified_w.index >= '2020-01-01']
print(f'  Training rows: {len(train)}  (2000-09 → 2019-12)')
print(f'  Test rows    : {len(test)}   (2020-01 → 2025-12)')

# Missing value report
miss = unified_w.isna().mean().sort_values(ascending=False).head(10)
print(f'\n  Top missing columns:')
for col, pct in miss[miss > 0].items():
    print(f'    {col:<35} {pct*100:.1f}%')

unified_w.to_csv(PROC_DIR / 'unified_weekly.csv')
print(f'\n✓  Saved: data/processed/unified_weekly.csv')


if __name__ == '__main__':
    pass  # all steps run at import time for script-style execution
