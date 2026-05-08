import os
import time
import random
import requests as req_lib
import polars as pl
import yfinance as yf
from pytrends.request import TrendReq
from dotenv import load_dotenv

load_dotenv()

# Countries to compare: Thailand vs major economies
COUNTRIES = {
    'TH': 'Thailand',
    'US': 'United States',
    'CN': 'China',
    'JP': 'Japan',
    'DE': 'Germany',
    'SG': 'Singapore',
}

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


# --- SECTION 1: LABOR MARKET (World Bank API) ---
# Annual data: unemployment, youth unemployment, labor force participation

LABOR_INDICATORS = {
    'unemployment_rate':        'SL.UEM.TOTL.ZS',   # % of total labor force
    'youth_unemployment_rate':  'SL.UEM.1524.ZS',   # % of youth labor force (15-24)
    'labor_force_participation': 'SL.TLF.CACT.ZS',  # % of working-age population
    'employment_ratio':         'SL.EMP.TOTL.SP.ZS', # Employment-to-population ratio
}

def fetch_worldbank(indicator_code, country='TH'):
    """
    Fetch a World Bank indicator for one country.
    Returns a polars DataFrame: year, value, country.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country}"
        f"/indicator/{indicator_code}?format=json&per_page=50&mrv=25"
    )
    resp = req_lib.get(url, headers=BROWSER_HEADERS, timeout=10)
    resp.raise_for_status()
    records = resp.json()[1]
    return pl.DataFrame([
        {'year': int(r['date']), 'value': r['value'], 'country': country}
        for r in records if r['value'] is not None
    ]).sort('year')

def fetch_labor_market(countries: dict = COUNTRIES):
    """
    Fetch all labor market indicators for all countries.
    Returns a dict of {indicator_name: polars DataFrame (all countries stacked)}.
    """
    result = {}
    for name, code in LABOR_INDICATORS.items():
        frames = []
        for iso in countries:
            try:
                frames.append(fetch_worldbank(code, country=iso))
            except Exception as e:
                print(f"[Skip] {iso} / {name}: {e}")
        if frames:
            result[name] = pl.concat(frames)
    return result


# --- SECTION 2: ECONOMIC CONTEXT (World Bank API) ---
# GDP growth, inflation, trade openness — for understanding why unemployment moves

ECONOMIC_INDICATORS = {
    'gdp_growth':      'NY.GDP.MKTP.KD.ZG',   # GDP growth rate (%)
    'inflation':       'FP.CPI.TOTL.ZG',       # Inflation, consumer prices (%)
    'trade_pct_gdp':   'NE.TRD.GNFS.ZS',       # Trade as % of GDP (globalization proxy)
    'fdi_inflows':     'BX.KLT.DINV.WD.GD.ZS', # Foreign direct investment (% of GDP)
}

def fetch_economic_context(countries: dict = COUNTRIES):
    """
    Fetch economic context indicators for all countries.
    Same structure as fetch_labor_market().
    """
    result = {}
    for name, code in ECONOMIC_INDICATORS.items():
        frames = []
        for iso in countries:
            try:
                frames.append(fetch_worldbank(code, country=iso))
            except Exception as e:
                print(f"[Skip] {iso} / {name}: {e}")
        if frames:
            result[name] = pl.concat(frames)
    return result


# --- SECTION 3: JOB SEARCH TRENDS (Google Trends) ---
# Search interest as a leading/coincident indicator for labor market sentiment.
# High "job search" queries often correlate with rising unemployment or career pivots.

JOB_KEYWORDS = {
    'TH': ['หางาน', 'สมัครงาน', 'ว่างงาน'],
    'global': ['unemployment', 'jobs', 'layoffs'],
}

def fetch_job_search_trends(keywords: list, geo='TH', timeframe='today 12-m', max_retries=5):
    """
    Fetch Google Trends interest for job-related keywords.
    Returns a pandas DataFrame (pytrends output), or None on failure.
    """
    pytrends = TrendReq(hl='en-US', tz=360, requests_args={'headers': BROWSER_HEADERS})

    for attempt in range(max_retries):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()
            if not data.empty:
                if 'isPartial' in data.columns:
                    data = data.drop(columns=['isPartial'])
                return data
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Rate limited] Retry {attempt + 1}/{max_retries} in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[Error] {e}")
                break

    return None


# --- SECTION 4: MARKET SIGNALS (yfinance) ---
# Financial markets often price in economic shifts before official data is published.

MARKET_TICKERS = {
    'SET_index':  '^SET.BK',  # Thailand stock market
    'USD_THB':    'THB=X',    # Baht strength
    'sp500':      '^GSPC',    # US S&P 500 (global risk appetite)
    'oil':        'CL=F',     # Crude oil (global growth proxy)
    'gold':       'GC=F',     # Gold (risk-off / uncertainty signal)
    'vix':        '^VIX',     # Volatility index (global fear gauge)
}

def fetch_market_signals(tickers: dict = MARKET_TICKERS, period='1y'):
    """
    Fetch closing price history for each market ticker.
    Returns a dict of {name: polars DataFrame with Date and Close columns}.
    """
    result = {}
    for name, symbol in tickers.items():
        try:
            df_pd = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if not df_pd.empty:
                df_pd = df_pd[['Close']].reset_index()
                df_pd.columns = ['date', 'close']
                result[name] = pl.from_pandas(df_pd)
        except Exception as e:
            print(f"[Skip] {name}: {e}")
    return result


if __name__ == "__main__":
    print("=== Thailand Unemployment Rate ===")
    df = fetch_worldbank('SL.UEM.TOTL.ZS', country='TH')
    print(df.tail(5))

    print("\n=== Market Signals (1 year) ===")
    markets = fetch_market_signals(period='1y')
    for name, df in markets.items():
        print(f"\n{name}:")
        print(df.tail(3))
