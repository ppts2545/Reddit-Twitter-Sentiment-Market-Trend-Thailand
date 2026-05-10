import os
import time
import random
import feedparser
import requests as req_lib
import polars as pl
import yfinance as yf
from fredapi import Fred
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# Common start date for all data sources — 2003 is the earliest year
# where ALL market tickers (including USD/THB) have data.
DATA_START = '2003-01-01'

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
    # Sector employment — tracks shift from agriculture → industry → services
    'employment_agriculture':   'SL.AGR.EMPL.ZS',   # % employed in agriculture
    'employment_industry':      'SL.IND.EMPL.ZS',   # % employed in industry
    'employment_services':      'SL.SRV.EMPL.ZS',   # % employed in services
}

def fetch_worldbank(indicator_code, country='TH'):
    """
    Fetch a World Bank indicator for one country.
    Returns a polars DataFrame: year, value, country.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country}"
        f"/indicator/{indicator_code}?format=json&per_page=50&mrv=22"
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
    # --- 1. Consumption ---
    'consumption_pct_gdp':    'NE.CON.PRVT.ZS',    # Household final consumption (% of GDP)

    # --- 2. Interest Rate / Monetary Policy ---
    'lending_rate':           'FR.INR.LEND',        # Commercial lending rate (%)
    'broad_money_gdp':        'FM.LBL.BMNY.GD.ZS',  # Broad money M2 (% of GDP)

    # --- 3. Inflation ---
    'inflation':              'FP.CPI.TOTL.ZG',     # CPI inflation (%)

    # --- 4. Unemployment → in LABOR_INDICATORS already ---

    # --- 5. Business Investment ---
    'gross_capital_formation': 'NE.GDI.TOTL.ZS',   # Gross capital formation (% of GDP)
    'fdi_inflows':             'BX.KLT.DINV.WD.GD.ZS', # FDI net inflows (% of GDP)

    # --- 6. Exports / Trade ---
    'exports_pct_gdp':        'NE.EXP.GNFS.ZS',    # Exports of goods & services (% of GDP)
    'imports_pct_gdp':        'NE.IMP.GNFS.ZS',    # Imports of goods & services (% of GDP)
    'trade_pct_gdp':          'NE.TRD.GNFS.ZS',    # Total trade openness (% of GDP)

    # --- 7. Geopolitical / Confidence → VIX + gold in MARKET_TICKERS ---

    # --- 8. Technology / Innovation ---
    'gdp_growth':             'NY.GDP.MKTP.KD.ZG',  # GDP growth (captures tech cycles)

    # --- 9. Government Policy ---
    'govt_expenditure_pct_gdp': 'GC.XPN.TOTL.GD.ZS', # Govt expenditure (% of GDP)
    'govt_debt_pct_gdp':        'GC.DOD.TOTL.GD.ZS',  # Central govt debt (% of GDP)
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

# Google Trends keyword groups mapped to the 9 economic factors
TREND_KEYWORD_GROUPS = {
    'consumption_TH':   ['ซื้อบ้าน', 'ซื้อรถ', 'ผ่อนบ้าน'],          # factor 1
    'confidence_TH':    ['ลงทุน', 'หุ้น', 'เศรษฐกิจไทย'],             # factor 5,9
    'technology_global':['artificial intelligence', 'automation', 'AI jobs'],  # factor 8
    'geopolitical':     ['war', 'sanctions', 'trade war'],              # factor 7
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
    # Thailand
    'SET_index':        '^SET.BK',  # Thailand stock market (factor 1,5,9)
    'USD_THB':          'THB=X',    # Baht strength (factor 6)
    # Global risk / geopolitical (factor 7)
    'vix':              '^VIX',     # Fear gauge — spikes during war/crisis
    'gold':             'GC=F',     # Safe haven — rises when confidence falls
    # Global growth (factor 6)
    'sp500':            '^GSPC',    # US market (global demand proxy)
    'oil':              'CL=F',     # Crude oil (global growth / supply chain)
    # Interest rate (factor 2)
    'us_10yr_treasury': '^TNX',     # US 10-yr yield — global rate benchmark
    # Technology / innovation (factor 8)
    'nasdaq':           '^IXIC',    # Nasdaq — tech sector performance
}

def fetch_market_signals(tickers: dict = MARKET_TICKERS, start: str = DATA_START):
    """
    Fetch closing price history for each market ticker from `start` date to today.
    Returns a dict of {name: polars DataFrame with date and close columns}.
    """
    result = {}
    for name, symbol in tickers.items():
        try:
            df_pd = yf.download(symbol, start=start, progress=False, auto_adjust=True)
            if not df_pd.empty:
                df_pd = df_pd[['Close']].reset_index()
                df_pd.columns = ['date', 'close']
                result[name] = pl.DataFrame({
                    'date':  df_pd['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'close': df_pd['close'].astype(float).tolist(),
                })
        except Exception as e:
            print(f"[Skip] {name}: {e}")
    return result


# --- SECTION 5: NEWS SENTIMENT (Bangkok Post RSS) ---
# Parse English economic headlines and score sentiment with VADER.
# Negative sentiment in business news often precedes economic deterioration.

BANGKOK_POST_RSS = {
    'business': 'https://www.bangkokpost.com/rss/data/business.xml',
    'economy':  'https://www.bangkokpost.com/rss/data/economy.xml',
}

def fetch_news_sentiment(feeds: dict = BANGKOK_POST_RSS):
    """
    Fetch latest headlines from Bangkok Post RSS and score each with VADER sentiment.
    Returns a polars DataFrame: title, published, compound, sentiment, feed.
    compound score: -1 (very negative) to +1 (very positive)
    """
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for feed_name, url in feeds.items():
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            score = analyzer.polarity_scores(entry.title)['compound']
            rows.append({
                'title':     entry.title,
                'published': entry.get('published', ''),
                'compound':  score,
                'sentiment': 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'),
                'feed':      feed_name,
            })
    return pl.DataFrame(rows)


# --- SECTION 6: IMF API — Monetary Policy Indicators ---
# Replaces BOT API (website unstable). IMF is free, no key needed, covers all countries.
# Dataset: IFS (International Financial Statistics)
# Docs: https://datahelp.imf.org/knowledgebase/articles/667681

# --- BOT API (commented out — website down as of 2026-05) ---
# BOT_BASE_URL = 'https://apigw1.bot.or.th/bot/public'
# Portal: https://www.bot.or.th — restore this section when BOT API is back.

# Switched from SDMX endpoint (down) to IMF DataMapper API (stable, no key needed)
IMF_BASE_URL = 'https://www.imf.org/external/datamapper/api/v1'

IMF_INDICATORS = {
    'gdp_growth':  'NGDP_RPCH',  # Real GDP growth (%)
    'inflation':   'PCPIPCH',    # Inflation, CPI (%)
    'unemployment':'LUR',         # Unemployment rate (%)
    'current_acct':'BCA_NGDPD',  # Current account (% of GDP)
}

# DataMapper uses 3-letter ISO codes
IMF_COUNTRY_CODES = {
    'TH': 'THA', 'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'SG': 'SGP',
}

def fetch_imf(indicator_key: str, country='TH'):
    """
    Fetch an IMF DataMapper indicator for one country.
    Returns a polars DataFrame: year, value, country, indicator.
    """
    iso3 = IMF_COUNTRY_CODES.get(country, country)
    indicator_code = IMF_INDICATORS[indicator_key]
    url = f"{IMF_BASE_URL}/{indicator_code}/{iso3}"
    resp = req_lib.get(url, headers=BROWSER_HEADERS, timeout=10)
    resp.raise_for_status()

    values = resp.json().get('values', {}).get(indicator_code, {}).get(iso3, {})
    return pl.DataFrame([
        {'year': int(yr), 'value': val, 'country': country, 'indicator': indicator_key}
        for yr, val in values.items() if val is not None and int(yr) >= 2003
    ]).sort('year')


# --- SECTION 7: UN COMTRADE — Import / Export by Sector ---
# Free API, no key required for basic queries (500 req/hour limit).
# Thailand reporter code = 764
# flowCode: M = imports, X = exports

# UN Comtrade removed — now requires paid subscription.
# Trade data (exports/imports % of GDP) moved to ECONOMIC_INDICATORS via World Bank.


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def run_and_save(label, filename, fn, *args, **kwargs):
    print(f"\n=== {label} ===")
    try:
        df = fn(*args, **kwargs)
        df.write_csv(os.path.join(DATA_DIR, filename))
        print(df.tail(5))
    except Exception as e:
        print(f"[SKIP] {label}: {e}")

if __name__ == "__main__":
    run_and_save('Thailand Unemployment Rate', 'thailand_unemployment_rate.csv',
                 fetch_worldbank, 'SL.UEM.TOTL.ZS', country='TH')

    print("\n=== Market Signals (2003→present) ===")
    try:
        markets = fetch_market_signals()
        for name, df in markets.items():
            df.write_csv(os.path.join(DATA_DIR, f'{name}_market_signals.csv'))
            print(f"{name}: {df.tail(1)}")
    except Exception as e:
        print(f"[SKIP] Market Signals: {e}")

    run_and_save('News Sentiment (Bangkok Post)', 'news_sentiment.csv',
                 fetch_news_sentiment)

    run_and_save('IMF GDP Growth (Thailand)', 'imf_gdp_growth_TH.csv',
                 fetch_imf, 'gdp_growth', country='TH')

    # --- 9 Economic Factors: new World Bank indicators ---
    WB_SAVES = {
        'consumption_pct_gdp':     'NE.CON.PRVT.ZS',
        'gross_capital_formation':  'NE.GDI.TOTL.ZS',
        'govt_expenditure_pct_gdp': 'GC.XPN.TOTL.GD.ZS',
        'govt_debt_pct_gdp':        'GC.DOD.TOTL.GD.ZS',
        'exports_pct_gdp':          'NE.EXP.GNFS.ZS',
        'imports_pct_gdp':          'NE.IMP.GNFS.ZS',
        'lending_rate':             'FR.INR.LEND',
        'inflation':                'FP.CPI.TOTL.ZG',
    }
    for fname, code in WB_SAVES.items():
        run_and_save(fname, f'{fname}_TH.csv', fetch_worldbank, code, country='TH')

    # --- FRED: Monthly data for Thailand (9 factors, high frequency) ---
    print("\n=== FRED Monthly Indicators (Thailand) ===")
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print("[SKIP] FRED_API_KEY not set in .env")
    else:
        fred = Fred(api_key=fred_key)

        FRED_SERIES = {
            # --- Thailand-specific (monthly/quarterly) ---
            'th_exchange_rate_real': 'RBTHBIS',      # Real effective exchange rate (factor 6)
            'th_us_imports':         'IMP5490',       # US imports from Thailand (factor 6)
            'th_property_prices':    'QTHR628BIS',    # Bangkok property prices (factor 1)
            'th_uncertainty':        'WUITHA',        # World uncertainty index TH (factor 7)

            # --- US/Global monthly (affects Thailand directly) ---
            # Factor 2: Interest rate
            'us_fed_funds_rate':     'FEDFUNDS',      # Fed rate → affects THB & capital flows
            # Factor 3: Inflation
            'us_cpi_monthly':        'CPIAUCSL',      # US CPI → global inflation signal
            # Factor 4: Unemployment
            'us_unemployment':       'UNRATE',        # US unemployment → global demand
            # Factor 5: Business investment
            'us_industrial_prod':    'INDPRO',        # US industrial production
            # Factor 7: Geopolitical / uncertainty
            'global_uncertainty':    'GEPUCURRENT',   # Global economic policy uncertainty
            # Factor 8: Technology
            'us_consumer_sentiment': 'UMCSENT',       # Consumer sentiment → spending intent
            # Factor 9: Government policy
            'us_govt_spending':      'FGEXPND',       # US federal expenditure (quarterly)
        }

        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=DATA_START)
                df = pl.DataFrame({
                    'date':  [str(d.date()) for d in s.index],
                    'value': s.values.tolist(),
                    'series': name,
                }).drop_nulls()
                df.write_csv(os.path.join(DATA_DIR, f'fred_{name}.csv'))
                print(f"  OK {name}: {len(df)} rows → fred_{name}.csv")
            except Exception as e:
                print(f"  [Skip] {name}: {e}")
