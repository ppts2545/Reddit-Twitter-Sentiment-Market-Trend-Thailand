# Thailand Economic Analysis — Global Comparison

<p align="center">
  <img src="https://flagcdn.com/w160/th.png" alt="Thailand" width="80"/>
  &nbsp;&nbsp;
  <img src="https://flagcdn.com/w160/us.png" alt="United States" width="80"/>
  &nbsp;&nbsp;
  <img src="https://flagcdn.com/w160/cn.png" alt="China" width="80"/>
  &nbsp;&nbsp;
  <img src="https://flagcdn.com/w160/jp.png" alt="Japan" width="80"/>
  &nbsp;&nbsp;
  <img src="https://flagcdn.com/w160/de.png" alt="Germany" width="80"/>
  &nbsp;&nbsp;
  <img src="https://flagcdn.com/w160/sg.png" alt="Singapore" width="80"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Polars-DataFrame-CD792C?logo=polars&logoColor=white" />
  <img src="https://img.shields.io/badge/World%20Bank-API-009FDA" />
  <img src="https://img.shields.io/badge/IMF-DataMapper-003087" />
  <img src="https://img.shields.io/badge/FRED-St.Louis%20Fed-E31837" />
  <img src="https://img.shields.io/badge/yfinance-Markets-000000?logo=yahoo&logoColor=purple" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## Project Goal

Thailand is often analyzed in isolation, which misses the bigger picture. This project takes a **comparative, multi-country approach** — placing Thailand alongside **major Asian economies, European powerhouses, and global benchmarks** to answer a core question:

> **How does Thailand's economic health compare to its regional peers and global leaders, and what structural factors explain the gaps?**

### Countries Under Analysis

| Flag | Country | Region | Role in Analysis |
|------|---------|--------|-----------------|
| <img src="https://flagcdn.com/w40/th.png" width="28"/> | **Thailand** | Southeast Asia | Primary subject |
| <img src="https://flagcdn.com/w40/us.png" width="28"/> | United States | North America | Global benchmark, capital flow driver |
| <img src="https://flagcdn.com/w40/cn.png" width="28"/> | China | East Asia | Thailand's largest trade partner |
| <img src="https://flagcdn.com/w40/jp.png" width="28"/> | Japan | East Asia | Regional anchor, FDI source |
| <img src="https://flagcdn.com/w40/de.png" width="28"/> | Germany | Europe | Export-led growth model (European comparison) |
| <img src="https://flagcdn.com/w40/sg.png" width="28"/> | Singapore | Southeast Asia | Regional high-income peer |

> **Planned expansion:** South Korea, India, UK, France, Vietnam, and Brazil — to broaden coverage across Europe, South Asia, and emerging markets.

---

## What We Track

This analysis is built around **9 structural economic factors** that together explain unemployment dynamics and overall economic health:

```
Factor 1  — Consumption          (Household spending % of GDP)
Factor 2  — Interest Rate        (Lending rate, broad money M2)
Factor 3  — Inflation            (CPI, year-over-year %)
Factor 4  — Unemployment         (Total, youth, by sector)
Factor 5  — Business Investment  (Gross capital formation, FDI inflows)
Factor 6  — Trade / Exports      (Exports & imports % of GDP, USD/THB)
Factor 7  — Geopolitical Risk    (VIX, gold, global uncertainty index)
Factor 8  — Technology / Innovation (Nasdaq, AI/automation trends)
Factor 9  — Government Policy    (Govt expenditure & debt % of GDP)
```

All indicators are tracked from **2003 to present** — long enough to capture the 2008 Global Financial Crisis, COVID-19, and post-pandemic recovery across all countries.

---

## Data Pipeline

```
World Bank API   ──►  Labor market indicators (unemployment, sector employment)
                       Economic context (GDP growth, inflation, trade openness)

IMF DataMapper   ──►  GDP growth, inflation, unemployment, current account
                       (multi-country, harmonized annual data)

FRED (St. Louis) ──►  High-frequency monthly data: Fed Funds rate, US CPI,
                       Thailand exchange rate, Bangkok property prices,
                       global economic policy uncertainty

yfinance         ──►  Market signals: SET index, USD/THB, VIX, Gold,
                       S&P 500, Crude Oil, US 10-yr Treasury, Nasdaq

Google Trends    ──►  Job search behavior in Thailand (หางาน, สมัครงาน)
                       Global signals: AI, automation, layoffs

Bangkok Post RSS ──►  English economic headlines → VADER sentiment scoring
```

All raw data is saved to `data/raw/` as CSV files for reproducibility.

---

## Project Structure

```
Thailand-Economic-Analysis/
├── src/
│   ├── fetch_data.py          # All data fetching (World Bank, IMF, FRED, yfinance, etc.)
│   ├── preprocess.py          # Data cleaning and normalization
│   └── feature_engineering.py # Feature construction for modeling
├── data/
│   └── raw/                   # Raw CSVs saved from fetch_data.py
├── notebooks/                 # Exploratory analysis and visualizations
├── .env                       # API keys (FRED_API_KEY) — not committed
├── requirement.txt
└── README.md
```

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd Thailand-Economic-Analysis
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt

# Set your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
echo "FRED_API_KEY=your_key_here" > .env

# Fetch all data
python src/fetch_data.py
```

---

## Research Questions

1. **Labor market resilience** — Why does Thailand maintain low headline unemployment even during recessions, and how does this compare to Germany or Japan?
2. **Export dependency** — Thailand's trade/GDP ratio (~120%) dwarfs most peers. Is this a strength or vulnerability?
3. **Digital transition** — How do Google Trends for AI/automation in Thailand compare to Singapore and the US? Is the workforce adapting?
4. **Capital flows & currency** — How does the USD/THB respond to Fed rate changes versus how EUR/USD responds?
5. **Geopolitical exposure** — Which countries are most affected by VIX spikes and trade war news?

---

## Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| [World Bank Open Data](https://data.worldbank.org/) | 200+ countries, annual | Free, no key |
| [IMF DataMapper](https://www.imf.org/external/datamapper/) | 190+ countries, annual | Free, no key |
| [FRED (St. Louis Fed)](https://fred.stlouisfed.org/) | US + global, monthly | Free API key |
| [Yahoo Finance (yfinance)](https://finance.yahoo.com/) | Markets, daily | Free |
| [Google Trends](https://trends.google.com/) | Search interest, weekly | Free (rate-limited) |
| [Bangkok Post RSS](https://www.bangkokpost.com/) | Thai business headlines | Free |
