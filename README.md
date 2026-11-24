# Market Breadth (NSE)

End-of-day market-breadth dashboard for NSE stocks above ₹1,000 cr market cap, inspired by
the SOIC *Market Regime Note* and the 52-week-high / RS-rating tools on markets.manasarora.com.

**What it shows:** % of stocks above their 20/50/200-day EMAs, stocks near 52-week highs/lows,
washouts and breadth thrusts, a regime scorecard, FII/DII flows, index strength and P/E vs
10-year history, industry strength scores, new 52-week highs, and a 1–99 RS rating.

## How it updates (free)
- `.github/workflows/daily.yml` runs Mon–Fri ~7:35 pm IST (retry 9:35 pm).
- `scripts/update.py` downloads NSE's daily bhavcopy, index-close file (with P/E) and FII/DII
  provisional figures; detects split/bonus adjustments from NSE's adjusted previous close.
- `scripts/compute.py` recomputes everything into `public/data/*.json`.
- Vercel serves `public/` and redeploys on every commit.

## Data
- `data/prices/YYYY.parquet` – split/bonus-adjusted daily OHLCV for all NSE symbols and indices
  since 1995 (seeded from the public eod2_data archive, built from NSE reports).
- `data/daily/` – raw NSE files added by the daily job.
- `data/universe.csv` – stock list, industries, market cap, P/E from StockScans (refreshed weekly).
- `data/flows.csv` – FII/DII net cash-market figures, accumulated daily.

Educational use only. Not investment advice.
