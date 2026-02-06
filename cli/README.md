# Alpha Scan CLI

Equity + options alpha scanner. Screens, scores, ranks, and exports to Excel.

## Install

```
cd /path/to/options-screener
pip install -r requirements.txt
```

## Usage

```
# Full scan (screen universe, score, display)
python -m cli scan

# Specific tickers
python -m cli scan -t AAPL,MSFT,GOOG,AMZN

# Top 10, export to Excel
python -m cli scan --top 10 --excel

# Custom output file
python -m cli scan --excel my_report.xlsx

# Disable trend filter
python -m cli scan --no-trend-filter

# Quiet mode (suppress progress)
python -m cli scan -q --excel
```

## IV Cache

Each scan stores ATM IV readings in a local sqlite database. Over time this
builds an IV history that powers the IV percentile column.

```
# View cache stats
python -m cli cache

# Clear cache
python -m cli cache --clear
```

## Excel Report Sheets

| Sheet        | Contents                                           |
| ------------ | -------------------------------------------------- |
| Rankings     | Composite score, grade, signal, sector, trend      |
| Volatility   | ATM IV, HV, IV/HV ratio, IV percentile, term slope |
| Greeks       | ATM delta, gamma, theta, vega, P(ITM)              |
| Fundamentals | ROE, margins, P/E, D/E, market cap, beta           |
| Events       | Next earnings date, ex-div date, event flags       |
| Methodology  | Weights, thresholds, model assumptions             |

## Configuration

Edit `cli/config.py` to adjust screening parameters, scoring weights,
rate limits, and signal thresholds.

## Architecture

```
cli/
  __main__.py   argparse entry point
  config.py     constants and defaults
  data.py       yfinance adapter + sqlite3 IV cache
  pricing.py    Black-Scholes Greeks (stdlib only)
  engine.py     scan orchestration and scoring
  report.py     Excel report generation (openpyxl)
```

All computation uses the Python standard library. External dependencies
are limited to data retrieval (yfinance, pandas) and Excel output (openpyxl).
