# Options Screener

Utilities and notebooks to find, monitor, and manage options trades.

## Goal

- Be a profitable options trader by buying stocks below fair value and selling at or near fair value.
- All tickers were pre-screened using Morningstar and CFRA research.

## Notebooks

- `options_monitor_v1_prob.ipynb` — Options position monitor & trading cheat sheet (refresh daily to update prices and analysis).
- `call_fan_discovery_v2.ipynb` — Mix-DTE call discovery and broker-ready buy-list generator.

## Quickstart

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebooks in JupyterLab or Jupyter Notebook.

## Data

- Primary data is pulled with `yfinance`; additional free sources are allowed.
