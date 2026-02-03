# Repository Guidelines

## Project Overview
This repository is a Jupyter notebook-based options analysis tool. The main workflow screens large-cap stocks, fetches options chains via `yfinance`, computes ATM implied volatility (IV), and visualizes IV smiles and term structure. The primary logic lives in the notebook; helper functions such as `screen_for_candidates()`, `fetch_chain()`, and `compute_atm_iv()` are defined there.

## Project Structure & Module Organization
- `options_iv_analysis.ipynb`: Primary source of truth for analysis, data fetching, and plotting.
- `.github/copilot-instructions.md`: Project-specific conventions and workflow notes.

## Setup, Build, and Development Commands
There is no build system. Use a Python environment with Jupyter.
- `python -m pip install yfinance pandas numpy matplotlib`: Install core dependencies.
- `jupyter lab` or `jupyter notebook`: Run the notebook locally.

## Coding Style & Naming Conventions
- Use 4-space indentation for Python code in the notebook.
- Prefer `snake_case` for functions and variables (e.g., `compute_atm_iv`).
- Use `UPPER_SNAKE_CASE` for constants (e.g., `TARGET_DTES`).
- Preserve the DataFrame enrichment pattern in `fetch_chain()` (add `ticker`, `dte`, `spot`, `mid`, `moneyness`).
- Respect rate limiting between `yfinance` calls (see `RATE_LIMIT_SLEEP`).

## Testing Guidelines
There is no automated test suite. Validate changes by running a small sample of tickers in the notebook and confirming:
- Spot price retrieval succeeds.
- Option chains load and IV values are non-null and positive.
- Plots render without errors.

## Commit & Pull Request Guidelines
There is no established commit history. Use clear, imperative commit subjects (e.g., “Add term structure plot”). For PRs:
- Describe the analysis changes and data sources.
- Include updated plots or screenshots when visuals change.
- Note any parameter adjustments (e.g., `TARGET_DTES`, `STRIKE_RANGE_PCT`).

## Configuration & Data Notes
- Keep target DTEs and strike ranges in the documented constants section.
- Avoid hardcoding secrets; this repo expects public data via `yfinance`.
- If data retrieval fails, prefer graceful warnings and skip logic over hard failures.
