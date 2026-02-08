# Repository Guidelines

## Project Structure & Module Organization

This repository is organized around research notebooks for options screening and trade analysis.

### Active Notebooks

| Notebook                         | Role             | Description                                                                                                                                       |
| -------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bto_call_put_screener.ipynb`    | Screener + Model | Multi-horizon BTO discovery for calls and puts (short, medium, LEAPS).                                                                            |
| `leaps_discovery_screener.ipynb` | Screener         | Automated US equity market scan, six-factor scoring, watchlist export.                                                                            |
| `leaps_trade_readiness.ipynb`    | Model            | Deep-dive Greeks, risk analysis, IV timing, pre-trade checklist for conviction picks.                                                             |
| `iv_strategy_analysis.ipynb`     | Screener + Model | IV scoring, covered calls, CSPs, LEAP+put combos, Kelly sizing. Set `QUICK_SCAN=True` for scorecard-only mode.                                    |
| `call_fan_discovery.ipynb`       | Screener + Model | Build fans of calls across all expirations (weeklies through LEAPS) per ticker with strike-zone filtering, fan-level scoring, and portfolio plan. |

### Pipeline

`leaps_discovery_screener` (discover) → `leaps_trade_readiness` (deep-dive and execute)

`call_fan_discovery` (build multi-expiration fans for conviction tickers → portfolio plan)

### Supporting Files

- `requirements.txt` — project dependencies.
- `outputs/` — CSV exports and ranking tables (git-ignored).
- `archive/` — retired notebooks (`leaps_screener.ipynb`, `options_iv_screener_mini.ipynb`).

Keep reusable logic in clearly labeled notebook sections (data load, feature engineering, scoring, export), and keep output files out of commits unless explicitly requested.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -r requirements.txt`: install project dependencies (`yfinance`, `pandas`, `numpy`, `plotly`, `openpyxl`).
- `jupyter lab` (or `jupyter notebook`): run notebooks locally.
- `git status`: verify only intended notebook/code changes are staged before committing.

## Coding Style & Naming Conventions

- Follow Python conventions in notebook code: 4-space indentation, `snake_case` names, and small helper functions where practical.
- Use descriptive notebook cell headers (for example: `# Fetch options chain`, `# Rank candidates`).
- Name exports with clear prefixes and timestamps, for example `outputs/bto_call_put_YYYYMMDD_HHMMSS_*.csv`.
- No emojis in notebook headers or display output — use clean, professional formatting with `display(Markdown())` and `display_table()`.
- Use the shared `REPORT_TEMPLATE` (Times New Roman, APA-like) for all Plotly figures.
- Use numbered figure titles (e.g., "Figure 1. Composite Opportunity Score").
- Minimize noisy notebook diffs: clear accidental debug prints and keep execution flow deterministic.

## Testing Guidelines

There is no formal automated test suite yet.

- Before opening a PR, run affected notebooks end-to-end from a clean kernel.
- Confirm key output tables are generated in `outputs/` and spot-check core metrics.
- If you add reusable Python modules in the future, add `pytest` tests alongside them and document run commands.

## Commit & Pull Request Guidelines

- Use imperative commit subjects (`Add ...`, `Refactor ...`, `Remove ...`) and keep each commit focused.
- PRs should include: purpose, notebooks changed, data/output impact, and any assumptions.
- Attach screenshots/plots when visualization or ranking behavior changes.
- Link related issues/tasks and call out breaking workflow changes explicitly.
