# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around research notebooks for options screening.
- Root notebooks: `master_bto_call_put.ipynb`, `leaps_screener.ipynb`, `screen_leaps.ipynb`, `conviction_leaps.ipynb`, and IV analysis notebooks.
- Dependencies: `requirements.txt`.
- Generated artifacts: `outputs/` (CSV exports and ranking tables).
- Local-only files are ignored via `.gitignore` (for example `outputs/`, `*.xlsx`, `*.pyc`).

Keep reusable logic in clearly labeled notebook sections (data load, feature engineering, scoring, export), and keep output files out of commits unless explicitly requested.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -r requirements.txt`: install project dependencies (`yfinance`, `pandas`, `numpy`, `plotly`, `openpyxl`).
- `jupyter lab` (or `jupyter notebook`): run notebooks locally.
- `git status`: verify only intended notebook/code changes are staged before committing.

## Coding Style & Naming Conventions
- Follow Python conventions in notebook code: 4-space indentation, `snake_case` names, and small helper functions where practical.
- Use descriptive notebook cell headers (for example: `# Fetch options chain`, `# Rank candidates`).
- Name exports with clear prefixes and timestamps, matching current patterns such as `outputs/master_bto_YYYYMMDD_HHMMSS_*.csv`.
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
