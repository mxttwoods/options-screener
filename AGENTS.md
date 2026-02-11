# AGENTS.md

## Project Overview

Options Trading Research & Screening Platform built with Jupyter Notebooks and Python.

**Primary Notebook**: `call_fan_discovery_v2.ipynb` - Consolidated options screener for call/put analysis

**Core Technologies**:

- Python 3.x with virtual environment (`.venv/`)
- Jupyter Lab for notebook execution
- Data: pandas, numpy, yfinance
- Visualization: plotly, matplotlib, seaborn
- Analysis: scipy for statistical functions

## Build & Setup Commands

```bash
# Setup virtual environment (if not exists)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab

# Format Python code (ad-hoc usage)
autopep8 --in-place --aggressive <file.py>
```

## Development Workflow

### Running the Project

1. Activate virtual environment: `source .venv/bin/activate`
2. Launch Jupyter Lab: `jupyter lab`
3. Open `call_fan_discovery_v2.ipynb`
4. Run all cells from clean kernel (Kernel → Restart Kernel and Run All Cells)

### Testing Changes

**No automated test suite** - Manual testing required:

- Before opening PR: Run complete notebook end-to-end from clean kernel
- Verify all visualizations render correctly
- Check that data exports work (CSV files with timestamps)
- Validate yfinance data fetching works for ticker symbols

### Code Quality

- **Linting**: Use autopep8 for Python formatting (no strict linting enforced)
- **Notebooks**: Ensure deterministic execution flow
- **Dependencies**: All packages listed in requirements.txt

## Code Style Guidelines

### Python Code

- **Indentation**: 4 spaces
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Line length**: Keep under 100 characters where practical
- **Imports**: Group by standard library, third-party, local (with blank lines)

```python
# Good
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Bad
import pandas, numpy
from datetime import *
```

### Jupyter Notebook Guidelines

- **Cell Headers**: Use markdown cells with clear, descriptive headers
- **Execution Order**: Notebooks must run top-to-bottom without errors
- **Output**: Commit notebooks with output cleared to minimize diffs
- **Parameters**: Define key parameters in early cells for easy modification
- **Documentation**: Use markdown cells to explain analysis steps and methodology

```python
# Parameter cell (early in notebook)
TICKER = "AAPL"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
```

### Data Handling

- **Caching**: Cache yfinance data when possible to avoid repeated API calls
- **Error Handling**: Wrap API calls in try/except blocks
- **Validation**: Check DataFrame shapes and column names after data loading
- **Exports**: Save results to `outputs/` directory with timestamps

### Visualization Standards

- **Template**: Use Times New Roman fonts for publication-ready plots
- **Consistency**: Apply consistent color schemes across visualizations
- **Labels**: Always include clear titles, axis labels, and legends
- **Sizing**: Ensure plots are appropriately sized for readability

```python
# Example visualization template
fig = px.line(df, x='date', y='value', title='Options Analysis')
fig.update_layout(
    font_family="Times New Roman",
    font_size=12,
    title_font_size=14
)
```

## Project Structure

**Single Notebook Architecture**:

- `call_fan_discovery_v2.ipynb` - Main analysis notebook (1960 lines)
- `export-tickers.js` - MarketBeat data extraction utility
- `requirements.txt` - Python dependencies (119 packages)
- `outputs/` - Generated CSV files (gitignored)
- `.venv/` - Virtual environment (gitignored)

**Historical Context**: Project evolved from multiple notebooks to single consolidated notebook for simplicity.

## Naming Conventions

- **Notebooks**: Use descriptive names with version suffixes (`*_v2.ipynb`)
- **Variables**: `snake_case` (e.g., `option_chain`, `implied_volatility`)
- **Functions**: `snake_case` with action verbs (e.g., `fetch_option_data()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `API_ENDPOINT`, `DEFAULT_TICKER`)
- **Exported Files**: Include timestamp and descriptive name (`bto_call_put_20260211_143022_analysis.csv`)

## Error Handling Patterns

```python
# For API calls
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
except Exception as e:
    print(f"Error fetching data for {ticker}: {e}")
    continue

# For DataFrame operations
if df is not None and not df.empty:
    result = df.calculate_some_metric()
else:
    print("Warning: Empty DataFrame, skipping calculation")
```

## Git Workflow

- **Commits**: Keep commits focused on single changes
- **Notebooks**: Clear outputs before committing: `Cell → All Output → Clear`
- **Requirements**: Update requirements.txt when adding new packages
- **Ignore**: Never commit `.venv/`, `outputs/`, `.xlsx`, or `.ipynb_checkpoints/`

## Performance Considerations

- **Data Fetching**: Batch yfinance requests when possible
- **Processing**: Use vectorized pandas/numpy operations instead of loops
- **Memory**: Clear large DataFrames when no longer needed (`del large_df`)
- **Visualization**: Limit data points in plots for performance

## Additional Notes

- **No CI/CD**: Manual testing and deployment only
- **No Type Checking**: Python is dynamically typed in this project
- **Research-Oriented**: Code prioritizes exploration and analysis over production robustness
- **Dependencies**: Project uses 119+ packages - be mindful of version conflicts when adding new ones
