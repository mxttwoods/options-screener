"""
Configuration constants for the alpha-scan CLI.

Single source of truth for screening parameters, scoring weights,
rate limits, and output formatting.
"""

# ---------------------------------------------------------------------------
# Universe screening defaults
# ---------------------------------------------------------------------------
SCREEN_DEFAULTS = dict(
    max_price=300.0,
    min_market_cap=2_000_000_000,
    min_roe=0.12,
    min_rev_growth=0.05,
    max_pe=40.0,
    max_ps=10.0,
    min_beta=1.0,
    min_inst_held=0.40,
    size=40,
    sort_by="eodvolume",
)

SECTORS = [
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Technology",
]

# ---------------------------------------------------------------------------
# Options / volatility
# ---------------------------------------------------------------------------
TARGET_DTE = 45
MAX_TERM_DTE = 120
TERM_STRUCTURE_SAMPLE = 2  # sample every Nth expiration for term structure
HV_WINDOW = 30  # calendar days for realized-vol window
RISK_FREE_RATE = 0.045  # annualized; update to match current T-bill

# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------
TREND_FILTER = True
MIN_TREND_SCORE = 0.34
MA_SHORT = 50
MA_LONG = 200
MA_SLOPE_LOOKBACK = 20
TREND_PERIOD = "1y"

# ---------------------------------------------------------------------------
# Rate limiting (seconds between yfinance calls)
# ---------------------------------------------------------------------------
RATE_LIMIT_SLEEP = 0.30

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
FUNDAMENTAL_WEIGHT = 0.40
IV_WEIGHT = 0.60

FUND_WEIGHTS = {
    "roe": 0.25,
    "rev_growth": 0.25,
    "profit_margin": 0.20,
    "debt_to_equity": 0.15,
    "pe": 0.15,
}

IV_WEIGHTS = {
    "atm_iv": 0.30,
    "iv_hv_ratio": 0.25,
    "term_slope": 0.20,
    "iv_pctl": 0.25,
}

# ---------------------------------------------------------------------------
# Earnings / event-risk buffer (days)
# ---------------------------------------------------------------------------
EARNINGS_BUFFER_DAYS = 7  # flag if earnings within this many DTE

# ---------------------------------------------------------------------------
# Output defaults
# ---------------------------------------------------------------------------
TOP_N = 15
MAX_TICKERS = 25

# ---------------------------------------------------------------------------
# Grade thresholds (score -> letter)
# ---------------------------------------------------------------------------
GRADE_THRESHOLDS = [
    (90, "A+"),
    (80, "A"),
    (70, "B"),
    (60, "C"),
    (50, "D"),
    (0, "F"),
]

# ---------------------------------------------------------------------------
# Signal logic (IV/HV thresholds)
# ---------------------------------------------------------------------------
SIGNAL_SELL_RICH = 1.30  # IV/HV above -> "Sell premium"
SIGNAL_BUY_CHEAP = 0.80  # IV/HV below -> "Buy vol"
SIGNAL_VERY_RICH = 1.50  # IV/HV above -> "IV very rich"

# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------
CACHE_DB = "iv_cache.db"
