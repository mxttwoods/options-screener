"""
Alpha-scan engine: orchestrates data fetch, scoring, and ranking.

Produces a single DataFrame with all metrics, grades, and signals.
"""

import sys
from datetime import date
from typing import Optional

import pandas as pd

from . import config as cfg
from . import data


# ---------------------------------------------------------------------------
# Scoring helpers (no numpy -- pure pandas)
# ---------------------------------------------------------------------------


def _rank_pct(series: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    r = s.rank(pct=True)
    if not higher_better:
        r = 1.0 - r
    return r.fillna(0.5)


def _assign_grade(score) -> str:
    if score is None or pd.isna(score):
        return "N/A"
    for threshold, label in cfg.GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "F"


def _assign_signal(iv_hv: Optional[float], trend: str) -> str:
    """One-line actionable signal derived from IV richness and trend."""
    if iv_hv is None:
        return "--"
    if iv_hv >= cfg.SIGNAL_VERY_RICH:
        return "IV rich -- sell premium"
    if iv_hv >= cfg.SIGNAL_SELL_RICH and trend == "Up":
        return "Sell CC / CSP"
    if iv_hv >= cfg.SIGNAL_SELL_RICH:
        return "Elevated IV -- hedge or sell"
    if iv_hv <= cfg.SIGNAL_BUY_CHEAP and trend == "Up":
        return "IV cheap -- buy calls / LEAPS"
    if iv_hv <= cfg.SIGNAL_BUY_CHEAP:
        return "IV cheap -- debit spreads"
    return "Neutral"


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------


def run_scan(
    tickers: Optional[list[str]] = None,
    top_n: int = cfg.TOP_N,
    max_tickers: int = cfg.MAX_TICKERS,
    trend_filter: bool = cfg.TREND_FILTER,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Execute the full alpha scan pipeline.

    Returns a scored and ranked DataFrame ready for terminal display or
    Excel export.
    """

    # 1. Universe -----------------------------------------------------------
    if tickers:
        universe = [t.strip().upper() for t in tickers]
    else:
        if not quiet:
            _log("Screening equity universe ...")
        universe = data.screen_universe()

    universe = universe[:max_tickers]
    if not quiet:
        _log(f"Universe: {len(universe)} tickers")

    # 2. Per-ticker data collection -----------------------------------------
    rows = []
    skipped = []

    for i, ticker in enumerate(universe):
        if not quiet and (i + 1) % 5 == 0:
            _log(f"  [{i + 1}/{len(universe)}] {ticker}")

        # Spot
        spot = data.get_spot(ticker)
        if spot is None:
            skipped.append((ticker, "no spot"))
            continue

        # Fundamentals
        fund = data.fetch_fundamentals(ticker)

        # Trend
        trend = data.compute_trend(ticker)
        if trend_filter and (trend["trend_score"] or 0) < cfg.MIN_TREND_SCORE:
            skipped.append((ticker, "trend filter"))
            continue

        # Events
        events = data.fetch_events(ticker)

        # Options expirations
        expirations = data.get_expirations(ticker)
        if not expirations:
            skipped.append((ticker, "no options"))
            continue

        # Target expiration
        target = min(expirations, key=lambda x: abs(x[1] - cfg.TARGET_DTE))
        exp_date, dte = target

        # Chain + ATM IV
        calls, puts = data.fetch_chain(ticker, exp_date)
        atm_iv = data.compute_atm_iv(calls, puts, spot)
        if atm_iv is None:
            skipped.append((ticker, "no IV"))
            continue

        # HV
        hv = data.compute_hv(ticker, cfg.HV_WINDOW)
        iv_hv = atm_iv / hv if hv and hv > 0 else None

        # Term structure (sample a few expirations)
        term_ivs = []
        term_exps = [e for e in expirations if e[1] <= cfg.MAX_TERM_DTE]
        term_exps = term_exps[:: cfg.TERM_STRUCTURE_SAMPLE][:4]
        for te, td in term_exps:
            tc, tp = data.fetch_chain(ticker, te)
            tiv = data.compute_atm_iv(tc, tp, spot)
            if tiv is not None:
                term_ivs.append({"dte": td, "iv": tiv})

        term_slope = None
        if len(term_ivs) >= 2:
            near, far = term_ivs[0], term_ivs[-1]
            denom = far["dte"] - near["dte"]
            if denom > 0:
                term_slope = (far["iv"] - near["iv"]) / denom

        # Greeks at ATM
        greeks = data.atm_greeks(calls, puts, spot, dte, atm_iv)

        # Cache IV reading
        data.cache_iv(ticker, atm_iv)
        iv_pctl = data.iv_percentile(ticker, atm_iv)

        # Event flag
        earn_dte = events.get("earnings_dte")
        event_flag = ""
        if earn_dte is not None and 0 < earn_dte <= cfg.EARNINGS_BUFFER_DAYS:
            event_flag = f"EARN {earn_dte}d"

        # Signal
        signal = _assign_signal(iv_hv, trend["trend_label"])

        rows.append(
            {
                # Identity
                "ticker": ticker,
                "name": fund.get("name", ""),
                "sector": fund.get("sector"),
                "spot": spot,
                # Fundamentals
                "market_cap": fund.get("market_cap"),
                "pe": fund.get("pe"),
                "roe": fund.get("roe"),
                "rev_growth": fund.get("rev_growth"),
                "profit_margin": fund.get("profit_margin"),
                "op_margin": fund.get("op_margin"),
                "debt_to_equity": fund.get("debt_to_equity"),
                "current_ratio": fund.get("current_ratio"),
                "beta": fund.get("beta"),
                # Trend
                "trend_score": trend["trend_score"],
                "trend": trend["trend_label"],
                # Volatility
                "dte": dte,
                "expiration": exp_date,
                "atm_iv": atm_iv,
                "hv_30": hv,
                "iv_hv": iv_hv,
                "iv_pctl": iv_pctl,
                "term_slope": term_slope,
                # Greeks (ATM)
                "atm_strike": greeks["atm_strike"],
                "call_delta": greeks["call_delta"],
                "put_delta": greeks["put_delta"],
                "gamma": greeks["gamma"],
                "call_theta": greeks["call_theta"],
                "put_theta": greeks["put_theta"],
                "vega": greeks["vega"],
                "prob_itm_call": greeks["prob_itm_call"],
                # Events
                "next_earnings": events.get("next_earnings"),
                "earnings_dte": events.get("earnings_dte"),
                "ex_div_date": events.get("ex_div_date"),
                "event_flag": event_flag,
                # Signal
                "signal": signal,
            }
        )

    if not quiet:
        _log(f"Collected {len(rows)} candidates, skipped {len(skipped)}")

    if not rows:
        return pd.DataFrame()

    # 3. Build DataFrame and score ------------------------------------------
    df = pd.DataFrame(rows)

    # Fundamental composite
    fund_score = (
        _rank_pct(df["roe"], True) * cfg.FUND_WEIGHTS["roe"]
        + _rank_pct(df["rev_growth"], True) * cfg.FUND_WEIGHTS["rev_growth"]
        + _rank_pct(df["profit_margin"], True) * cfg.FUND_WEIGHTS["profit_margin"]
        + _rank_pct(df["debt_to_equity"], False) * cfg.FUND_WEIGHTS["debt_to_equity"]
        + _rank_pct(df["pe"], False) * cfg.FUND_WEIGHTS["pe"]
    )
    df["fund_score"] = (fund_score * 100).round(1)

    # IV composite
    iv_score = (
        _rank_pct(df["atm_iv"], True) * cfg.IV_WEIGHTS["atm_iv"]
        + _rank_pct(df["iv_hv"], True) * cfg.IV_WEIGHTS["iv_hv_ratio"]
        + _rank_pct(df["term_slope"], False) * cfg.IV_WEIGHTS["term_slope"]
        + _rank_pct(df["iv_pctl"], True) * cfg.IV_WEIGHTS["iv_pctl"]
    )
    df["iv_score"] = (iv_score * 100).round(1)

    # Composite
    df["score"] = (
        cfg.FUNDAMENTAL_WEIGHT * df["fund_score"] + cfg.IV_WEIGHT * df["iv_score"]
    ).round(1)

    df["grade"] = df["score"].apply(_assign_grade)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    if top_n:
        df = df.head(top_n)

    return df


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def print_report(df: pd.DataFrame):
    """Compact terminal report."""
    if df.empty:
        print("No results.")
        return

    run_date = date.today().isoformat()
    print(f"\n{'=' * 80}")
    print(f"  ALPHA SCAN  |  {run_date}  |  {len(df)} candidates")
    print(f"{'=' * 80}\n")

    # Rankings table
    cols = [
        "ticker",
        "grade",
        "score",
        "signal",
        "spot",
        "atm_iv",
        "hv_30",
        "iv_hv",
        "iv_pctl",
        "call_delta",
        "call_theta",
        "event_flag",
    ]
    display_df = df[cols].copy()

    # Format for terminal
    display_df["spot"] = display_df["spot"].map(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "--"
    )
    display_df["atm_iv"] = display_df["atm_iv"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "--"
    )
    display_df["hv_30"] = display_df["hv_30"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "--"
    )
    display_df["iv_hv"] = display_df["iv_hv"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "--"
    )
    display_df["iv_pctl"] = display_df["iv_pctl"].map(
        lambda x: f"{x:.0f}" if pd.notna(x) else "--"
    )
    display_df["call_delta"] = display_df["call_delta"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "--"
    )
    display_df["call_theta"] = display_df["call_theta"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "--"
    )
    display_df["score"] = display_df["score"].map(lambda x: f"{x:.1f}")

    display_df.columns = [
        "TICKER",
        "GRD",
        "SCORE",
        "SIGNAL",
        "SPOT",
        "IV",
        "HV30",
        "IV/HV",
        "IV%",
        "DELTA",
        "THETA",
        "EVENT",
    ]

    print(display_df.to_string(index=False))
    print(f"\n{'─' * 80}")
    print("  IV%  = IV percentile vs history  |  SIGNAL = directional bias")
    print("  GRD  = composite grade (A+ best) |  EVENT  = earnings within buffer")
    print(f"{'─' * 80}\n")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str):
    print(f"  {msg}", file=sys.stderr, flush=True)
