"""
Data adapter: yfinance calls, rate limiting, sqlite3 IV-history cache.

External deps: yfinance, pandas (data only).
Everything else is stdlib.
"""

import math
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from . import config as cfg

# ---------------------------------------------------------------------------
# Rate-limit helper
# ---------------------------------------------------------------------------

_last_call = 0.0


def _throttle():
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < cfg.RATE_LIMIT_SLEEP:
        time.sleep(cfg.RATE_LIMIT_SLEEP - elapsed)
    _last_call = time.monotonic()


# ---------------------------------------------------------------------------
# SQLite IV cache
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    return Path(__file__).resolve().parent / cfg.CACHE_DB


def _ensure_db():
    db = _db_path()
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE IF NOT EXISTS iv_history ("
        "  ticker TEXT NOT NULL,"
        "  date   TEXT NOT NULL,"
        "  atm_iv REAL NOT NULL,"
        "  PRIMARY KEY (ticker, date)"
        ")"
    )
    con.commit()
    con.close()


def cache_iv(ticker: str, atm_iv: float, as_of: Optional[str] = None):
    _ensure_db()
    as_of = as_of or date.today().isoformat()
    con = sqlite3.connect(str(_db_path()))
    con.execute(
        "INSERT OR REPLACE INTO iv_history (ticker, date, atm_iv) VALUES (?, ?, ?)",
        (ticker, as_of, atm_iv),
    )
    con.commit()
    con.close()


def iv_percentile(ticker: str, current_iv: float) -> Optional[float]:
    """Percentile rank of current_iv vs cached history (0-100)."""
    _ensure_db()
    con = sqlite3.connect(str(_db_path()))
    rows = con.execute(
        "SELECT atm_iv FROM iv_history WHERE ticker = ? ORDER BY date", (ticker,)
    ).fetchall()
    con.close()
    if len(rows) < 5:
        return None
    vals = [r[0] for r in rows]
    below = sum(1 for v in vals if v < current_iv)
    return round(100.0 * below / len(vals), 1)


def cache_stats() -> list[dict]:
    """Return per-ticker row counts in cache."""
    _ensure_db()
    con = sqlite3.connect(str(_db_path()))
    rows = con.execute(
        "SELECT ticker, COUNT(*), MIN(date), MAX(date) FROM iv_history GROUP BY ticker"
    ).fetchall()
    con.close()
    return [
        {"ticker": r[0], "readings": r[1], "first": r[2], "last": r[3]} for r in rows
    ]


def clear_cache():
    _ensure_db()
    con = sqlite3.connect(str(_db_path()))
    con.execute("DELETE FROM iv_history")
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def screen_universe(params: Optional[dict] = None) -> list[str]:
    """Run the yfinance equity screener; return ticker list."""
    p = {**cfg.SCREEN_DEFAULTS, **(params or {})}
    _throttle()
    filters = [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        EquityQuery(
            "btwn", ["intradaymarketcap", p["min_market_cap"], 4_000_000_000_000]
        ),
        EquityQuery("btwn", ["intradayprice", 10, p["max_price"]]),
        EquityQuery("btwn", ["peratio.lasttwelvemonths", 0, p["max_pe"]]),
        EquityQuery(
            "lt", ["lastclosemarketcaptotalrevenue.lasttwelvemonths", p["max_ps"]]
        ),
        EquityQuery("gte", ["returnontotalcapital.lasttwelvemonths", p["min_roe"]]),
        EquityQuery("gte", ["returnonequity.lasttwelvemonths", p["min_roe"]]),
        EquityQuery(
            "gte", ["totalrevenues1yrgrowth.lasttwelvemonths", p["min_rev_growth"]]
        ),
        EquityQuery("gte", ["pctheldinst", p["min_inst_held"]]),
        EquityQuery("gte", ["beta", p["min_beta"]]),
        EquityQuery("is-in", ["sector"] + cfg.SECTORS),
    ]
    q = EquityQuery("and", filters)
    resp = yf.screen(q, size=p["size"], sortField=p["sort_by"], sortAsc=False)
    quotes = []
    if resp:
        if "quotes" in resp:
            quotes = resp["quotes"]
        elif "finance" in resp:
            result = resp.get("finance", {}).get("result", [])
            if result:
                quotes = result[0].get("quotes", [])
    return [r["symbol"] for r in quotes if r.get("symbol")]


# ---------------------------------------------------------------------------
# Per-ticker data fetch
# ---------------------------------------------------------------------------


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def get_spot(ticker: str) -> Optional[float]:
    _throttle()
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_fundamentals(ticker: str) -> dict:
    _throttle()
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "ticker": ticker,
        "name": info.get("shortName", ""),
        "sector": info.get("sector"),
        "market_cap": _safe_float(info.get("marketCap")),
        "pe": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
        "roe": _safe_float(info.get("returnOnEquity")),
        "rev_growth": _safe_float(info.get("revenueGrowth")),
        "profit_margin": _safe_float(info.get("profitMargins")),
        "op_margin": _safe_float(info.get("operatingMargins")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "current_ratio": _safe_float(info.get("currentRatio")),
        "beta": _safe_float(info.get("beta")),
        "inst_held": _safe_float(info.get("heldPercentInstitutions")),
    }


def fetch_events(ticker: str) -> dict:
    """Next earnings date and ex-dividend date."""
    _throttle()
    result = {
        "next_earnings": None,
        "earnings_dte": None,
        "ex_div_date": None,
        "div_dte": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return result

    today = date.today()

    # Earnings
    raw_earn = info.get("earningsTimestamp") or info.get("earningsDate")
    if raw_earn:
        try:
            if isinstance(raw_earn, (list, tuple)):
                raw_earn = raw_earn[0]
            if isinstance(raw_earn, (int, float)):
                earn_dt = datetime.fromtimestamp(raw_earn).date()
            else:
                earn_dt = datetime.strptime(str(raw_earn)[:10], "%Y-%m-%d").date()
            result["next_earnings"] = earn_dt.isoformat()
            result["earnings_dte"] = (earn_dt - today).days
        except Exception:
            pass

    # Ex-dividend
    raw_div = info.get("exDividendDate")
    if raw_div:
        try:
            if isinstance(raw_div, (int, float)):
                div_dt = datetime.fromtimestamp(raw_div).date()
            else:
                div_dt = datetime.strptime(str(raw_div)[:10], "%Y-%m-%d").date()
            result["ex_div_date"] = div_dt.isoformat()
            result["div_dte"] = (div_dt - today).days
        except Exception:
            pass

    return result


def get_expirations(ticker: str) -> list[tuple[str, int]]:
    _throttle()
    try:
        exp_dates = yf.Ticker(ticker).options
        if not exp_dates:
            return []
        today = datetime.now().date()
        out = []
        for s in exp_dates:
            try:
                d = datetime.strptime(s, "%Y-%m-%d").date()
                dte = (d - today).days
                if dte > 0:
                    out.append((s, dte))
            except ValueError:
                continue
        return sorted(out, key=lambda x: x[1])
    except Exception:
        return []


def fetch_chain(ticker: str, exp_str: str):
    """Return (calls_df, puts_df) for a given expiration."""
    _throttle()
    try:
        chain = yf.Ticker(ticker).option_chain(exp_str)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def compute_atm_iv(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float
) -> Optional[float]:
    ivs = []
    for df in [calls, puts]:
        if df.empty or "impliedVolatility" not in df.columns:
            continue
        valid = df[df["impliedVolatility"].notna() & (df["impliedVolatility"] > 0)]
        if valid.empty:
            continue
        idx = (valid["strike"] - spot).abs().idxmin()
        ivs.append(float(valid.loc[idx, "impliedVolatility"]))
    return sum(ivs) / len(ivs) if ivs else None


def compute_hv(ticker: str, window: int = 30) -> Optional[float]:
    _throttle()
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < window:
            return None
        rets = (closes / closes.shift(1)).apply(math.log).dropna()
        return float(rets.iloc[-window:].std() * math.sqrt(252))
    except Exception:
        return None


def compute_trend(ticker: str) -> dict:
    """Return trend_score (0-1) and trend_label."""
    default = {"trend_score": None, "trend_label": "N/A"}
    _throttle()
    try:
        hist = yf.Ticker(ticker).history(period=cfg.TREND_PERIOD)
        if hist.empty or "Close" not in hist.columns:
            return default
        closes = hist["Close"].dropna()
        if len(closes) < cfg.MA_LONG:
            return default
        price = float(closes.iloc[-1])
        ma_s = float(closes.rolling(cfg.MA_SHORT).mean().iloc[-1])
        ma_l = float(closes.rolling(cfg.MA_LONG).mean().iloc[-1])
        ma_s_prev = float(
            closes.rolling(cfg.MA_SHORT).mean().iloc[-(cfg.MA_SLOPE_LOOKBACK + 1)]
        )
        slope = (ma_s - ma_s_prev) / ma_s_prev if ma_s_prev else 0

        flags = [
            1 if price > ma_l else 0,
            1 if ma_s > ma_l else 0,
            1 if slope > 0 else 0,
        ]
        score = sum(flags) / len(flags)
        label = "Up" if score >= 0.67 else ("Down" if score <= 0.33 else "Flat")
        return {"trend_score": score, "trend_label": label}
    except Exception:
        return default


def atm_greeks(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float, dte: int, iv: float
) -> dict:
    """Compute Greeks at the ATM strike using the chain's ATM IV."""
    from . import pricing

    T = dte / 365.0
    r = cfg.RISK_FREE_RATE
    K = spot  # ATM approximation

    # Find nearest listed strike
    for df in [calls]:
        if not df.empty and "strike" in df.columns:
            K = float(df.iloc[(df["strike"] - spot).abs().argsort().iloc[0]]["strike"])
            break

    return {
        "atm_strike": K,
        "call_delta": round(pricing.delta(spot, K, T, r, iv, "call"), 4),
        "put_delta": round(pricing.delta(spot, K, T, r, iv, "put"), 4),
        "gamma": round(pricing.gamma(spot, K, T, r, iv), 6),
        "call_theta": round(pricing.theta(spot, K, T, r, iv, "call"), 4),
        "put_theta": round(pricing.theta(spot, K, T, r, iv, "put"), 4),
        "vega": round(pricing.vega(spot, K, T, r, iv), 4),
        "prob_itm_call": round(pricing.prob_itm(spot, K, T, r, iv, "call"), 4),
        "prob_itm_put": round(pricing.prob_itm(spot, K, T, r, iv, "put"), 4),
    }
