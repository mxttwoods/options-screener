import math
import time
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from options_strategy_finder import screen_for_candidates


@dataclass(frozen=True)
class IVAnalysisConfig:
    target_dtes: tuple[int, ...] = (30, 50, 70)
    max_term_dte: int = 180
    dte_tolerance_days: int = 12
    min_open_interest: int = 50
    min_volume: int = 10
    max_spread_pct: float = 0.20
    min_valid_iv: float = 0.05
    max_valid_iv: float = 5.00
    strike_range_pct: float = 0.20
    history_period: str = "1y"
    rate_limit_sleep: float = 0.25
    risk_free_rate: float = 0.04


def safe_float(value, default: float | None = np.nan) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _next_earnings_date(tkr: yf.Ticker) -> date | None:
    try:
        info = tkr.info or {}
        for key in ("earningsTimestamp", "earningsTimestampStart"):
            ts = info.get(key)
            if ts:
                return datetime.fromtimestamp(int(ts)).date()
    except Exception:
        pass

    try:
        cal = tkr.calendar
        if cal is not None and not cal.empty:
            vals = pd.to_datetime(cal.iloc[:, 0], errors="coerce").dropna()
            if not vals.empty:
                return vals.iloc[0].date()
    except Exception:
        pass

    return None


def _realized_vol(close: pd.Series, window: int) -> float | None:
    if close is None or len(close) < window + 1:
        return np.nan
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return np.nan
    return float(log_ret.iloc[-window:].std() * math.sqrt(252))


def get_spot(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])

        if hasattr(t, "fast_info") and t.fast_info:
            price = t.fast_info.get("lastPrice") or t.fast_info.get(
                "regularMarketPrice"
            )
            if price:
                return float(price)

        info = t.info
        if info:
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if price:
                return float(price)
    except Exception:
        return None
    return None


def get_expirations(ticker: str) -> list[tuple[str, int]]:
    try:
        t = yf.Ticker(ticker)
        exp_dates = t.options
        if not exp_dates:
            return []

        today = datetime.now().date()
        result = []
        for exp_str in exp_dates:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte > 0:
                    result.append((exp_str, dte))
            except ValueError:
                continue
        return sorted(result, key=lambda x: x[1])
    except Exception:
        return []


def pick_expiration(
    expirations: list[tuple[str, int]], target_dte: int, tolerance_days: int
) -> tuple[str, int] | None:
    if not expirations:
        return None
    best = min(expirations, key=lambda x: abs(x[1] - target_dte))
    if abs(best[1] - target_dte) > tolerance_days:
        return None
    return best


def fetch_underlying_metrics(
    tickers: list[str], config: IVAnalysisConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    today = datetime.now().date()

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            time.sleep(config.rate_limit_sleep)
            info = t.info or {}
            hist = t.history(period=config.history_period)
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"].dropna()
            if close.empty:
                continue

            spot = float(close.iloc[-1])
            next_earn = _next_earnings_date(t)
            days_to_earnings = (
                (next_earn - today).days if next_earn is not None else np.nan
            )
            if pd.notna(days_to_earnings) and days_to_earnings < 0:
                days_to_earnings = np.nan
                next_earn = None
            rows.append(
                {
                    "ticker": ticker,
                    "spot": spot,
                    "rv20": _realized_vol(close, 20),
                    "rv30": _realized_vol(close, 30),
                    "rv60": _realized_vol(close, 60),
                    "ret_1m": safe_float(close.iloc[-1] / close.iloc[-22] - 1)
                    if len(close) >= 22
                    else np.nan,
                    "next_earnings_date": next_earn.isoformat() if next_earn else None,
                    "days_to_earnings": days_to_earnings,
                    "market_cap": safe_float(info.get("marketCap")),
                    "beta": safe_float(info.get("beta")),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def fetch_chain(
    ticker: str, expiration: str, spot: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return pd.DataFrame(), pd.DataFrame()

        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        dte = (exp_date - datetime.now().date()).days

        for frame, opt_type in ((calls, "C"), (puts, "P")):
            if frame.empty:
                continue
            frame["ticker"] = ticker
            frame["expiration"] = expiration
            frame["dte"] = dte
            frame["spot"] = spot
            frame["optionType"] = opt_type
            frame["mid"] = (frame["bid"].fillna(0.0) + frame["ask"].fillna(0.0)) / 2.0
            frame.loc[frame["mid"] <= 0, "mid"] = frame["lastPrice"]
            frame["spread"] = frame["ask"].fillna(0.0) - frame["bid"].fillna(0.0)
            frame["spread_pct"] = np.where(
                frame["mid"] > 0, frame["spread"] / frame["mid"], np.nan
            )
            frame["moneyness"] = frame["strike"] / spot
            frame["distance_to_spot"] = (frame["strike"] - spot).abs()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def apply_liquidity_filter(
    frame: pd.DataFrame, config: IVAnalysisConfig
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    filtered = filtered[
        filtered["impliedVolatility"].notna()
        & (filtered["impliedVolatility"] >= config.min_valid_iv)
        & (filtered["impliedVolatility"] <= config.max_valid_iv)
        & filtered["mid"].notna()
        & (filtered["mid"] > 0)
        & (filtered["spread_pct"].fillna(config.max_spread_pct + 1) <= config.max_spread_pct)
        & (
            (filtered["openInterest"].fillna(0) >= config.min_open_interest)
            | (filtered["volume"].fillna(0) >= config.min_volume)
        )
    ]
    return filtered


def _nearest_otm_option(
    frame: pd.DataFrame, spot: float, option_type: str
) -> pd.Series | None:
    if frame.empty:
        return None

    if option_type == "C":
        otm = frame[frame["strike"] >= spot].copy()
    else:
        otm = frame[frame["strike"] <= spot].copy()

    if otm.empty:
        otm = frame.copy()
    if otm.empty:
        return None

    idx = otm["distance_to_spot"].idxmin()
    return otm.loc[idx]


def compute_atm_iv_details(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    config: IVAnalysisConfig,
) -> dict[str, float | str | None]:
    calls_valid = apply_liquidity_filter(calls, config)
    puts_valid = apply_liquidity_filter(puts, config)

    call_row = _nearest_otm_option(calls_valid, spot, "C")
    put_row = _nearest_otm_option(puts_valid, spot, "P")
    if call_row is None and put_row is None:
        return {"atm_iv": None}

    ivs = []
    result: dict[str, float | str | None] = {"atm_iv": None}

    if call_row is not None:
        call_iv = float(call_row["impliedVolatility"])
        ivs.append(call_iv)
        result.update(
            {
                "call_strike": float(call_row["strike"]),
                "call_iv": call_iv,
                "call_mid": float(call_row["mid"]),
                "call_spread_pct": float(call_row["spread_pct"]),
            }
        )

    if put_row is not None:
        put_iv = float(put_row["impliedVolatility"])
        ivs.append(put_iv)
        result.update(
            {
                "put_strike": float(put_row["strike"]),
                "put_iv": put_iv,
                "put_mid": float(put_row["mid"]),
                "put_spread_pct": float(put_row["spread_pct"]),
            }
        )

    if ivs:
        result["atm_iv"] = float(np.mean(ivs))
    return result


def expected_move_pct(atm_iv: float | None, dte: int) -> float | None:
    if atm_iv is None or not np.isfinite(atm_iv) or dte <= 0:
        return np.nan
    return float(atm_iv * math.sqrt(dte / 365.0))


def bs_call_delta(
    spot: float, strike: float, dte: int, iv: float, r: float = 0.04
) -> float | None:
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        return np.nan
    t = dte / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    return float(norm_cdf(d1))


def compute_covered_call_rows(
    calls: pd.DataFrame,
    spot: float,
    dte: int,
    otm_levels: list[float],
    config: IVAnalysisConfig,
) -> list[dict[str, object]]:
    calls_valid = apply_liquidity_filter(calls, config)
    if calls_valid.empty:
        return []

    rows: list[dict[str, object]] = []
    for otm_pct in otm_levels:
        target_strike = spot * (1.0 + otm_pct)
        idx = (calls_valid["strike"] - target_strike).abs().idxmin()
        row = calls_valid.loc[idx]
        actual_strike = float(row["strike"])
        mid_price = float(row["mid"])
        iv = float(row["impliedVolatility"])
        premium_yield = mid_price / spot
        annualized_yield = premium_yield * (365.0 / dte) if dte > 0 else np.nan
        break_even = spot - mid_price
        downside_buffer_pct = mid_price / spot
        max_called_return_pct = ((actual_strike - spot) + mid_price) / spot
        annualized_max_return_pct = (
            max_called_return_pct * (365.0 / dte) if dte > 0 else np.nan
        )
        approx_itm_prob = bs_call_delta(
            spot=spot,
            strike=actual_strike,
            dte=dte,
            iv=iv,
            r=config.risk_free_rate,
        )
        rows.append(
            {
                "target_otm_pct": otm_pct,
                "actual_strike": actual_strike,
                "actual_otm_pct": actual_strike / spot - 1.0,
                "mid_price": mid_price,
                "premium_yield": premium_yield,
                "annualized_yield": annualized_yield,
                "break_even": break_even,
                "downside_buffer_pct": downside_buffer_pct,
                "max_called_return_pct": max_called_return_pct,
                "annualized_max_return_pct": annualized_max_return_pct,
                "approx_itm_prob": approx_itm_prob,
                "spread_pct": float(row["spread_pct"]),
            }
        )
    return rows
