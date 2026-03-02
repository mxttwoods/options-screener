import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class StrategyFinderConfig:
    budget: float = 25_000.0
    target_horizon_days: int = 180
    min_dte: int = 100
    max_dte: int = 700
    max_dte_gap: int = 600
    min_open_interest: int = 50
    min_volume: int = 10
    max_spread_pct: float = 0.25
    min_delta: float = 0.15
    max_delta: float = 0.85
    min_analyst_upside: float = 0.07
    min_analyst_count: int = 5
    min_rr: float = 1.25
    max_rr: float = 2.50
    max_be_pct: float = 0.20
    max_iv_hv: float = 2.25
    slippage_pct_of_spread: float = 0.35
    risk_free_rate: float = 0.04
    history_period: str = "1y"
    corr_period: str = "6mo"
    rate_limit_sleep: float = 0.20
    max_contracts_per_exp: int = 40
    max_underlyings_to_scan: int = 12
    max_contracts_per_line: int = 4
    max_lines: int = 10
    position_cap_pct: float = 0.15
    ticker_cap_pct: float = 0.20
    target_price_source: str = "blended"


def safe_float(value: Any, default: float | None = np.nan) -> float | None:
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


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def rsi(series: pd.Series, window: int = 14) -> float | None:
    if series is None or len(series) < window + 2:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean().iloc[-1]
    avg_loss = loss.rolling(window).mean().iloc[-1]
    if avg_loss is None or pd.isna(avg_loss):
        return None
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


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


def screen_for_candidates(
    max_price: float = 300.0,
    min_market_cap: float = 2_000_000_000,
    min_roe: float = 0.12,
    min_rev_growth: float = 0.05,
    max_pe: float = 40.0,
    max_ps: float = 10.0,
    min_beta: float = 1.0,
    min_inst_held: float = 0.40,
    size: int = 50,
    sort_by: str = "eodvolume",
) -> list[str]:
    sectors = [
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Technology",
    ]

    q = yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("eq", ["region", "us"]),
            yf.EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
            yf.EquityQuery(
                "btwn", ["intradaymarketcap", min_market_cap, 14_000_000_000_000]
            ),
            yf.EquityQuery("btwn", ["intradayprice", 10, max_price]),
            yf.EquityQuery("btwn", ["peratio.lasttwelvemonths", 0, max_pe]),
            yf.EquityQuery(
                "lt", ["lastclosemarketcaptotalrevenue.lasttwelvemonths", max_ps]
            ),
            yf.EquityQuery("gte", ["returnontotalcapital.lasttwelvemonths", min_roe]),
            yf.EquityQuery("gte", ["returnonequity.lasttwelvemonths", min_roe]),
            yf.EquityQuery(
                "gte", ["totalrevenues1yrgrowth.lasttwelvemonths", min_rev_growth]
            ),
            yf.EquityQuery("gte", ["pctheldinst", min_inst_held]),
            yf.EquityQuery("gte", ["beta", min_beta]),
            yf.EquityQuery("is-in", ["sector"] + sectors),
        ],
    )
    resp = yf.screen(q, size=size, sortField=sort_by, sortAsc=False)
    quotes = []
    if resp:
        if "quotes" in resp:
            quotes = resp.get("quotes", [])
        elif "finance" in resp:
            result = resp.get("finance", {}).get("result", [])
            if result and len(result) > 0:
                quotes = result[0].get("quotes", [])
    return [row.get("symbol") for row in quotes if row.get("symbol")]


def choose_target_price(
    spot: float,
    target_mean: float | None,
    target_high: float | None,
    source: str = "blended",
) -> float | None:
    valid_mean = (
        float(target_mean)
        if target_mean is not None and np.isfinite(target_mean) and target_mean > 0
        else None
    )
    valid_high = (
        float(target_high)
        if target_high is not None and np.isfinite(target_high) and target_high > 0
        else None
    )

    if source == "mean":
        return valid_mean
    if source == "high":
        return valid_high or valid_mean
    if source == "max":
        values = [v for v in (valid_mean, valid_high) if v is not None]
        return max(values) if values else None

    values = [v for v in (valid_mean, valid_high) if v is not None]
    if not values:
        return None

    if valid_mean is None:
        return valid_high
    if valid_high is None:
        return valid_mean

    blended = 0.7 * valid_mean + 0.3 * valid_high
    return max(blended, spot)


def scale_target_to_horizon(
    spot: float, full_target: float | None, horizon_days: int, annual_days: int = 252
) -> float | None:
    if full_target is None or not np.isfinite(full_target) or full_target <= 0 or spot <= 0:
        return None
    if full_target <= spot:
        return float(full_target)
    horizon_scale = min(max(horizon_days, 1) / float(annual_days), 1.0)
    horizon_target = spot + (full_target - spot) * horizon_scale
    return float(max(horizon_target, spot))


def compute_horizon_return_quantile(
    close: pd.Series, horizon_days: int, quantile: float = 0.01
) -> float | None:
    if close is None or len(close) < horizon_days + 10:
        return None
    horizon_returns = close / close.shift(horizon_days) - 1.0
    horizon_returns = horizon_returns.dropna()
    if horizon_returns.empty:
        return None
    return float(horizon_returns.quantile(quantile))


def fetch_underlying_metrics(
    tickers: list[str],
    config: StrategyFinderConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
            log_ret = np.log(close / close.shift(1)).dropna()
            hv_30 = (
                float(log_ret.iloc[-30:].std() * math.sqrt(252))
                if len(log_ret) >= 30
                else np.nan
            )

            ret_1m = (
                float(close.iloc[-1] / close.iloc[-22] - 1)
                if len(close) >= 22
                else np.nan
            )
            ret_3m = (
                float(close.iloc[-1] / close.iloc[-64] - 1)
                if len(close) >= 64
                else np.nan
            )
            ret_6m = (
                float(close.iloc[-1] / close.iloc[-127] - 1)
                if len(close) >= 127
                else np.nan
            )

            target_mean = safe_float(info.get("targetMeanPrice"))
            target_high = safe_float(info.get("targetHighPrice"))
            analyst_target_full = choose_target_price(
                spot=spot,
                target_mean=target_mean,
                target_high=target_high,
                source=config.target_price_source,
            )
            bullish_target = scale_target_to_horizon(
                spot=spot,
                full_target=analyst_target_full,
                horizon_days=config.target_horizon_days,
            )
            analyst_upside_full_pct = (
                float(analyst_target_full / spot - 1.0)
                if analyst_target_full is not None and analyst_target_full > 0 and spot > 0
                else np.nan
            )
            analyst_upside_pct = (
                float(bullish_target / spot - 1.0)
                if bullish_target is not None and bullish_target > 0 and spot > 0
                else np.nan
            )
            analyst_count = safe_float(info.get("numberOfAnalystOpinions"), 0.0) or 0.0
            recommendation_mean = safe_float(info.get("recommendationMean"))
            recommendation_key = info.get("recommendationKey")
            horizon_return_p01 = compute_horizon_return_quantile(
                close, config.target_horizon_days, quantile=0.01
            )
            next_earn = _next_earnings_date(t)

            rows.append(
                {
                    "ticker": ticker,
                    "spot": spot,
                    "beta": safe_float(info.get("beta"), 1.0),
                    "hv_30": hv_30,
                    "rsi_14": rsi(close, 14),
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "target_mean": target_mean,
                    "target_high": target_high,
                    "analyst_target_full": analyst_target_full,
                    "analyst_upside_full_pct": analyst_upside_full_pct,
                    "bullish_target": bullish_target,
                    "analyst_upside_pct": analyst_upside_pct,
                    "analyst_count": float(analyst_count),
                    "recommendation_mean": recommendation_mean,
                    "recommendation_key": recommendation_key,
                    "horizon_return_p01": horizon_return_p01,
                    "next_earnings_date": next_earn.isoformat() if next_earn else None,
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["analyst_score"] = (
        out["analyst_upside_full_pct"].fillna(-1.0).rank(method="dense", pct=True)
    )
    if "recommendation_mean" in out.columns:
        out["rating_score"] = (
            (6.0 - out["recommendation_mean"].fillna(5.0)).clip(lower=0.0) / 5.0
        )
    else:
        out["rating_score"] = 0.0
    out["coverage_score"] = (out["analyst_count"].fillna(0.0) / 20.0).clip(0.0, 1.0)
    out["underlying_score"] = (
        60.0 * out["analyst_score"].fillna(0.0)
        + 25.0 * out["rating_score"].fillna(0.0)
        + 15.0 * out["coverage_score"].fillna(0.0)
    )
    return out.sort_values(
        ["analyst_upside_full_pct", "underlying_score"], ascending=False
    ).reset_index(drop=True)


def select_top_underlyings(
    metrics_df: pd.DataFrame, config: StrategyFinderConfig
) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df

    filtered = metrics_df[
        (metrics_df["bullish_target"].notna())
        & (
            metrics_df["analyst_upside_full_pct"].fillna(-1.0)
            >= config.min_analyst_upside
        )
        & (metrics_df["analyst_count"].fillna(0.0) >= config.min_analyst_count)
    ].copy()
    if filtered.empty:
        return filtered

    return filtered.nlargest(
        config.max_underlyings_to_scan, "analyst_upside_full_pct"
    )


def build_correlation_matrix(
    tickers: list[str], period: str = "6mo", rate_limit_sleep: float = 0.15
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        try:
            t = yf.Ticker(ticker)
            time.sleep(rate_limit_sleep)
            hist = t.history(period=period)
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            close = hist["Close"].dropna()
            if len(close) < 40:
                continue
            series[ticker] = np.log(close / close.shift(1)).dropna()
        except Exception:
            continue
    if len(series) < 2:
        return pd.DataFrame()
    returns = pd.DataFrame(series).dropna()
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr()


def bs_call_greeks(
    spot: float, strike: float, dte: int, iv: float, r: float = 0.04
) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan}
    t = dte / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    delta = norm_cdf(d1)
    gamma = norm_pdf(d1) / (spot * iv * sqrt_t)
    theta = (
        -(spot * norm_pdf(d1) * iv) / (2 * sqrt_t)
        - r * strike * math.exp(-r * t) * norm_cdf(d2)
    ) / 365.0
    vega = spot * norm_pdf(d1) * sqrt_t / 100.0
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }


def pop_log_normal(
    spot: float, threshold_price: float, dte: int, sigma: float, r: float = 0.04
) -> float:
    if dte <= 0 or sigma <= 0 or spot <= 0 or threshold_price <= 0:
        return 0.0
    t = dte / 365.0
    z = (math.log(spot / threshold_price) + (r - 0.5 * sigma * sigma) * t) / (
        sigma * math.sqrt(t)
    )
    return float(norm_cdf(z))


def estimate_entry_fill(mid: float, bid: float, ask: float, slippage_pct: float) -> float:
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and bid > 0 and ask >= bid:
        edge = (ask - bid) * slippage_pct
        return float(min(ask, max(mid, mid + edge)))
    return float(mid)


def derive_strike_range(
    spot: float,
    target_price: float | None,
    manual: tuple[float, float] | None = None,
) -> tuple[float, float, str]:
    if manual is not None:
        return float(manual[0]), float(manual[1]), "manual"

    if target_price is not None and np.isfinite(target_price) and target_price > spot:
        lo = round(spot * 0.90, -1)
        hi = round(max(target_price * 1.05, spot * 1.05), -1)
        if hi <= lo:
            hi = lo + 20
        return float(lo), float(hi), "analyst_target"

    lo = round(spot * 0.90, -1)
    hi = round(spot * 1.15, -1)
    if hi <= lo:
        hi = lo + 20
    return float(lo), float(hi), "moneyness"


def fetch_call_candidates(
    ticker: str,
    spot: float,
    config: StrategyFinderConfig,
    strike_range: tuple[float, float],
) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        time.sleep(config.rate_limit_sleep)
        all_exps = t.options
        if not all_exps:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    lo_strike, hi_strike = strike_range
    today = date.today()
    rows: list[dict[str, Any]] = []

    for exp in all_exps:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < config.min_dte or dte > config.max_dte:
                continue

            time.sleep(config.rate_limit_sleep)
            chain = t.option_chain(exp)
            frame = chain.calls
            if frame is None or frame.empty:
                continue

            work = frame.copy()
            work["mid"] = (work["bid"].fillna(0) + work["ask"].fillna(0)) / 2.0
            work.loc[work["mid"] <= 0, "mid"] = work["lastPrice"]
            work["spread"] = work["ask"].fillna(0) - work["bid"].fillna(0)
            work["spread_pct"] = np.where(
                work["mid"] > 0, work["spread"] / work["mid"], np.nan
            )
            work = work[
                (work["strike"] >= lo_strike)
                & (work["strike"] <= hi_strike)
                & (work["impliedVolatility"] > 0)
                & (work["mid"] > 0)
                & (work["openInterest"].fillna(0) >= config.min_open_interest)
                & (work["volume"].fillna(0) >= config.min_volume)
                & (
                    work["spread_pct"].fillna(config.max_spread_pct + 1)
                    <= config.max_spread_pct
                )
            ]
            if work.empty:
                continue

            work = work.sort_values(
                ["openInterest", "volume", "mid"], ascending=[False, False, True]
            ).head(config.max_contracts_per_exp)
            for _, row in work.iterrows():
                rows.append(
                    {
                        "ticker": ticker,
                        "expiration": exp,
                        "dte": int(dte),
                        "dte_gap": abs(int(dte) - config.target_horizon_days),
                        "contract": row.get("contractSymbol"),
                        "strike": safe_float(row.get("strike"), np.nan),
                        "bid": safe_float(row.get("bid"), np.nan),
                        "ask": safe_float(row.get("ask"), np.nan),
                        "mid": safe_float(row.get("mid"), np.nan),
                        "iv": safe_float(row.get("impliedVolatility"), np.nan),
                        "oi": safe_float(row.get("openInterest"), 0.0),
                        "volume": safe_float(row.get("volume"), 0.0),
                        "spread_pct": safe_float(row.get("spread_pct"), np.nan),
                        "spot": float(spot),
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(rows)


def _liquidity_quality(
    oi: float,
    volume: float,
    spread_pct: float,
    max_spread_pct: float,
) -> float:
    return float(
        0.45 * min(oi / 1000.0, 1.0)
        + 0.35 * min(volume / 500.0, 1.0)
        + 0.20
        * max(
            1.0 - (spread_pct / max_spread_pct if max_spread_pct > 0 else 1.0),
            0.0,
        )
    )


def score_call_candidates_for_target(
    chain_df: pd.DataFrame,
    metrics_row: pd.Series,
    config: StrategyFinderConfig,
) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    spot = float(metrics_row["spot"])
    target_price = safe_float(metrics_row.get("bullish_target"))
    analyst_upside_pct = safe_float(metrics_row.get("analyst_upside_pct"))
    analyst_upside_full_pct = safe_float(metrics_row.get("analyst_upside_full_pct"))
    analyst_target_full = safe_float(metrics_row.get("analyst_target_full"))
    analyst_count = safe_float(metrics_row.get("analyst_count"), 0.0) or 0.0
    recommendation_mean = safe_float(metrics_row.get("recommendation_mean"))
    recommendation_key = metrics_row.get("recommendation_key")
    hv = (
        float(metrics_row["hv_30"])
        if pd.notna(metrics_row["hv_30"]) and float(metrics_row["hv_30"]) > 0
        else 0.28
    )
    rsi_14 = float(metrics_row["rsi_14"]) if pd.notna(metrics_row["rsi_14"]) else 50.0
    ret_3m = float(metrics_row["ret_3m"]) if pd.notna(metrics_row["ret_3m"]) else 0.0
    next_earn = metrics_row.get("next_earnings_date")
    earn_date = (
        datetime.fromisoformat(next_earn).date() if isinstance(next_earn, str) else None
    )
    horizon_return_p01 = safe_float(metrics_row.get("horizon_return_p01"))

    if (
        target_price is None
        or not np.isfinite(target_price)
        or target_price <= spot
        or analyst_upside_pct is None
        or not np.isfinite(analyst_upside_pct)
    ):
        return pd.DataFrame()

    scored_rows: list[dict[str, Any]] = []
    rr_mid = (config.min_rr + config.max_rr) / 2.0
    rr_half = max((config.max_rr - config.min_rr) / 2.0, 0.10)

    for _, option_row in chain_df.iterrows():
        dte = int(option_row["dte"])
        strike = float(option_row["strike"])
        iv = float(option_row["iv"])
        oi = float(option_row["oi"])
        volume = float(option_row["volume"])
        spread_pct = (
            float(option_row["spread_pct"])
            if pd.notna(option_row["spread_pct"])
            else np.nan
        )
        entry = estimate_entry_fill(
            float(option_row["mid"]),
            float(option_row["bid"]),
            float(option_row["ask"]),
            config.slippage_pct_of_spread,
        )
        if not np.isfinite(entry) or entry <= 0:
            continue

        greeks = bs_call_greeks(spot=spot, strike=strike, dte=dte, iv=iv)
        delta = greeks["delta"]
        theta = greeks["theta"]
        gamma = greeks["gamma"]
        vega = greeks["vega"]
        if not np.isfinite(delta):
            continue

        breakeven = strike + entry
        be_pct = breakeven / spot - 1.0
        simulation_sigma = float(np.nanmean([iv, hv])) if np.isfinite(hv) else iv
        pop = pop_log_normal(
            spot, breakeven, dte, sigma=max(simulation_sigma, 0.01), r=config.risk_free_rate
        )
        target_prob = pop_log_normal(
            spot,
            target_price,
            dte,
            sigma=max(simulation_sigma, 0.01),
            r=config.risk_free_rate,
        )

        target_intrinsic = max(target_price - strike, 0.0)
        target_profit_per_share = target_intrinsic - entry
        target_profit_per_contract = target_profit_per_share * 100.0
        rr_target = target_profit_per_share / entry
        target_value_multiple = target_intrinsic / entry if entry > 0 else np.nan
        iv_hv = iv / hv if hv > 0 else np.nan
        theta_ratio = abs(theta) / entry if entry > 0 and np.isfinite(theta) else np.nan
        liq_quality = _liquidity_quality(
            oi=oi,
            volume=volume,
            spread_pct=spread_pct if np.isfinite(spread_pct) else config.max_spread_pct,
            max_spread_pct=config.max_spread_pct,
        )
        dte_fit = max(
            1.0 - abs(dte - config.target_horizon_days) / max(config.max_dte_gap, 1),
            0.0,
        )
        rr_fit = max(1.0 - abs(rr_target - rr_mid) / rr_half, 0.0)
        analyst_strength = min(
            analyst_upside_pct / max(config.min_analyst_upside, 0.01), 2.0
        ) / 2.0
        coverage_strength = min(analyst_count / 20.0, 1.0)
        rating_strength = (
            (6.0 - recommendation_mean) / 5.0
            if recommendation_mean is not None and np.isfinite(recommendation_mean)
            else 0.4
        )
        spread_penalty = (
            max(spread_pct / config.max_spread_pct - 1.0, 0.0)
            if np.isfinite(spread_pct) and config.max_spread_pct > 0
            else 1.0
        )

        earnings_risk = False
        if earn_date is not None:
            days_to_earn = (earn_date - date.today()).days
            earnings_risk = 0 <= days_to_earn <= dte

        score = (
            40.0 * rr_fit
            + 18.0 * analyst_strength
            + 14.0 * liq_quality
            + 10.0 * target_prob
            + 8.0 * dte_fit
            + 6.0 * coverage_strength
            + 4.0 * rating_strength
            - 12.0 * spread_penalty
            - (8.0 if earnings_risk else 0.0)
        )

        gate_reasons: list[str] = []
        if delta < config.min_delta or delta > config.max_delta:
            gate_reasons.append("delta")
        if abs(dte - config.target_horizon_days) > config.max_dte_gap:
            gate_reasons.append("dte")
        if analyst_upside_pct < config.min_analyst_upside:
            gate_reasons.append("target")
        if rr_target < config.min_rr or rr_target > config.max_rr:
            gate_reasons.append("rr")
        if be_pct > config.max_be_pct:
            gate_reasons.append("be")
        if iv_hv > config.max_iv_hv:
            gate_reasons.append("iv_hv")
        if not np.isfinite(target_profit_per_contract) or target_profit_per_contract <= 0:
            gate_reasons.append("payoff")

        if gate_reasons:
            continue

        flags: list[str] = []
        if earnings_risk:
            flags.append("earnings")
        if rsi_14 > 75:
            flags.append("rsi_hot")
        if ret_3m < -0.10:
            flags.append("downtrend")
        if target_prob < 0.25:
            flags.append("low_target_prob")

        scored_rows.append(
            {
                "ticker": option_row["ticker"],
                "contract": option_row["contract"],
                "expiration": option_row["expiration"],
                "dte": dte,
                "dte_gap": abs(dte - config.target_horizon_days),
                "strike": strike,
                "spot": spot,
                "entry_fill": entry,
                "cost_1ct_exec": entry * 100.0,
                "bid": float(option_row["bid"]),
                "ask": float(option_row["ask"]),
                "mid": float(option_row["mid"]),
                "iv": iv,
                "hv": hv,
                "iv_hv": iv_hv,
                "oi": oi,
                "volume": volume,
                "spread_pct": spread_pct,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "theta_ratio": theta_ratio,
                "breakeven": breakeven,
                "be_pct": be_pct,
                "pop": pop,
                "target_prob": target_prob,
                "target_price": target_price,
                "target_upside_pct": analyst_upside_pct,
                "analyst_target_full": analyst_target_full,
                "analyst_upside_full_pct": analyst_upside_full_pct,
                "target_intrinsic": target_intrinsic,
                "target_profit_per_contract": target_profit_per_contract,
                "target_rr": rr_target,
                "target_value_multiple": target_value_multiple,
                "analyst_count": analyst_count,
                "recommendation_mean": recommendation_mean,
                "recommendation_key": recommendation_key,
                "liq_quality": liq_quality,
                "score": score,
                "horizon_return_p01": horizon_return_p01,
                "underlying_drawdown_99_pct": horizon_return_p01,
                "flags": ", ".join(flags) if flags else "--",
            }
        )

    if not scored_rows:
        return pd.DataFrame()

    out = pd.DataFrame(scored_rows)
    out["rr_gap"] = (out["target_rr"] - rr_mid).abs()
    out["signal_confidence"] = (
        100.0
        * (
            0.40 * out["liq_quality"].clip(0.0, 1.0)
            + 0.25 * out["target_prob"].clip(0.0, 1.0)
            + 0.20 * (1.0 - (out["rr_gap"] / max(rr_half, 0.10)).clip(0.0, 1.0))
            + 0.15
            * np.minimum(
                out["target_upside_pct"] / max(config.min_analyst_upside, 0.01), 1.0
            )
        )
    ).clip(0.0, 100.0)
    return out.sort_values(
        ["score", "target_upside_pct", "target_rr", "liq_quality"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def build_candidate_pool(
    metrics_df: pd.DataFrame,
    config: StrategyFinderConfig,
    manual_zones: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_zones = manual_zones or {}
    overview_rows: list[dict[str, Any]] = []
    pool_parts: list[pd.DataFrame] = []

    for _, metric_row in metrics_df.iterrows():
        ticker = metric_row["ticker"]
        spot = float(metric_row["spot"])
        target_price = safe_float(metric_row.get("bullish_target"))
        lo, hi, zone_source = derive_strike_range(
            spot=spot,
            target_price=target_price,
            manual=manual_zones.get(ticker),
        )

        chain = fetch_call_candidates(
            ticker=ticker,
            spot=spot,
            config=config,
            strike_range=(lo, hi),
        )
        scored = score_call_candidates_for_target(
            chain_df=chain,
            metrics_row=metric_row,
            config=config,
        )

        if not scored.empty:
            best_per_exp = (
                scored.sort_values(
                    ["score", "target_rr", "liq_quality"],
                    ascending=[False, True, False],
                )
                .groupby("expiration", as_index=False)
                .head(1)
                .reset_index(drop=True)
            )
            pool_parts.append(best_per_exp)

        overview_rows.append(
            {
                "ticker": ticker,
                "spot": spot,
                "bullish_target": target_price,
                "analyst_upside_pct": metric_row.get("analyst_upside_pct"),
                "analyst_target_full": metric_row.get("analyst_target_full"),
                "analyst_upside_full_pct": metric_row.get("analyst_upside_full_pct"),
                "analyst_count": metric_row.get("analyst_count"),
                "recommendation_mean": metric_row.get("recommendation_mean"),
                "recommendation_key": metric_row.get("recommendation_key"),
                "horizon_return_p01": metric_row.get("horizon_return_p01"),
                "next_earnings": metric_row.get("next_earnings_date"),
                "zone": f"${lo:,.0f}-${hi:,.0f}",
                "zone_source": zone_source,
                "contracts_fetched": len(chain),
                "contracts_scored": len(scored),
            }
        )

    overview_df = pd.DataFrame(overview_rows).sort_values(
        ["analyst_upside_full_pct", "contracts_scored"], ascending=[False, False]
    )
    pool_df = (
        pd.concat(pool_parts, ignore_index=True)
        if pool_parts
        else pd.DataFrame()
    )
    if not pool_df.empty:
        pool_df = pool_df.sort_values(
            ["score", "target_rr", "liq_quality"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
    return overview_df, pool_df


def optimize_buy_list(
    pool_df: pd.DataFrame, config: StrategyFinderConfig
) -> tuple[pd.DataFrame, float]:
    if pool_df.empty:
        return pd.DataFrame(), config.budget

    candidates = pool_df.sort_values(
        ["score", "target_upside_pct", "target_rr", "liq_quality"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    selected: list[pd.Series] = []
    budget_left = float(config.budget)
    spent_by_ticker: dict[str, float] = {}

    for _, row in candidates.iterrows():
        cost = float(row["cost_1ct_exec"])
        if not np.isfinite(cost) or cost <= 0:
            continue

        ticker = row["ticker"]
        pos_cap = config.budget * config.position_cap_pct
        tkr_cap = config.budget * config.ticker_cap_pct

        qty_by_pos = int(pos_cap // cost)
        qty_by_ticker = int(
            max(tkr_cap - spent_by_ticker.get(ticker, 0.0), 0.0) // cost
        )
        qty_by_budget = int(budget_left // cost)
        qty = min(
            qty_by_pos,
            qty_by_ticker,
            qty_by_budget,
            config.max_contracts_per_line,
        )
        if qty <= 0:
            continue

        line = row.copy()
        line["qty"] = int(qty)
        line["line_cost"] = float(qty * cost)
        line["target_profit_line"] = float(qty * row["target_profit_per_contract"])
        line["max_loss"] = float(qty * cost)
        selected.append(line)

        spent_by_ticker[ticker] = spent_by_ticker.get(ticker, 0.0) + float(qty * cost)
        budget_left -= float(qty * cost)

        if len(selected) >= config.max_lines:
            break

    if not selected:
        return pd.DataFrame(), float(config.budget)

    buy_df = pd.DataFrame(selected).reset_index(drop=True)
    buy_df.insert(0, "#", range(1, len(buy_df) + 1))
    total_deployed = float(buy_df["line_cost"].sum())
    buy_df["portfolio_weight"] = (
        buy_df["line_cost"] / total_deployed if total_deployed > 0 else 0.0
    )
    buy_df["target_return_pct"] = (
        buy_df["target_profit_per_contract"] / buy_df["cost_1ct_exec"]
    )
    return buy_df, float(budget_left)
