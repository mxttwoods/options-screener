"""Reusable helpers for options research notebooks.

The functions here are deliberately defensive around missing market data so
notebooks can run end-to-end with partial inputs.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf

try:
    from yfinance import EquityQuery
except Exception:  # pragma: no cover - fallback for older yfinance
    EquityQuery = None


DEFAULT_CURATED_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSM",
    "AMD",
    "QCOM",
    "JPM",
    "V",
    "MA",
    "LLY",
    "UNH",
    "GE",
    "CAT",
    "DE",
    "KLAC",
    "WDC",
]


def setup_report_style(renderer: str | None = None) -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)

    chosen_renderer = renderer or os.getenv("PLOTLY_RENDERER", "notebook_connected")
    pio.renderers.default = chosen_renderer

    report_template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Times New Roman", size=14, color="#111827"),
            title=dict(font=dict(size=20)),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                linecolor="#111827",
                mirror=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                linecolor="#111827",
                mirror=True,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=60, r=30, t=70, b=50),
        )
    )
    pio.templates["report"] = report_template
    pio.templates.default = "report"


def display_table(
    df: pd.DataFrame, caption: str = "", format_dict: dict[str, Any] | None = None
):
    from IPython.display import display

    styler = df.style
    if format_dict:
        styler = styler.format(format_dict, na_rep="--")
    if caption:
        styler = styler.set_caption(caption)
    display(styler)


def parse_env_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


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


def _parse_screen_quotes(response: Any) -> list[str]:
    quotes: list[dict[str, Any]] = []
    if isinstance(response, dict):
        if "quotes" in response:
            quotes = response.get("quotes", []) or []
        else:
            result = (response.get("finance", {}) or {}).get("result", []) or []
            if result:
                quotes = result[0].get("quotes", []) or []
    return [
        row.get("symbol")
        for row in quotes
        if isinstance(row, dict) and row.get("symbol")
    ]


def screen_universe(
    *,
    use_screen: bool,
    ticker_override: Iterable[str] | None,
    max_tickers: int,
    size: int = 60,
) -> list[str]:
    override = [t for t in (ticker_override or []) if t]
    if override:
        return list(dict.fromkeys(override))[:max_tickers]

    tickers: list[str] = []
    if use_screen and EquityQuery is not None:
        try:
            sectors = [
                "Communication Services",
                "Consumer Cyclical",
                "Consumer Defensive",
                "Financial Services",
                "Healthcare",
                "Industrials",
                "Technology",
            ]
            filters = [
                EquityQuery("eq", ["region", "us"]),
                EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                EquityQuery(
                    "btwn", ["intradaymarketcap", 2_000_000_000, 4_000_000_000_000]
                ),
                EquityQuery("btwn", ["intradayprice", 10, 500]),
                EquityQuery("gte", ["pctheldinst", 0.35]),
                EquityQuery("is-in", ["sector"] + sectors),
            ]
            query = EquityQuery("and", filters)
            response = yf.screen(query, size=size, sortField="eodvolume", sortAsc=False)
            tickers = _parse_screen_quotes(response)
        except Exception:
            tickers = []

    if not tickers:
        tickers = DEFAULT_CURATED_UNIVERSE.copy()

    return list(dict.fromkeys(tickers))[:max_tickers]


def fetch_underlying_metrics(
    tickers: Iterable[str],
    *,
    history_period: str = "1y",
    rate_limit_sleep: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            time.sleep(rate_limit_sleep)

            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}

            hist = t.history(period=history_period)
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"].dropna()
            if close.empty:
                continue

            spot = float(close.iloc[-1])
            log_returns = np.log(close / close.shift(1)).dropna()

            hv_30 = (
                float(log_returns.iloc[-30:].std() * math.sqrt(252))
                if len(log_returns) >= 30
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
            rsi_14 = rsi(close, 14)

            rows.append(
                {
                    "ticker": ticker,
                    "spot": spot,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": safe_float(info.get("marketCap")),
                    "beta": safe_float(info.get("beta")),
                    "pe": safe_float(info.get("forwardPE") or info.get("trailingPE")),
                    "ps": safe_float(info.get("priceToSalesTrailing12Months")),
                    "roe": safe_float(info.get("returnOnEquity")),
                    "rev_growth": safe_float(info.get("revenueGrowth")),
                    "profit_margin": safe_float(info.get("profitMargins")),
                    "avg_volume_3m": safe_float(
                        info.get("averageVolume")
                        or info.get("averageDailyVolume3Month")
                    ),
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "rsi_14": rsi_14,
                    "hv_30": hv_30,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def _expirations_for_horizon(
    ticker: str,
    *,
    horizon_cfg: dict[str, int],
    max_exp_per_horizon: int,
    rate_limit_sleep: float,
) -> list[tuple[str, int]]:
    try:
        t = yf.Ticker(ticker)
        time.sleep(rate_limit_sleep)
        exps = t.options
        if not exps:
            return []
    except Exception:
        return []

    today = date.today()
    pairs: list[tuple[str, int]] = []
    for exp in exps:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if horizon_cfg["min_dte"] <= dte <= horizon_cfg["max_dte"]:
                pairs.append((exp, dte))
        except Exception:
            continue

    target = horizon_cfg["target_dte"]
    pairs = sorted(pairs, key=lambda x: abs(x[1] - target))
    return pairs[:max_exp_per_horizon]


def fetch_option_candidates(
    ticker: str,
    *,
    side: str,
    spot: float,
    horizons: dict[str, dict[str, int]],
    moneyness_bounds: dict[str, tuple[float, float]],
    max_exp_per_horizon: int = 2,
    max_contracts_per_exp: int = 60,
    min_open_interest: int = 25,
    min_volume: int = 5,
    max_spread_pct: float = 0.50,
    rate_limit_sleep: float = 0.2,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    t = yf.Ticker(ticker)

    for horizon, cfg in horizons.items():
        expirations = _expirations_for_horizon(
            ticker,
            horizon_cfg=cfg,
            max_exp_per_horizon=max_exp_per_horizon,
            rate_limit_sleep=rate_limit_sleep,
        )
        if not expirations:
            continue

        mon_low, mon_high = moneyness_bounds[horizon]

        for exp_str, dte in expirations:
            try:
                time.sleep(rate_limit_sleep)
                chain = t.option_chain(exp_str)
                frame = chain.calls if side == "call" else chain.puts
                if frame is None or frame.empty:
                    continue

                work = frame.copy()
                work["mid"] = (work["bid"].fillna(0) + work["ask"].fillna(0)) / 2.0
                work.loc[work["mid"] <= 0, "mid"] = work["lastPrice"]
                work["spread"] = work["ask"].fillna(0) - work["bid"].fillna(0)
                work["spread_pct"] = np.where(
                    work["mid"] > 0, work["spread"] / work["mid"], np.nan
                )
                work["moneyness"] = work["strike"] / spot

                work = work[
                    (work["strike"] > 0)
                    & (work["mid"] > 0)
                    & (work["impliedVolatility"] > 0)
                    & (work["openInterest"].fillna(0) >= min_open_interest)
                    & (work["volume"].fillna(0) >= min_volume)
                    & (work["spread_pct"].fillna(max_spread_pct + 1) <= max_spread_pct)
                    & (work["moneyness"] >= mon_low)
                    & (work["moneyness"] <= mon_high)
                ]

                if work.empty:
                    continue

                work = work.sort_values(
                    ["openInterest", "volume", "mid"], ascending=[False, False, True]
                ).head(max_contracts_per_exp)

                for _, row in work.iterrows():
                    all_rows.append(
                        {
                            "ticker": ticker,
                            "side": side,
                            "horizon": horizon,
                            "expiration": exp_str,
                            "dte": int(dte),
                            "contract_symbol": row.get("contractSymbol"),
                            "strike": safe_float(row.get("strike"), 0.0),
                            "mid": safe_float(row.get("mid"), np.nan),
                            "bid": safe_float(row.get("bid"), np.nan),
                            "ask": safe_float(row.get("ask"), np.nan),
                            "iv": safe_float(row.get("impliedVolatility"), np.nan),
                            "open_interest": safe_float(row.get("openInterest"), 0.0),
                            "volume": safe_float(row.get("volume"), 0.0),
                            "spread_pct": safe_float(row.get("spread_pct"), np.nan),
                            "moneyness": safe_float(row.get("moneyness"), np.nan),
                            "spot": spot,
                        }
                    )
            except Exception:
                continue

    return pd.DataFrame(all_rows)


def score_option_candidates(
    candidates: pd.DataFrame, underlyings: pd.DataFrame
) -> pd.DataFrame:
    if candidates.empty or underlyings.empty:
        return pd.DataFrame()

    merged = candidates.merge(
        underlyings[
            [
                "ticker",
                "sector",
                "spot",
                "hv_30",
                "ret_3m",
                "rsi_14",
                "beta",
                "avg_volume_3m",
                "pe",
                "roe",
                "rev_growth",
                "profit_margin",
            ]
        ],
        on=["ticker", "spot"],
        how="left",
    )

    merged["hv_30"] = merged["hv_30"].fillna(0.28)
    merged["ret_3m"] = merged["ret_3m"].fillna(0.0)
    merged["rsi_14"] = merged["rsi_14"].fillna(50.0)
    merged["iv_hv_ratio"] = np.where(
        merged["hv_30"] > 0, merged["iv"] / merged["hv_30"], np.nan
    )

    expected_moves = merged["hv_30"] * np.sqrt(np.maximum(merged["dte"], 1) / 365.0)

    call_target = merged["spot"] * (
        1.0 + np.maximum(merged["ret_3m"], 0.01) + 0.35 * expected_moves
    )
    put_target = merged["spot"] * (
        1.0 - np.maximum(-merged["ret_3m"], 0.01) - 0.35 * expected_moves
    )

    call_pnl = np.maximum(call_target - merged["strike"], 0) - merged["mid"]
    put_pnl = np.maximum(merged["strike"] - put_target, 0) - merged["mid"]

    merged["expected_pnl"] = np.where(merged["side"] == "call", call_pnl, put_pnl)
    merged["expected_return"] = merged["expected_pnl"] / merged["mid"]

    merged["iv_value_score"] = np.clip(
        (1.5 - merged["iv_hv_ratio"].fillna(1.5)) / 1.5 * 100.0, 0, 100
    )

    liq_oi = np.clip(merged["open_interest"] / 600.0 * 40.0, 0, 40)
    liq_vol = np.clip(merged["volume"] / 200.0 * 30.0, 0, 30)
    liq_spread = np.clip(
        (0.45 - merged["spread_pct"].fillna(0.45)) / 0.45 * 30.0, 0, 30
    )
    merged["liquidity_score"] = liq_oi + liq_vol + liq_spread

    trend_call = np.clip(
        55 + merged["ret_3m"] * 250 - (merged["rsi_14"] - 58).abs() * 0.9, 0, 100
    )
    trend_put = np.clip(
        55 + (-merged["ret_3m"]) * 250 - (merged["rsi_14"] - 42).abs() * 0.9, 0, 100
    )
    merged["trend_score"] = np.where(merged["side"] == "call", trend_call, trend_put)

    merged["return_score"] = np.clip(
        (merged["expected_return"] + 0.4) / 1.8 * 100.0, 0, 100
    )

    merged["fundamental_score"] = np.clip(
        50
        + merged["roe"].fillna(0.10) * 150
        + merged["rev_growth"].fillna(0.05) * 120
        + merged["profit_margin"].fillna(0.08) * 80
        - merged["pe"].fillna(20).clip(0, 45) * 0.8,
        0,
        100,
    )

    merged["master_score"] = (
        0.30 * merged["return_score"]
        + 0.22 * merged["iv_value_score"]
        + 0.20 * merged["liquidity_score"]
        + 0.18 * merged["trend_score"]
        + 0.10 * merged["fundamental_score"]
    )

    merged["master_rank"] = merged.groupby(["side", "horizon"])["master_score"].rank(
        method="min", ascending=False
    )
    return merged.sort_values(
        ["master_score", "expected_return"], ascending=False
    ).reset_index(drop=True)


def top_by_bucket(
    scored: pd.DataFrame,
    *,
    bucket_cols: list[str],
    top_n: int,
    sort_cols: list[str] | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    by = sort_cols or ["master_score", "expected_return"]
    out = (
        scored.sort_values(by, ascending=False)
        .groupby(bucket_cols, as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def build_correlation_matrix(
    tickers: Iterable[str],
    *,
    period: str = "6mo",
    rate_limit_sleep: float = 0.15,
) -> pd.DataFrame:
    series = {}
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
    spot: float,
    strike: float,
    dte: int,
    iv: float,
    r: float = 0.04,
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


def scenario_pnl_call(
    spot: float, strike: float, premium: float, moves: dict[str, float]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, move in moves.items():
        terminal = spot * (1.0 + move)
        out[label] = max(terminal - strike, 0.0) - premium
    return out
