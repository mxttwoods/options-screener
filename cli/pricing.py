"""
Black-Scholes pricing and Greeks.

All functions use the Python standard library only (math, statistics).
No scipy or numpy required.
"""

import math
from statistics import NormalDist

_N = NormalDist().cdf  # cumulative normal
_n = NormalDist().pdf  # normal density


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """European option price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return _N(d1)
    return _N(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return _n(d1) / (S * sigma * math.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Per-calendar-day theta (divide annual by 365)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    common = -(S * _n(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "call":
        annual = common - r * K * math.exp(-r * T) * _N(d2)
    else:
        annual = common + r * K * math.exp(-r * T) * _N(-d2)
    return annual / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1-point (1.00) move in vol.  Divide by 100 for per-1% move."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * _n(d1) * math.sqrt(T)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


def prob_itm(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Risk-neutral probability of finishing in-the-money."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return 1.0 if S < K else 0.0
    d2 = _d2(S, K, T, r, sigma)
    if option_type == "call":
        return _N(d2)
    return _N(-d2)
