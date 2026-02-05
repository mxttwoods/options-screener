import numpy as np, pandas as pd, math

# Bonds transcribed from your screenshot (edit/add rows as needed)
bonds = [
    {
        "id": "META 5.625 2055-11-15",
        "issuer": "Meta",
        "coupon_pct": 5.625,
        "ytm_pct": 5.95,
        "ttm_years": 29 + 285 / 365,
    },
    {
        "id": "ORCL 5.95 2055-09-26",
        "issuer": "Oracle",
        "coupon_pct": 5.95,
        "ytm_pct": 6.62,
        "ttm_years": 29 + 234 / 365,
    },
    {
        "id": "SWMA 4.0 2028-05-31",
        "issuer": "Swedish Match",
        "coupon_pct": 4.0,
        "ytm_pct": 4.30,
        "ttm_years": 2 + 117 / 365,
    },
    {
        "id": "ORCL 6.0 2055-08-03",
        "issuer": "Oracle",
        "coupon_pct": 6.0,
        "ytm_pct": 6.68,
        "ttm_years": 29 + 180 / 365,
    },
    {
        "id": "ORCL 5.875 2045-09-26",
        "issuer": "Oracle",
        "coupon_pct": 5.875,
        "ytm_pct": 6.53,
        "ttm_years": 19 + 234 / 365,
    },
    {
        "id": "MRK 5.55 2055-12-04",
        "issuer": "Merck",
        "coupon_pct": 5.55,
        "ytm_pct": 5.63,
        "ttm_years": 29 + 304 / 365,
    },
    {
        "id": "JEF 6.2 2034-04-14",
        "issuer": "Jefferies",
        "coupon_pct": 6.2,
        "ytm_pct": 5.39,
        "ttm_years": 8 + 69 / 365,
    },
    {
        "id": "XAI 12.5 2030-06-30",
        "issuer": "xAI",
        "coupon_pct": 12.5,
        "ytm_pct": 8.76,
        "ttm_years": 4 + 146 / 365,
    },
    {
        "id": "META 5.5 2045-11-15",
        "issuer": "Meta",
        "coupon_pct": 5.5,
        "ytm_pct": 5.78,
        "ttm_years": 19 + 285 / 365,
    },
    {
        "id": "ORCL 5.2 2035-09-26",
        "issuer": "Oracle",
        "coupon_pct": 5.2,
        "ytm_pct": 5.61,
        "ttm_years": 9 + 234 / 365,
    },
    {
        "id": "ORCL 5.375 2054-09-27",
        "issuer": "Oracle",
        "coupon_pct": 5.375,
        "ytm_pct": 6.64,
        "ttm_years": 28 + 235 / 365,
    },
    {
        "id": "PAYX 5.35 2032-04-15",
        "issuer": "Paychex",
        "coupon_pct": 5.35,
        "ytm_pct": 4.79,
        "ttm_years": 6 + 71 / 365,
    },
    {
        "id": "APO 5.15 2035-08-12",
        "issuer": "Apollo",
        "coupon_pct": 5.15,
        "ytm_pct": 5.23,
        "ttm_years": 9 + 189 / 365,
    },
    {
        "id": "META 4.875 2035-11-15",
        "issuer": "Meta",
        "coupon_pct": 4.875,
        "ytm_pct": 5.02,
        "ttm_years": 9 + 285 / 365,
    },
    {
        "id": "ORCL 2.95 2030-04-01",
        "issuer": "Oracle",
        "coupon_pct": 2.95,
        "ytm_pct": 4.85,
        "ttm_years": 4 + 56 / 365,
    },
    {
        "id": "BOWL ~6 2031-01-23",
        "issuer": "Blue Owl",
        "coupon_pct": 6.0,
        "ytm_pct": 6.47,
        "ttm_years": 4 + 354 / 365,
    },
]

# -------- Assumptions (EDIT THESE) --------
HORIZON = 5.0
REINVEST = 0.04  # reinvest rate for bonds that mature before 5y

sp_arith_mu = 0.08  # assumed S&P expected annual return
sp_vol = 0.18  # assumed S&P annual volatility

sigma_y_annual = 0.0075  # assumed annual yield volatility for bonds you sell at year 5
# -----------------------------------------

FREQ = 2
FACE = 100.0


def bond_price(face, coupon_pct, ytm_pct, maturity_years, freq=2):
    if maturity_years <= 0:
        return face
    n = max(1, int(round(maturity_years * freq)))
    c = face * (coupon_pct / 100) / freq
    r = (ytm_pct / 100) / freq
    t = np.arange(1, n + 1)
    return float(np.sum(c / ((1 + r) ** t)) + face / ((1 + r) ** n))


def mod_duration(face, coupon_pct, ytm_pct, maturity_years, freq=2):
    if maturity_years <= 0:
        return 0.0
    n = max(1, int(round(maturity_years * freq)))
    c = face * (coupon_pct / 100) / freq
    r = (ytm_pct / 100) / freq
    t = np.arange(1, n + 1)
    cf = np.full(n, c, dtype=float)
    cf[-1] += face
    pv = cf / ((1 + r) ** t)
    price = pv.sum()
    macaulay = (t / freq * pv).sum() / price
    return float(macaulay / (1 + r))


def bond_hpr(b, horizon_years=5.0, reinvest_rate=0.04, delta_y=0.0):
    y0, cpn, T0 = b["ytm_pct"], b["coupon_pct"], b["ttm_years"]
    P0 = bond_price(FACE, cpn, y0, T0, FREQ)

    if T0 <= horizon_years:
        n_total = max(1, int(round(T0 * FREQ)))
        coupon_cash = (FACE * (cpn / 100) / FREQ) * n_total
        proceeds = coupon_cash + FACE
        rem = horizon_years - T0
        end_value = proceeds * ((1 + reinvest_rate) ** rem)
        return end_value / P0 - 1.0
    else:
        n_hold = max(1, int(round(horizon_years * FREQ)))
        coupon_cash = (FACE * (cpn / 100) / FREQ) * n_hold
        y1 = max(0.0, y0 + delta_y)
        P1 = bond_price(FACE, cpn, y1, T0 - horizon_years, FREQ)
        return (coupon_cash + P1) / P0 - 1.0


# Monte Carlo
N = 80_000
rng = np.random.default_rng(42)

mu_log = math.log(1 + sp_arith_mu) - 0.5 * sp_vol**2
z = rng.standard_normal(N)
sp_gross = np.exp(mu_log * HORIZON + sp_vol * math.sqrt(HORIZON) * z)
sp_ret = sp_gross - 1.0

sigma_y_5y = sigma_y_annual * math.sqrt(HORIZON)

rows = []
for b in bonds:
    P0 = bond_price(FACE, b["coupon_pct"], b["ytm_pct"], b["ttm_years"], FREQ)
    dur = mod_duration(FACE, b["coupon_pct"], b["ytm_pct"], b["ttm_years"], FREQ)

    if b["ttm_years"] <= HORIZON:
        bond_ret = np.full(N, bond_hpr(b, HORIZON, REINVEST, 0.0))
    else:
        dy = rng.normal(0.0, sigma_y_5y, size=N) * 100  # percentage points
        bond_ret = np.array([bond_hpr(b, HORIZON, REINVEST, d) for d in dy])

    bond_ann = (1.0 + bond_ret) ** (1 / HORIZON) - 1.0
    p_win = float(np.mean(bond_ret > sp_ret))

    rows.append(
        {
            "bond": b["id"],
            "issuer": b["issuer"],
            "coupon%": b["coupon_pct"],
            "ytm%": b["ytm_pct"],
            "ttm_years": round(b["ttm_years"], 2),
            "price_per_100": round(P0, 2),
            "mod_dur_yrs": round(dur, 1),
            "bond_ann_median%": round(float(np.median(bond_ann)) * 100, 2),
            "bond_ann_10_90%": f"{np.percentile(bond_ann, 10) * 100:.2f}%–{np.percentile(bond_ann, 90) * 100:.2f}%",
            "P(bond>S&P)": round(p_win, 3),
        }
    )

df = pd.DataFrame(rows).sort_values("P(bond>S&P)", ascending=False)
print(df.to_string(index=False))
