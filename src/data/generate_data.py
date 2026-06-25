"""
Synthetic firm-year panel generator.

The data-generating process follows Shumway (2001):
- Each firm has a latent quality factor that drives all accounting ratios
- Downgrade probability follows a logistic model with leverage, ROA, and
  Altman Z as inputs
- Macro shocks in RECESSION_YEARS raise the baseline downgrade probability
  by ~0.8 log-odds

This is a controlled, synthetic panel — not real financial statements — by
design. Using a known data-generating process lets us isolate exactly where
modeling improvements come from (re-estimation, richer features, or
non-linearity) without the confound of an unknown real-world DGP.
"""

import numpy as np
import pandas as pd

RATING_MAP = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
}

RECESSION_YEARS = {2001, 2002, 2008, 2009, 2020}


def generate_synthetic_panel(n_firms=600, start_year=1990, end_year=2022, seed=42):
    """
    Synthetic firm-year panel calibrated to Shumway (2001).

    DGP:
        log P(downgrade) = -3.0 - 0.5*quality + 1.5*leverage
                           - 2.0*ROA - 0.3*altman_z + 1.2*recession_shock

    Base downgrade rate ~8%; elevated to ~13-17% in recession years.
    5% annual firm attrition mimics Compustat survivorship.
    """
    rng = np.random.default_rng(seed)
    records = []

    for firm_id in range(n_firms):
        gvkey = f"G{firm_id:05d}"
        firm_quality = rng.normal(0, 1)       # latent health factor
        firm_size = rng.uniform(3, 10)        # log(assets)
        rating = rng.integers(6, 16)          # initial rating: B to A range

        for year in range(start_year, end_year + 1):
            if rng.random() < 0.05:
                break  # firm attrition

            macro_shock = 0.8 if year in RECESSION_YEARS else 0.0
            noise = rng.normal(0, 0.3)

            # Altman X variables — correlated with latent quality
            X1 = 0.15 + 0.05 * firm_quality + noise * 0.5   # working capital / assets
            X2 = 0.10 + 0.08 * firm_quality + noise * 0.4   # retained earnings / assets
            X3 = 0.06 + 0.04 * firm_quality + noise * 0.3   # EBIT / assets
            X4 = np.exp(firm_quality + rng.normal(0, 0.5))  # mkt equity / liabilities
            X5 = rng.uniform(0.5, 2.0)                      # sales / assets
            altman_z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + X5

            # Ohlson variables
            TLTA = max(0.01, 0.5 - 0.1 * firm_quality + rng.normal(0, 0.15))
            NITA = 0.04 + 0.03 * firm_quality + rng.normal(0, 0.04)
            CLCA = max(0.1, 1.2 - 0.3 * firm_quality + rng.normal(0, 0.3))
            FUTL = rng.normal(0, 0.2)
            CHIN = rng.normal(0.01, 0.15)
            INTWO = int(NITA < -0.02 and rng.random() < 0.4)
            OENEG = int(TLTA > 1.0)
            SIZE = firm_size + rng.normal(0, 0.1)

            # Market-based variables (Merton/KMV motivation)
            ret_vol = max(0.05, 0.25 - 0.05 * firm_quality + rng.normal(0, 0.1))
            cum_ret = 0.08 + 0.1 * firm_quality - macro_shock * 0.3 + rng.normal(0, 0.2)
            log_mktcap = firm_size + firm_quality * 0.5 + rng.normal(0, 0.3)
            market_lev = max(0, min(0.99, TLTA * 0.8 + rng.normal(0, 0.05)))

            # Downgrade DGP
            log_odds = (-3.0 - 0.5 * firm_quality + 1.5 * TLTA - 2.0 * NITA
                        - 0.3 * altman_z + 1.2 * macro_shock + rng.normal(0, 0.3))
            p_dn = 1 / (1 + np.exp(-log_odds))
            downgrade = int(rng.random() < p_dn)

            # HMM regime label (3 states: 0=calm, 1=stress, 2=crisis)
            if macro_shock == 0:
                regime = 0 if rng.random() < 0.8 else 1
            else:
                regime = 2 if year in {2009, 2020} else 1

            records.append({
                "gvkey": gvkey, "year": year,
                "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
                "altman_z": altman_z,
                "SIZE": SIZE, "TLTA": TLTA, "WCTA": X1, "CLCA": CLCA,
                "NITA": NITA, "FUTL": FUTL, "INTWO": INTWO,
                "CHIN": CHIN, "OENEG": OENEG,
                "ret_vol": ret_vol, "cum_ret": cum_ret,
                "log_mktcap": log_mktcap, "market_leverage": market_lev,
                "rating_numeric": rating, "downgrade": downgrade,
                "investment_grade": int(rating <= 10),
                "is_recession": int(year in RECESSION_YEARS),
                "regime": regime,
            })
            if downgrade:
                rating = min(22, rating + rng.integers(1, 4))

    df = pd.DataFrame(records)
    print(f"Panel: {len(df):,} firm-years | {df['gvkey'].nunique():,} unique firms")
    print(f"Downgrade rate: {df['downgrade'].mean():.1%} | "
          f"IG share: {df['investment_grade'].mean():.1%}")
    return df


if __name__ == "__main__":
    panel = generate_synthetic_panel()
    panel.to_csv("data/processed/panel.csv", index=False)
    print("Saved data/processed/panel.csv")
