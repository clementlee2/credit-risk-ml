"""
Feature families used across all models.

- ALTMAN_FEATURES: the five ratios in Altman's (1968) original Z-score
- OHLSON_FEATURES: the nine accounting/distress-flag variables in
  Ohlson's (1980) O-score
- MARKET_FEATURES: market-based variables added in Models 3 and 4,
  motivated by the Merton/KMV distance-to-default literature

Comparing models that use successively larger feature sets isolates how
much of any modeling improvement is coming from richer inputs rather than
a more flexible functional form.
"""

import pandas as pd

ALTMAN_FEATURES = ["X1", "X2", "X3", "X4", "X5"]
OHLSON_FEATURES = ["SIZE", "TLTA", "WCTA", "CLCA", "NITA", "FUTL", "INTWO", "CHIN", "OENEG"]
MARKET_FEATURES = ["ret_vol", "cum_ret", "log_mktcap", "market_leverage"]
ALL_FEATURES = ALTMAN_FEATURES + OHLSON_FEATURES + MARKET_FEATURES

FEATURE_LABELS = {
    "X1": "Working capital / assets", "X2": "Retained earnings / assets",
    "X3": "EBIT / assets", "X4": "Mkt equity / liabilities",
    "X5": "Sales / assets", "SIZE": "Log total assets",
    "TLTA": "Total liabilities / assets", "WCTA": "Working capital / assets",
    "CLCA": "Current liab / current assets", "NITA": "Net income / assets (ROA)",
    "FUTL": "Earnings trend", "INTWO": "Two consecutive loss years",
    "CHIN": "Change in net income ratio", "OENEG": "Liabilities > assets",
    "ret_vol": "Equity return volatility", "cum_ret": "12-month equity return",
    "log_mktcap": "Log market cap", "market_leverage": "Market leverage ratio",
}


def winsorize(panel: pd.DataFrame, features=ALL_FEATURES, lower=0.01, upper=0.99) -> pd.DataFrame:
    """Clip each feature to its [lower, upper] quantile range, in place-safe copy."""
    panel = panel.copy()
    for col in features:
        if col in panel.columns:
            lo, hi = panel[col].quantile([lower, upper])
            panel[col] = panel[col].clip(lo, hi)
    return panel


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply winsorization and return the feature-ready panel."""
    return winsorize(panel)


if __name__ == "__main__":
    panel = pd.read_csv("data/processed/panel.csv")
    panel = build_features(panel)
    panel.to_csv("data/processed/panel_features.csv", index=False)
    print("Saved data/processed/panel_features.csv")
