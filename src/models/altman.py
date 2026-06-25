"""
Model 1 — Altman Z-score (1968).

Published coefficients applied directly. No fitting. This is the pure
baseline: if any modern model can't beat a fixed formula from 1968,
something is wrong with the approach.
"""

import pandas as pd
from src.features.build_features import ALTMAN_FEATURES  # noqa: F401  (kept for reference)


def altman_score(df: pd.DataFrame):
    """
    Altman (1968) original coefficients for public firms.
    Negated so higher score = higher downgrade risk (for AUC consistency).
    """
    z = (1.2 * df["X1"] + 1.4 * df["X2"] + 3.3 * df["X3"]
         + 0.6 * df["X4"] + 1.0 * df["X5"]).values
    return -z
