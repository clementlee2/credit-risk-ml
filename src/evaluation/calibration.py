"""
Calibration utilities.

A model can rank firms correctly (good AUC) while still producing
miscalibrated probabilities — e.g. consistently overstating downgrade
risk. Calibration curves check this directly: for firms predicted at
probability p, what fraction actually downgraded?
"""

import numpy as np
from sklearn.calibration import calibration_curve


def normalized_calibration_curve(y_true, y_score, n_bins=10, strategy="quantile"):
    """
    Min-max normalize scores to [0, 1] (needed for non-probabilistic
    scores like Altman Z) and compute a quantile-binned calibration curve.

    Returns (mean_predicted, fraction_positive) or (None, None) if the
    curve cannot be computed (e.g. degenerate bins).
    """
    s = np.asarray(y_score, dtype=float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    try:
        frac, mean_pred = calibration_curve(y_true, s, n_bins=n_bins, strategy=strategy)
        return mean_pred, frac
    except Exception:
        return None, None
