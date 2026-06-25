"""
DeLong, DeLong & Clarke-Pearson (1988) test for comparing two correlated
AUC values estimated on the same sample. Used here to test whether each
model's AUC improvement over the Altman baseline is statistically
significant, rather than relying on the point estimate alone.
"""

import numpy as np
from scipy import stats


def delong_test(y_true, scores_a, scores_b):
    """
    Returns (z-statistic, two-sided p-value). H0: AUC_A = AUC_B.
    """
    def _auc_var(y, s):
        n1, n0 = y.sum(), (y == 0).sum()
        if n1 == 0 or n0 == 0:
            return np.nan, np.nan
        pos, neg = s[y == 1], s[y == 0]
        V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        V01 = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
        return V10.mean(), np.var(V10) / n1 + np.var(V01) / n0

    auc_a, var_a = _auc_var(y_true, scores_a)
    auc_b, var_b = _auc_var(y_true, scores_b)
    if any(np.isnan([auc_a, auc_b, var_a, var_b])):
        return np.nan, np.nan
    se = np.sqrt(var_a + var_b)
    if se == 0:
        return np.nan, np.nan
    z = (auc_a - auc_b) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))
