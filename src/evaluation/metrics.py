"""
Per-fold evaluation metrics.

FoldResult wraps one (test_year, model) pair's predictions and exposes
three metrics relevant to a credit-risk use case:

- auc: ranking quality (can this model order risky firms above safe ones?)
- brier: probability calibration (lower is better; a proper scoring rule)
- precision_top_decile: operational usefulness — if an analyst can only
  act on the riskiest 10% of firms, how many of those flags are correct?
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


@dataclass
class FoldResult:
    test_year: int
    model_name: str
    y_true: np.ndarray
    y_score: np.ndarray
    n_train: int
    n_test: int

    @property
    def auc(self):
        if self.y_true.sum() in (0, len(self.y_true)):
            return np.nan
        return roc_auc_score(self.y_true, self.y_score)

    @property
    def brier(self):
        """
        Proper scoring rule — lower is better.
        Altman Z is not a probability, so we min-max normalize before
        computing. Note: this means Altman's Brier score reflects
        best-possible calibration of a monotone transform of Z, not the
        Z-score itself.
        """
        s = self.y_score.copy()
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min)
        return brier_score_loss(self.y_true, s)

    @property
    def precision_top_decile(self):
        """
        Fraction of actual downgrades in the top-10% highest-risk firms.
        Operationally relevant: a credit analyst flags the top-decile
        firms — this measures how many of those flags are actually
        correct.
        """
        thresh = np.percentile(self.y_score, 90)
        top = self.y_true[self.y_score >= thresh]
        return top.mean() if len(top) > 0 else np.nan
