"""
Model 4 — XGBoost.

Gradient boosting captures non-linear interactions — e.g. the impact of
high leverage is likely different in a high-growth firm vs. a declining
one. scale_pos_weight handles class imbalance by weighting positive
(downgrade) cases proportionally.

Hyperparameters are conservative (max_depth=4, small learning_rate) to
avoid overfitting on small test-year samples.
"""

import numpy as np

from src.features.build_features import ALL_FEATURES

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def fit_xgboost(X_train, y_train):
    if not HAS_XGB:
        return None
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=42, verbosity=0,
    )
    m.fit(X_train[ALL_FEATURES].fillna(0), y_train)
    return m


def predict_xgboost(m, X_test):
    if m is None:
        return np.full(len(X_test), np.nan)
    return m.predict_proba(X_test[ALL_FEATURES].fillna(0))[:, 1]
