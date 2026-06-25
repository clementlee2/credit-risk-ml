"""
Model 2 — Ohlson O-score (re-estimated logistic).

Ohlson's 1980 paper introduced logistic regression for credit modeling —
a major improvement over Altman's discriminant analysis because it
produces calibrated probabilities directly.

We re-estimate on each training window rather than using Ohlson's
original 1970s coefficients. This is the methodologically honest
comparison: same features, data-driven weights. Comparing this model
against Altman isolates the effect of fitting weights to data, holding
the feature set and functional form fixed.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.build_features import OHLSON_FEATURES


def fit_ohlson(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train[OHLSON_FEATURES].fillna(0))
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                            class_weight="balanced", random_state=0)
    m.fit(Xs, y_train)
    m._scaler = scaler
    return m


def predict_ohlson(m, X_test):
    return m.predict_proba(
        m._scaler.transform(X_test[OHLSON_FEATURES].fillna(0)))[:, 1]
