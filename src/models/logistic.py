"""
Model 3 — Regularized logistic regression (full feature set).

Adds market-based variables to Ohlson's feature set. Stronger L2
regularization (C=0.1 vs C=1.0 for Ohlson) penalizes the larger feature
space.

This isolates a key question: how much of any ML advantage comes from
better features vs. better functional form? Comparing this model against
Ohlson isolates the feature effect; comparing XGBoost against this model
isolates the non-linearity effect.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.build_features import ALL_FEATURES


def fit_logistic(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train[ALL_FEATURES].fillna(0))
    m = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
                            class_weight="balanced", random_state=0)
    m.fit(Xs, y_train)
    m._scaler = scaler
    return m


def predict_logistic(m, X_test):
    return m.predict_proba(
        m._scaler.transform(X_test[ALL_FEATURES].fillna(0)))[:, 1]
