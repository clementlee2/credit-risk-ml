"""
Walk-forward (rolling-origin) cross-validation.

Standard k-fold randomly shuffles observations, which creates look-ahead
bias in time-series settings: a model trained partly on 2010 data could
be evaluated on 2005 data, having implicitly learned from the future.

Walk-forward validation is strictly causal:

    Fold 1:  train [1990–2004] -> test [2005]
    Fold 2:  train [1990–2005] -> test [2006]
    ...
    Fold 18: train [1990–2021] -> test [2022]

All feature scaling (StandardScaler) is fit within each training window
only, never on the test fold, to avoid leakage.
"""

from src.evaluation.metrics import FoldResult
from src.models.altman import altman_score
from src.models.ohlson import fit_ohlson, predict_ohlson
from src.models.logistic import fit_logistic, predict_logistic
from src.models.XgBoost import fit_xgboost, predict_xgboost, HAS_XGB


def walk_forward_cv(panel, train_start=1990, test_start=2005, test_end=2022, verbose=True):
    results = []
    for t in range(test_start, test_end + 1):
        X_train = panel[(panel["year"] < t) & (panel["year"] >= train_start)]
        X_test = panel[panel["year"] == t]
        y_train = X_train["downgrade"].values
        y_test = X_test["downgrade"].values

        if len(X_train) < 200 or len(X_test) < 20 or y_train.sum() < 10:
            continue

        if verbose:
            print(f"  {t}: train={len(X_train):,} | test={len(X_test):,} | "
                  f"downgrades={y_test.sum()} ({y_test.mean():.1%})")

        results.append(FoldResult(t, "Altman Z-score",
                                   y_test, altman_score(X_test), len(X_train), len(X_test)))

        m2 = fit_ohlson(X_train, y_train)
        results.append(FoldResult(t, "Ohlson (re-estimated)",
                                   y_test, predict_ohlson(m2, X_test), len(X_train), len(X_test)))

        m3 = fit_logistic(X_train, y_train)
        results.append(FoldResult(t, "Logistic (full features)",
                                   y_test, predict_logistic(m3, X_test), len(X_train), len(X_test)))

        if HAS_XGB:
            m4 = fit_xgboost(X_train, y_train)
            results.append(FoldResult(t, "XGBoost",
                                       y_test, predict_xgboost(m4, X_test), len(X_train), len(X_test)))

    return results
