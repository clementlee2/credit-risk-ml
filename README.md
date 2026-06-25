# Credit Default Prediction: Does Modern ML Beat Altman (1968)?

This project benchmarks four credit risk models — spanning 55 years of methodology, from Altman's 1968 discriminant analysis to a modern XGBoost ensemble — on a single, controlled task: predicting a corporate credit rating downgrade within 12 months.

The question it tries to answer is simple: **how much has modeling actually improved, and where does the improvement actually come from?**

> **Note:** All experiments run on a synthetic firm-year panel generated to follow a Shumway (2001)-style hazard process. The data is not based on real financial statements and this is not investment advice — it's a controlled methodology demonstration.

## Why this exists

Most credit risk comparisons are confounded — newer models use different (and more) features, different sample periods, and different validation schemes than the classical benchmarks they're compared against, so it's never clear whether gains are coming from better data, better features, or better algorithms.

This notebook controls for that by isolating each source of improvement one step at a time:

| Step | Comparison | Isolates |
|------|------------|----------|
| Model 2 vs. Model 1 | Re-estimated logistic vs. fixed 1968 coefficients | Effect of fitting weights to data |
| Model 3 vs. Model 2 | Full feature set vs. 9 Ohlson accounting ratios | Effect of richer features |
| Model 4 vs. Model 3 | Gradient boosting vs. linear logistic | Effect of non-linear functional form |

## Model lineup

| # | Model | Source | Method |
|---|-------|--------|--------|
| 1 | **Altman Z-score** | Altman (1968) | Linear discriminant analysis, original published coefficients, no fitting |
| 2 | **Ohlson O-score** | Ohlson (1980) | Logistic regression on 9 accounting predictors, re-estimated per fold |
| 3 | **Logistic (L2)** | — | Logistic regression on the full accounting + market feature set, L2-regularized |
| 4 | **XGBoost** | — | Gradient-boosted trees on the full feature set, interpreted via SHAP |

## Data

A synthetic panel of 600 firms from 1990–2022, generated from a latent firm-quality factor that drives all accounting ratios. Key properties:

- **Target**: rating downgrade within 12 months (~8% base rate — realistically class-imbalanced)
- **Macro shocks**: recession years (2001, 2002, 2008, 2009, 2020) raise baseline downgrade risk by ~0.8 log-odds
- **Features**: 18 predictors across three families
  - **Altman ratios** (X1–X5): working capital/assets, retained earnings/assets, EBIT/assets, market equity/liabilities, sales/assets
  - **Ohlson ratios**: size, leverage, liquidity, profitability, and distress-flag variables (TLTA, WCTA, CLCA, NITA, FUTL, INTWO, CHIN, OENEG)
  - **Market-based**: return volatility, cumulative return, log market cap, market leverage

## Evaluation framework

- **Walk-forward validation** — train on all years up to *t*, test on year *t+1*, roll forward through 2022 (18 folds total). This avoids the look-ahead bias that standard k-fold introduces in time-series settings.
- **Metrics**:
  - **AUC-ROC** — ranking quality
  - **Brier score** — probability calibration (lower is better)
  - **Precision@top-decile** — operational usefulness if you can only act on your highest-risk 10%
  - Metrics are also split by **recession vs. expansion** years
- **DeLong test (1988)** — non-parametric significance test comparing each model's AUC against the Altman baseline

## Results

| Model | AUC | ± | Brier ↓ | P@10% | AUC (rec.) | AUC (exp.) | vs. Altman | DeLong *p* |
|---|---|---|---|---|---|---|---|---|
| Altman Z-score | 0.705 | 0.061 | 0.5217 | 0.166 | 0.704 | 0.705 | baseline | — |
| Ohlson (re-est.) | 0.721 | 0.085 | 0.2408 | **0.251** | 0.717 | 0.721 | +0.018 | 0.4038 |
| Logistic (L2) | **0.741** | 0.060 | 0.2757 | 0.248 | 0.741 | 0.741 | **+0.045** | **0.0269** \* |
| XGBoost | 0.729 | 0.084 | **0.1903** | 0.231 | 0.731 | 0.728 | +0.029 | 0.1560 |

**Takeaways:**
- Modern models edge out Altman, but the only statistically significant improvement is **Logistic (L2)** — and the gain is modest (~0.04 AUC).
- Logistic regression wins on ranking because the synthetic data is approximately linear in log-odds; it captures the structure well.
- **XGBoost wins on calibration**, not ranking — non-linearities refine probability estimates without materially reordering risk.
- **Ohlson is best at flagging extreme distress** (highest precision in the riskiest decile), even though it ranks lower overall.
- **All models degrade during recessions** — accounting variables adjust with a lag, limiting their ability to capture fast-moving credit deterioration in real time.

  

## Getting started

### Requirements
```
numpy
pandas
matplotlib
scipy
scikit-learn
xgboost      # optional — Model 4 is skipped without it
shap         # optional — feature importance plot is skipped without it
```

### Install & run
```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost shap
jupyter notebook credit_default_risk_ML.ipynb
```

The notebook is self-contained: it generates the synthetic panel, fits all four models, runs walk-forward evaluation, and saves figures/tables to `results/` — no external data files needed.

## What's inside the notebook

1. **Environment setup** — imports, optional-dependency checks, output folders
2. **Data generation** — synthetic firm-year panel with a latent quality factor and recession shocks
3. **Feature engineering** — Altman, Ohlson, and market feature families
4. **Model definitions** — Altman Z-score, Ohlson O-score, regularized logistic, XGBoost
5. **Walk-forward evaluation** — 18 rolling train/test folds (1990–2022)
6. **Diagnostics** — pooled ROC curves, DeLong significance tests, calibration curves, SHAP feature importance
7. **Results & discussion** — full metrics table and write-up

## References

- Altman, E.I. (1968). Financial ratios, discriminant analysis and prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.
- Beaver, W.H. (1966). Financial ratios as predictors of failure. *Journal of Accounting Research*, 4, 71–111.
- Ohlson, J.A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, 18(1), 109–131.
- Shumway, T. (2001). Forecasting bankruptcy more accurately: a simple hazard model. *Journal of Business*, 74(1), 101–124.
- DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*, 44(3), 837–845.
- Bharath, S.T., & Shumway, T. (2008). Forecasting default with the Merton distance-to-default model. *Review of Financial Studies*, 21(3), 1339–1369.
- Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.

## Disclaimer

All data used in this project is synthetically generated for methodology demonstration. It does not represent real firms, real financial statements, or real credit outcomes, and nothing here constitutes investment advice.
