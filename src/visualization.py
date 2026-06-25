"""
All figure-generation functions, kept separate from analysis logic so the
pipeline script stays readable. Every function saves its own PNG to
results/figures/ and also returns the matplotlib Figure for inline use
in a notebook.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.calibration import normalized_calibration_curve
from src.features.build_features import (
    ALTMAN_FEATURES, OHLSON_FEATURES, ALL_FEATURES, FEATURE_LABELS,
)

MODEL_COLORS = {
    "Altman Z-score": "#888780",
    "Ohlson (re-estimated)": "#534AB7",
    "Logistic (full features)": "#0F6E56",
    "XGBoost": "#D85A30",
}
MODEL_LS = {
    "Altman Z-score": "--", "Ohlson (re-estimated)": "-.",
    "Logistic (full features)": ":", "XGBoost": "-",
}


def plot_panel_overview(panel: pd.DataFrame, recession_years: set, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    annual = panel.groupby("year").agg(
        rate=("downgrade", "mean"),
        recession=("is_recession", "first"),
    ).reset_index()

    axes[0].bar(
        annual["year"], annual["rate"] * 100,
        color=[("#D85A30" if r else "#B5D4F4") for r in annual["recession"]],
        width=0.8, edgecolor="white", linewidth=0.4,
    )
    axes[0].set_ylabel("Downgrade rate (%)")
    axes[0].set_xlabel("Year")
    axes[0].set_title("Annual downgrade rate  (orange = recession year)")
    axes[0].tick_params(axis="x", rotation=45)

    bins = np.linspace(-2, 12, 60)
    axes[1].hist(panel[panel["downgrade"] == 0]["altman_z"], bins=bins,
                 alpha=0.55, color="#B5D4F4", label="No downgrade", density=True)
    axes[1].hist(panel[panel["downgrade"] == 1]["altman_z"], bins=bins,
                 alpha=0.65, color="#D85A30", label="Downgrade", density=True)
    axes[1].axvline(1.81, color="#534AB7", lw=1.2, ls="--", label="Distress (Z < 1.81)")
    axes[1].axvline(2.99, color="#0F6E56", lw=1.2, ls=":", label="Safe (Z > 2.99)")
    axes[1].set_xlabel("Altman Z-score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Z-score distribution by downgrade outcome")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_correlations(panel: pd.DataFrame, out_path: str):
    corr = panel[ALL_FEATURES + ["downgrade"]].corr()["downgrade"].drop("downgrade").sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(corr)), corr.values,
            color=["#D85A30" if v < 0 else "#185FA5" for v in corr.values],
            height=0.65)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in corr.index], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson correlation with downgrade indicator")
    ax.set_title("Univariate feature\u2013downgrade correlations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def pool_predictions(results):
    """Pool fold-level predictions across all test years, per model."""
    pooled_true, pooled_scores = defaultdict(list), defaultdict(list)
    for r in results:
        pooled_true[r.model_name].extend(r.y_true.tolist())
        pooled_scores[r.model_name].extend(r.y_score.tolist())
    return pooled_true, pooled_scores


def plot_roc_auc_over_time(pooled_true, pooled_scores, detail: pd.DataFrame,
                            recession_years: set, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name in pooled_true:
        y = np.array(pooled_true[name])
        s = np.array(pooled_scores[name])
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "k"),
                ls=MODEL_LS.get(name, "-"), lw=1.8,
                label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves \u2014 pooled out-of-sample (2005\u20132022)")
    ax.legend(fontsize=9, framealpha=0.9)

    ax2 = axes[1]
    for name, grp in detail.groupby("model"):
        grp = grp.sort_values("year")
        ax2.plot(grp["year"], grp["auc"],
                 color=MODEL_COLORS.get(name, "k"), ls=MODEL_LS.get(name, "-"),
                 lw=1.8, marker="o", markersize=3.5, label=name)
    for s, e in [(2001, 2002), (2008, 2009), (2020, 2020)]:
        ax2.axvspan(s - 0.4, e + 0.4, color="#D3D1C7", alpha=0.45, zorder=0)
    ax2.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.5)
    ax2.set_xlabel("Test year")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("AUC by year  (grey = recession)")
    ax2.legend(fontsize=9, framealpha=0.9, loc="lower left")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration_recession(pooled_true, pooled_scores, detail: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5, label="Perfect calibration")
    for name in pooled_true:
        y = np.array(pooled_true[name])
        s = np.array(pooled_scores[name])
        mean_pred, frac = normalized_calibration_curve(y, s)
        if mean_pred is not None:
            ax.plot(mean_pred, frac, color=MODEL_COLORS.get(name, "k"),
                     lw=1.8, marker="s", markersize=4, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves")
    ax.legend(fontsize=9, framealpha=0.9)

    ax2 = axes[1]
    rec_auc = detail[detail["recession"] == 1].groupby("model")["auc"].mean()
    exp_auc = detail[detail["recession"] == 0].groupby("model")["auc"].mean()
    delta = (rec_auc - exp_auc).sort_values()
    ax2.barh(range(len(delta)), delta.values,
             color=["#D85A30" if v < 0 else "#185FA5" for v in delta.values],
             height=0.55)
    ax2.set_yticks(range(len(delta)))
    ax2.set_yticklabels(delta.index, fontsize=10)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("\u0394AUC  (recession \u2212 expansion)")
    ax2.set_title("AUC degradation in recessions")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_importance(panel: pd.DataFrame, fit_xgboost_fn, out_path: str, split_year=2015):
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping feature importance plot (pip install shap)")
        return None

    train_data = panel[panel["year"] <= split_year]
    test_data = panel[panel["year"] > split_year]
    m_shap = fit_xgboost_fn(train_data, train_data["downgrade"].values)
    if m_shap is None:
        print("xgboost not installed — skipping feature importance plot")
        return None

    explainer = shap.TreeExplainer(m_shap)
    shap_vals = explainer.shap_values(test_data[ALL_FEATURES].fillna(0))

    importance = (
        pd.DataFrame({
            "feature": ALL_FEATURES,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(14)
    )
    importance["label"] = importance["feature"].map(FEATURE_LABELS).fillna(importance["feature"])

    def gcol(f):
        if f in ALTMAN_FEATURES:
            return "#888780"
        if f in OHLSON_FEATURES:
            return "#534AB7"
        return "#D85A30"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(importance)), importance["mean_abs_shap"],
            color=[gcol(f) for f in importance["feature"]], height=0.65)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance["label"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"XGBoost feature importance (SHAP)  \u2014  train \u2264 {split_year}, test {split_year+1}\u20132022")
    ax.legend(handles=[
        Patch(facecolor="#888780", label="Altman"),
        Patch(facecolor="#534AB7", label="Ohlson"),
        Patch(facecolor="#D85A30", label="Market-based"),
    ], fontsize=9, framealpha=0.9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig, importance
