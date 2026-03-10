"""
StatusNow — Comprehensive Model Visualization Suite

Generates:
1. Feature importance by category (top 10 per category)
2. Model performance across versions (V3 → V5 → V5+Enriched)
3. Full evaluation metrics (accuracy, precision, recall, F1, ROC AUC)
4. Per-category precision/recall heatmaps
5. Confusion matrices per version
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
V3_PATH = "data/processed_for_ml_testing.parquet"
OUT_DIR = "/Users/anthonylamas/.gemini/antigravity/brain/6dcd71ed-bf93-479d-b5a1-6ed4db48813c"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Import V5 feature builder
import importlib.util
spec = importlib.util.spec_from_file_location("v5", "scripts/experiments/v5_category_reduction.py")
v5_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5_mod)

# Style constants
COLORS = {
    "v3": "#94a3b8",
    "v5": "#3b82f6",
    "v5e": "#22c55e",
    "accent": "#8b5cf6",
    "red": "#ef4444",
    "orange": "#f97316",
    "bg": "#f8fafc",
}

CATEGORY_DISPLAY = [
    ("Pharmacy", "cat_pharmacy"),
    ("Coffee Shop", "cat_coffee_shop"),
    ("Hotel", "cat_hotel"),
    ("ATMs", "cat_atms"),
    ("Restaurant", "cat_restaurant"),
    ("Pizza Restaurant", "cat_pizza_restaurant"),
    ("Other", "cat_other"),
    ("Package Locker", "cat_package_locker"),
    ("Corporate Office", "cat_corporate_office"),
    ("Art Gallery", "cat_art_gallery"),
    ("Dentist", "cat_dentist"),
    ("Doctor", "cat_doctor"),
    ("Hair Salon", "cat_hair_salon"),
    ("Beauty Salon", "cat_beauty_salon"),
    ("Clothing Store", "cat_clothing_store"),
    ("Jewelry Store", "cat_jewelry_store"),
    ("Church Cathedral", "cat_church_cathedral"),
    ("Community Services", "cat_community_services_non_profits"),
    ("Professional Services", "cat_professional_services"),
    ("Real Estate Agent", "cat_real_estate_agent"),
    ("Landmark", "cat_landmark_and_historical_building"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


# ════════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS AND COLLECT PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
def run_all_models():
    """Train V3, V5, V5+Enriched and collect predictions + probabilities."""
    print("=" * 80)
    print("TRAINING ALL MODEL VERSIONS")
    print("=" * 80)

    # ── V3 ─────────────────────────────────────────────────────────────
    print("\n  [1/3] V3 Baseline …")
    df_v3 = pd.read_parquet(V3_PATH)
    X_v3 = df_v3.drop(columns=["open"])
    y_v3 = df_v3["open"]

    m_v3 = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    v3_preds = cross_val_predict(m_v3, X_v3, y_v3, cv=CV, n_jobs=-1)
    v3_proba = cross_val_predict(m_v3, X_v3, y_v3, cv=CV, method="predict_proba", n_jobs=-1)
    m_v3.fit(X_v3, y_v3)  # fit for feature importance
    v3_fi = pd.Series(m_v3.feature_importances_, index=X_v3.columns)
    print(f"    BalAcc: {balanced_accuracy_score(y_v3, v3_preds):.4f}")

    # ── V5 ─────────────────────────────────────────────────────────────
    print("  [2/3] V5 (category-specific features) …")
    df_v5 = v5_mod.add_v5_features(df_v3)
    X_v5 = df_v5.drop(columns=["open"])
    y_v5 = df_v5["open"]

    m_v5 = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    v5_preds = cross_val_predict(m_v5, X_v5, y_v5, cv=CV, n_jobs=-1)
    v5_proba = cross_val_predict(m_v5, X_v5, y_v5, cv=CV, method="predict_proba", n_jobs=-1)
    m_v5.fit(X_v5, y_v5)
    v5_fi = pd.Series(m_v5.feature_importances_, index=X_v5.columns)
    print(f"    BalAcc: {balanced_accuracy_score(y_v5, v5_preds):.4f}")

    # ── V5+Enriched ────────────────────────────────────────────────────
    print("  [3/3] V5+Enriched (NPI/IRS) …")
    enrichment_path = "data/enrichment_cache.parquet"
    if os.path.exists(enrichment_path):
        enrichment = pd.read_parquet(enrichment_path)
        df_v5e = df_v3.copy()
        for col in enrichment.columns:
            df_v5e[col] = enrichment[col].values
        df_v5e = v5_mod.add_v5_features(df_v5e)
    else:
        df_v5e = df_v5.copy()

    X_v5e = df_v5e.drop(columns=["open"])
    y_v5e = df_v5e["open"]

    m_v5e = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    v5e_preds = cross_val_predict(m_v5e, X_v5e, y_v5e, cv=CV, n_jobs=-1)
    v5e_proba = cross_val_predict(m_v5e, X_v5e, y_v5e, cv=CV, method="predict_proba", n_jobs=-1)
    m_v5e.fit(X_v5e, y_v5e)
    v5e_fi = pd.Series(m_v5e.feature_importances_, index=X_v5e.columns)
    print(f"    BalAcc: {balanced_accuracy_score(y_v5e, v5e_preds):.4f}")

    return {
        "v3": {"X": X_v3, "y": y_v3, "preds": v3_preds, "proba": v3_proba, "fi": v3_fi, "model": m_v3},
        "v5": {"X": X_v5, "y": y_v5, "preds": v5_preds, "proba": v5_proba, "fi": v5_fi, "model": m_v5},
        "v5e": {"X": X_v5e, "y": y_v5e, "preds": v5e_preds, "proba": v5e_proba, "fi": v5e_fi, "model": m_v5e},
    }


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (Open)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (Open)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 (Open)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision (Closed)": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall (Closed)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1 (Closed)": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_proba[:, 1]),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
    }


def per_category_metrics(X, y, preds, proba):
    """Compute per-category metrics."""
    results = []
    for cat_name, cat_col in CATEGORY_DISPLAY:
        if cat_col not in X.columns:
            continue
        mask = X[cat_col] == 1
        if mask.sum() < 10:
            continue
        cy = y[mask]
        cp = preds[mask]
        n = mask.sum()

        # Handle single-class edge case
        n_classes = cy.nunique()
        results.append({
            "Category": cat_name,
            "N": n,
            "Accuracy": accuracy_score(cy, cp),
            "Precision": precision_score(cy, cp, pos_label=1, zero_division=0) if n_classes > 1 else 0,
            "Recall": recall_score(cy, cp, pos_label=1, zero_division=0) if n_classes > 1 else 0,
            "F1": f1_score(cy, cp, pos_label=1, zero_division=0) if n_classes > 1 else 0,
            "Error Rate": 1 - accuracy_score(cy, cp),
            "FP": int(((cy == 0) & (cp == 1)).sum()),
            "FN": int(((cy == 1) & (cp == 0)).sum()),
        })
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════
# CHART 1: Feature Importance by Category (Top 10 per category)
# ════════════════════════════════════════════════════════════════════════════
def chart_feature_importance_by_category(data):
    """Generate per-category feature importance using V5e model."""
    print("\n  Chart 1: Feature importance by category …")

    v5e = data["v5e"]
    X, y, model = v5e["X"], v5e["y"], v5e["model"]
    fi_global = v5e["fi"].sort_values(ascending=False)

    # Select categories with enough samples
    cats_to_plot = []
    for cat_name, cat_col in CATEGORY_DISPLAY:
        if cat_col in X.columns and (X[cat_col] == 1).sum() >= 30:
            cats_to_plot.append((cat_name, cat_col))

    # Limit to top 8 most interesting categories
    cats_to_plot = cats_to_plot[:8]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, (cat_name, cat_col) in enumerate(cats_to_plot):
        ax = axes[idx]
        mask = X[cat_col] == 1
        cat_X = X[mask]
        cat_y = y[mask]

        # Train a small model on just this category to get category-specific importance
        if cat_y.nunique() < 2 or mask.sum() < 30:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cat_name)
            continue

        cat_model = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        )
        cat_model.fit(cat_X, cat_y)
        cat_fi = pd.Series(cat_model.feature_importances_, index=cat_X.columns)
        top10 = cat_fi.sort_values(ascending=False).head(10)

        colors = []
        for f in top10.index:
            if f.startswith(("food_", "pharmacy_", "hotel_", "atm_", "total_", "generic_",
                            "nonbrand_", "stale_", "contact_", "decay_", "conf_",
                            "verification_", "transition_", "churn_")):
                colors.append(COLORS["v5"])  # V5 features
            elif f.startswith("has_npi") or f.startswith("npi_") or f.startswith("has_irs") or f.startswith("irs_"):
                colors.append(COLORS["v5e"])  # Enrichment features
            else:
                colors.append(COLORS["v3"])  # V3 features

        labels = [f.replace("_", " ").title()[:22] for f in top10.index]
        bars = ax.barh(range(len(top10)), top10.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(f"{cat_name} (n={mask.sum()})", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["v3"], label="V3 Core Features"),
        Patch(facecolor=COLORS["v5"], label="V5 Category-Specific"),
        Patch(facecolor=COLORS["v5e"], label="V5b API Enrichment"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Top 10 Feature Importances by Place Category", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feat_importance_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 2: Model Performance Across Versions
# ════════════════════════════════════════════════════════════════════════════
def chart_version_comparison(all_metrics):
    """Bar charts comparing all metrics across V3, V5, V5e."""
    print("\n  Chart 2: Model performance across versions …")

    versions = ["V3 Baseline", "V5 Category", "V5+Enriched"]
    metrics_to_plot = ["Accuracy", "Balanced Accuracy", "Macro F1", "ROC AUC"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = [COLORS["v3"], COLORS["v5"], COLORS["v5e"]]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        vals = [all_metrics[v][metric] for v in ["v3", "v5", "v5e"]]
        bars = ax.bar(versions, vals, color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars, [f"{v:.4f}" for v in vals], padding=5, fontsize=10, fontweight="600")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.015)
        ax.tick_params(axis='x', labelsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("Model Performance Across Versions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "version_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 3: Full Evaluation Metrics Table (as chart)
# ════════════════════════════════════════════════════════════════════════════
def chart_metrics_table(all_metrics):
    """Render the full metrics comparison as a styled table chart."""
    print("\n  Chart 3: Evaluation metrics table …")

    metrics_order = [
        "Accuracy", "Balanced Accuracy", "ROC AUC", "Macro F1",
        "Precision (Open)", "Recall (Open)", "F1 (Open)",
        "Precision (Closed)", "Recall (Closed)", "F1 (Closed)",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    cell_data = []
    for m in metrics_order:
        row = [m]
        for v in ["v3", "v5", "v5e"]:
            row.append(f"{all_metrics[v][m]:.4f}")
        # Delta column
        delta = all_metrics["v5e"][m] - all_metrics["v3"][m]
        row.append(f"{delta:+.4f}")
        cell_data.append(row)

    col_labels = ["Metric", "V3 Baseline", "V5 Category", "V5+Enriched", "Δ (V5e−V3)"]
    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white", fontweight="bold")

    # Style delta column
    for i in range(1, len(cell_data) + 1):
        delta_val = float(cell_data[i-1][-1])
        cell = table[i, 4]
        if delta_val > 0:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(color="#166534", fontweight="bold")
        elif delta_val < 0:
            cell.set_facecolor("#fee2e2")
            cell.set_text_props(color="#991b1b", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(cell_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f1f5f9")

    fig.suptitle("Complete Evaluation Metrics — All Versions", fontsize=14, fontweight="bold", y=0.95)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "evaluation_metrics_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 4: Precision, Recall, F1 Across Versions (Grouped Bar)
# ════════════════════════════════════════════════════════════════════════════
def chart_prf_versions(all_metrics):
    """Grouped bar chart: P/R/F1 for Open and Closed across versions."""
    print("\n  Chart 4: Precision/Recall/F1 across versions …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, label_class in enumerate(["Open", "Closed"]):
        ax = axes[idx]
        metrics = [f"Precision ({label_class})", f"Recall ({label_class})", f"F1 ({label_class})"]
        x = np.arange(len(metrics))
        width = 0.22

        for i, (version, color, vlabel) in enumerate([
            ("v3", COLORS["v3"], "V3"),
            ("v5", COLORS["v5"], "V5"),
            ("v5e", COLORS["v5e"], "V5+E"),
        ]):
            vals = [all_metrics[version][m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, color=color, label=vlabel, edgecolor="white")
            ax.bar_label(bars, [f"{v:.3f}" for v in vals], padding=3, fontsize=8, fontweight="600")

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=11)
        ax.set_title(f"Class: {label_class}", fontsize=13, fontweight="bold")
        ax.set_ylim(0.82, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=9)

    fig.suptitle("Precision, Recall & F1 Score Across Model Versions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "prf_across_versions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 5: Per-Category Precision & Recall Heatmap
# ════════════════════════════════════════════════════════════════════════════
def chart_per_category_pr(data):
    """Heatmap of precision and recall per category for V5+Enriched."""
    print("\n  Chart 5: Per-category precision & recall …")

    v5e = data["v5e"]
    cat_metrics = per_category_metrics(v5e["X"], v5e["y"], v5e["preds"], v5e["proba"])
    cat_metrics = cat_metrics.sort_values("Error Rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Precision/Recall bars side by side
    cats = cat_metrics["Category"].tolist()
    y_pos = np.arange(len(cats))
    bh = 0.35

    ax = axes[0]
    ax.barh(y_pos + bh/2, cat_metrics["Precision"], bh, color="#3b82f6", label="Precision", alpha=0.85)
    ax.barh(y_pos - bh/2, cat_metrics["Recall"], bh, color="#8b5cf6", label="Recall", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Precision & Recall per Category (V5+Enriched)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)

    # Error rate + FN/FP
    ax = axes[1]
    err_colors = plt.cm.RdYlGn_r(cat_metrics["Error Rate"] / cat_metrics["Error Rate"].max())
    ax.barh(y_pos, cat_metrics["Error Rate"], color=err_colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Error Rate")
    ax.set_title("Error Rate per Category", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    for i, (_, row) in enumerate(cat_metrics.iterrows()):
        ax.text(row["Error Rate"] + 0.005, i,
                f"FP:{row['FP']} FN:{row['FN']}",
                va="center", fontsize=7.5, color="#64748b")

    fig.suptitle("Per-Category Model Performance (V5+Enriched)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "per_category_pr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 6: Per-Category Error Rate Change (V3 → V5e)
# ════════════════════════════════════════════════════════════════════════════
def chart_error_rate_change(data):
    """Show error rate changes from V3 to V5+Enriched per category."""
    print("\n  Chart 6: Error rate change V3 → V5e …")

    v3_metrics = per_category_metrics(data["v3"]["X"], data["v3"]["y"], data["v3"]["preds"], data["v3"]["proba"])
    v5e_metrics = per_category_metrics(data["v5e"]["X"], data["v5e"]["y"], data["v5e"]["preds"], data["v5e"]["proba"])

    merged = v3_metrics.merge(v5e_metrics, on="Category", suffixes=("_v3", "_v5e"))
    merged["Δ Error"] = merged["Error Rate_v5e"] - merged["Error Rate_v3"]
    merged = merged.sort_values("Δ Error")

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(merged))
    colors = ["#22c55e" if d < 0 else "#ef4444" if d > 0 else "#94a3b8" for d in merged["Δ Error"]]

    bars = ax.barh(y_pos, merged["Δ Error"], color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    labels = [f"{row['Category']}  (V3:{row['Error Rate_v3']:.1%} → V5e:{row['Error Rate_v5e']:.1%})"
              for _, row in merged.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#1e293b", linewidth=1.2)
    ax.set_xlabel("Change in Error Rate (negative = improved)", fontsize=11)
    ax.set_title("Error Rate Change by Category: V3 → V5+Enriched", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()

    for i, (_, row) in enumerate(merged.iterrows()):
        delta = row["Δ Error"]
        ha = "left" if delta >= 0 else "right"
        offset = 0.002 if delta >= 0 else -0.002
        ax.text(delta + offset, i, f"{delta:+.1%}", va="center", fontsize=8, fontweight="600",
                color=colors[i])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "error_rate_change.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 7: Confusion Matrices
# ════════════════════════════════════════════════════════════════════════════
def chart_confusion_matrices(data):
    """Side-by-side confusion matrices for all versions."""
    print("\n  Chart 7: Confusion matrices …")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    version_names = [("v3", "V3 Baseline"), ("v5", "V5 Category"), ("v5e", "V5+Enriched")]

    for idx, (vkey, vlabel) in enumerate(version_names):
        ax = axes[idx]
        d = data[vkey]
        cm = confusion_matrix(d["y"], d["preds"])

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Closed", "Open"], fontsize=10)
        ax.set_yticklabels(["Closed", "Open"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(vlabel, fontsize=12, fontweight="bold")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm.sum():.1%})",
                        ha="center", va="center", fontsize=11, color=color, fontweight="600")

    fig.suptitle("Confusion Matrices Across Model Versions", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# CHART 8: Global Feature Importance (Top 20)
# ════════════════════════════════════════════════════════════════════════════
def chart_global_feature_importance(data):
    """Top 20 features for V5+Enriched model with color coding."""
    print("\n  Chart 8: Global feature importance (top 20) …")

    fi = data["v5e"]["fi"].sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = []
    for f in fi.index:
        if f.startswith(("food_", "pharmacy_", "hotel_", "atm_", "total_liveness", "generic_liveness",
                         "nonbrand_", "stale_food", "stale_pharmacy", "stale_hotel", "stale_atm",
                         "contact_food", "contact_hotel", "contact_pharmacy",
                         "decay_food", "decay_hotel", "conf_food", "conf_hotel", "conf_atm",
                         "verification_", "transition_", "churn_")):
            colors.append(COLORS["v5"])
        elif f.startswith(("has_npi", "npi_", "has_irs", "irs_")):
            colors.append(COLORS["v5e"])
        else:
            colors.append(COLORS["v3"])

    bars = ax.barh(range(len(fi)), fi.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels([f.replace("_", " ").title() for f in fi.index], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Top 20 Feature Importances (V5+Enriched Model)", fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["v3"], label="V3 Core"),
        Patch(facecolor=COLORS["v5"], label="V5 Category-Specific"),
        Patch(facecolor=COLORS["v5e"], label="V5b API Enrichment"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "global_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("STATUSNOW — COMPREHENSIVE MODEL VISUALIZATION SUITE")
    print("=" * 80)

    # Run all models
    data = run_all_models()

    # Compute metrics
    all_metrics = {}
    for vkey in ["v3", "v5", "v5e"]:
        d = data[vkey]
        all_metrics[vkey] = compute_metrics(d["y"], d["preds"], d["proba"])

    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    for m in ["Accuracy", "Balanced Accuracy", "Macro F1", "ROC AUC"]:
        print(f"  {m:25s}  V3={all_metrics['v3'][m]:.4f}  V5={all_metrics['v5'][m]:.4f}  V5e={all_metrics['v5e'][m]:.4f}")

    # Generate all charts
    print("\n" + "=" * 80)
    print("GENERATING CHARTS")
    print("=" * 80)

    chart_global_feature_importance(data)
    chart_feature_importance_by_category(data)
    chart_version_comparison(all_metrics)
    chart_metrics_table(all_metrics)
    chart_prf_versions(all_metrics)
    chart_per_category_pr(data)
    chart_error_rate_change(data)
    chart_confusion_matrices(data)

    print("\n✅ All 8 charts generated successfully!")
    print(f"   Output directory: {OUT_DIR}")
