"""
Experiment Runner V3 — builds on V2 with:
  1. Dynamic Label Refinement (high-confidence mislabel detection)
  2. Brand-stratified analysis (micro-ensembling insight)
  3. Full V2 vs V3 comparison
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    make_scorer, precision_score, recall_score,
    balanced_accuracy_score, classification_report
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def _make_models():
    return {
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            scale_pos_weight=1, eval_metric="auc",
            random_state=42, n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000,
            random_state=42, n_jobs=-1,
        ),
    }


SCORING = {
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "precision_closed": make_scorer(precision_score, pos_label=0),
    "recall_closed": make_scorer(recall_score, pos_label=0),
    "balanced_acc": "balanced_accuracy",
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DYNAMIC LABEL REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════
def label_refinement(X, y, threshold=0.90):
    """
    Use cross-validated predictions to find high-confidence mislabels.
    A sample is flagged when the model is >= threshold confident the TRUE label
    is wrong (i.e. model predicts opposite class very confidently).
    Returns cleaned X, y with suspect labels removed.
    """
    print("\n" + "=" * 80)
    print("DYNAMIC LABEL REFINEMENT")
    print("=" * 80)

    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )

    # Cross-validated probability predictions
    print(f"  Running 5-fold CV for mislabel detection (threshold={threshold}) …")
    proba = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)

    # For each sample: probability the model assigns to the GIVEN label
    label_prob = np.where(y == 1, proba[:, 1], proba[:, 0])
    # If the model is very confident the label is WRONG, flag it
    suspect_mask = label_prob < (1 - threshold)

    n_suspect = suspect_mask.sum()
    print(f"  Found {n_suspect} suspect labels ({n_suspect / len(y):.1%} of data)")

    if n_suspect > 0:
        suspect_idx = np.where(suspect_mask)[0]
        suspect_y = y.iloc[suspect_idx]
        n_open_suspect = (suspect_y == 1).sum()
        n_closed_suspect = (suspect_y == 0).sum()
        print(f"    ↳ {n_open_suspect} labeled 'Open' but model says 'Closed'")
        print(f"    ↳ {n_closed_suspect} labeled 'Closed' but model says 'Open'")

        # Show worst offenders
        confidence_gap = 1 - label_prob
        worst = np.argsort(confidence_gap)[::-1][:10]
        print(f"\n  Top 10 most confidently mislabeled:")
        print(f"  {'Index':>7}  {'Given':>6}  {'Model P(given)':>14}  {'Confidence gap':>14}")
        for idx in worst:
            print(f"  {idx:>7}  {'Open' if y.iloc[idx]==1 else 'Closed':>6}  "
                  f"{label_prob[idx]:>14.3f}  {confidence_gap[idx]:>14.3f}")

    # Remove suspect samples
    clean_mask = ~suspect_mask
    X_clean = X[clean_mask].reset_index(drop=True)
    y_clean = y[clean_mask].reset_index(drop=True)

    print(f"\n  ✅ Cleaned dataset: {len(X_clean)} samples (removed {n_suspect})")
    print(f"     New class balance: {y_clean.mean():.2%} Open")

    return X_clean, y_clean, n_suspect


# ═══════════════════════════════════════════════════════════════════════════
# 2. BRAND-STRATIFIED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def brand_stratified_analysis(X, y):
    """Check if brands vs non-brands have different accuracy profiles."""
    print("\n" + "=" * 80)
    print("BRAND-STRATIFIED ANALYSIS")
    print("=" * 80)

    if "is_brand" not in X.columns:
        print("  ⚠️  'is_brand' not in features — skipping")
        return

    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )

    preds = cross_val_predict(model, X, y, cv=CV, n_jobs=-1)

    brand_mask = X["is_brand"] == 1
    nonbrand_mask = X["is_brand"] == 0

    brand_acc = balanced_accuracy_score(y[brand_mask], preds[brand_mask]) if brand_mask.sum() > 30 else float("nan")
    nonbrand_acc = balanced_accuracy_score(y[nonbrand_mask], preds[nonbrand_mask]) if nonbrand_mask.sum() > 30 else float("nan")

    print(f"\n  Brand samples:     {brand_mask.sum():>5d}  BalAcc = {brand_acc:.4f}")
    print(f"  Non-brand samples: {nonbrand_mask.sum():>5d}  BalAcc = {nonbrand_acc:.4f}")
    gap = brand_acc - nonbrand_acc
    print(f"  Gap:               {gap:+.4f}")

    if abs(gap) > 0.05:
        print(f"\n  🚨 Significant gap! A stratified/micro-ensemble approach is recommended.")
        print(f"     → Train a precision-weighted model for brands")
        print(f"     → Train a recall-weighted model for non-brands")
    else:
        print(f"\n  ✅ Gap is small — a single global model is adequate.")

    # Category-level breakdown
    if "category_churn_risk" in X.columns:
        # Split into high/low churn
        median_risk = X["category_churn_risk"].median()
        high_churn = X["category_churn_risk"] >= median_risk
        low_churn = ~high_churn

        high_acc = balanced_accuracy_score(y[high_churn], preds[high_churn]) if high_churn.sum() > 30 else float("nan")
        low_acc = balanced_accuracy_score(y[low_churn], preds[low_churn]) if low_churn.sum() > 30 else float("nan")

        print(f"\n  High-churn categories: {high_churn.sum():>5d}  BalAcc = {high_acc:.4f}")
        print(f"  Low-churn categories:  {low_churn.sum():>5d}  BalAcc = {low_acc:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════
def run_cv(X, y, label=""):
    models = _make_models()
    results = {}
    for name, model in models.items():
        print(f"  Training {name} …")
        t0 = time.time()
        cv_res = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            "Time (s)": elapsed,
            "ROC AUC": cv_res["test_roc_auc"].mean(),
            "F1 Macro": cv_res["test_f1_macro"].mean(),
            "Prec(Closed)": cv_res["test_precision_closed"].mean(),
            "Recall(Closed)": cv_res["test_recall_closed"].mean(),
            "Balanced Acc": cv_res["test_balanced_acc"].mean(),
        }
    df_r = pd.DataFrame(results).T
    print(f"\n{'─'*70}")
    print(f" {label}")
    print(f"{'─'*70}")
    print(df_r.round(4).to_string())
    return df_r


def run_experiments_v3(v3_path="data/processed_for_ml_v3.parquet"):
    print("=" * 80)
    print("V3 EXPERIMENT SUITE")
    print("=" * 80)

    # ── load all versions ─────────────────────────────────────────────────
    v2 = pd.read_parquet("data/processed_for_ml_v2.parquet")
    v2 = pd.read_parquet("data/processed_for_ml_v2.parquet")
    v3 = pd.read_parquet(v3_path)

    X_v2, y_v2 = v2.drop(columns=["open"]), v2["open"]
    X_v3, y_v3 = v3.drop(columns=["open"]), v3["open"]

    print(f"\nV2: {X_v2.shape[0]} samples, {X_v2.shape[1]} features")
    print(f"V3: {X_v3.shape[0]} samples, {X_v3.shape[1]} features")

    # ── A. V2 baseline ────────────────────────────────────────────────────
    print("\n\n▶ A. V2 BASELINE")
    r_v2 = run_cv(X_v2, y_v2, label="V2 — Interactions + PCA + Churn Risk")

    # ── B. V3 (recency bins + brand interactions) ─────────────────────────
    print("\n\n▶ B. V3 FEATURES (Recency bins + Brand-aware)")
    r_v3 = run_cv(X_v3, y_v3, label="V3 — V2 + Recency Decay + Brand-aware")

    # ── C. Label Refinement on V3 ─────────────────────────────────────────
    print("\n\n▶ C. LABEL REFINEMENT ON V3")
    X_clean, y_clean, n_removed = label_refinement(X_v3, y_v3, threshold=0.90)
    r_v3_clean = run_cv(X_clean, y_clean, label="V3 + Label Refinement (cleaned)")

    # ── D. Brand-stratified analysis ──────────────────────────────────────
    brand_stratified_analysis(X_v3, y_v3)

    # ── COMPARISON ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)

    best_v2 = r_v2["Balanced Acc"].max()
    best_v3 = r_v3["Balanced Acc"].max()
    best_v3c = r_v3_clean["Balanced Acc"].max()

    print(f"\n  V2 best:          {best_v2:.4f}  ({r_v2['Balanced Acc'].idxmax()})")
    print(f"  V3 best:          {best_v3:.4f}  ({r_v3['Balanced Acc'].idxmax()})  Δ {best_v3 - best_v2:+.4f}")
    print(f"  V3+Clean best:    {best_v3c:.4f}  ({r_v3_clean['Balanced Acc'].idxmax()})  Δ {best_v3c - best_v2:+.4f}")

    if best_v3c > 0.72:
        print("\n  🏆 BREAKTHROUGH: Exceeded 72% balanced accuracy!")
    elif best_v3c > best_v2:
        print(f"\n  📈 Improved from V2 ({best_v2:.4f}) → V3+Clean ({best_v3c:.4f})")

    print(f"\n  Label refinement removed {n_removed} suspect samples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-input", "-i", type=str, default="data/processed_for_ml_v3.parquet", help="Path to V3 processed parquet file")
    args = parser.parse_args()
    
    run_experiments_v3(v3_path=args.v3_input)
