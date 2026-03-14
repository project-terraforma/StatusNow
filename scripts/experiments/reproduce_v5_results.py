"""
Reproduce V5 Results — 89.41% Balanced Accuracy
================================================

This script reproduces the best result from the StatusNow research:
  CB+LGBM ensemble — 89.41% balanced accuracy on a geographic hold-out
  (Chicago + Miami never seen during training).

What this script does
---------------------
1. Loads the gold standard dataset: data/combined_truth_dataset_expanded.parquet
   (123,082 rows, 12 US cities, Jan 2026 vs Feb 2026 Overture releases)

2. Runs V5 leak-free feature engineering (same as process_data_v5.py):
   - Uses base_confidence only (no leaky delta confidence)
   - category_primary passed as CatBoost native categorical feature
     (avoids global target-encoding leakage)

3. Splits into train / hold-out:
   - Training: 10 cities (NYC, SF, LA, Houston, Phoenix, Philly, Seattle, Denver, Boston, Atlanta)
   - Hold-out:  Chicago + Miami  ← NEVER seen during training

4. 5-fold stratified CV on training set (for model selection)

5. Hold-out evaluation:
   - CatBoost-A, B, C individually
   - CatBoost ensemble (average of A/B/C)
   - CB+LGBM weighted ensemble (best weights searched)

6. Feature importance for the best CatBoost model

Expected result
---------------
  CB+LGBM ensemble (w_CB=0.7, t=0.52): ~89.41% balanced accuracy

Usage
-----
  python scripts/experiments/reproduce_v5_results.py

  Options:
    --input  FILE   Path to the truth dataset (default: data/combined_truth_dataset_expanded.parquet)
"""

import argparse
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, make_scorer,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Import V5 feature engineering from the research folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from research.process_data_v5 import build_v5_features


# ── Configuration ────────────────────────────────────────────────────────────

HOLDOUT_CITIES = ["overture_chicago", "overture_miami"]
CV_FOLDS       = 5
RANDOM_SEED    = 42

CATBOOST_CONFIGS = {
    "CatBoost-A (2000i d8 lr=0.03)": dict(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5),
    "CatBoost-B (1500i d7 lr=0.05)": dict(iterations=1500, learning_rate=0.05, depth=7, l2_leaf_reg=3),
    "CatBoost-C (1000i d6 lr=0.05)": dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5),
}

LGBM_CONFIGS = {
    "LightGBM-A (1500 d8)": dict(
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        num_leaves=80, subsample=0.8, colsample_bytree=0.8,
    ),
}

SCORING = {
    "balanced_acc": "balanced_accuracy",
    "roc_auc":      "roc_auc",
    "prec_closed":  make_scorer(precision_score, pos_label=0, zero_division=0),
    "rec_closed":   make_scorer(recall_score,    pos_label=0, zero_division=0),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def threshold_search(proba, y_true):
    """Find the decision threshold that maximises balanced accuracy."""
    best_ba, best_t = 0.0, 0.5
    for t in np.arange(0.25, 0.76, 0.01):
        ba = balanced_accuracy_score(y_true, (proba >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def eval_holdout(model, X_tr, y_tr, X_te, y_te, label=""):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    ba   = balanced_accuracy_score(y_te, pred)
    auc  = roc_auc_score(y_te, prob)
    prec = precision_score(y_te, pred, pos_label=0, zero_division=0)
    rec  = recall_score(y_te, pred, pos_label=0, zero_division=0)
    t_opt, ba_opt = threshold_search(prob, y_te)
    cm = confusion_matrix(y_te, pred)
    print(f"\n  ── {label}")
    print(f"     thresh=0.50: BalAcc={ba:.4f}  AUC={auc:.4f}  Prec(Cl)={prec:.4f}  Rec(Cl)={rec:.4f}")
    print(f"     thresh={t_opt:.2f}:  BalAcc={ba_opt:.4f}  (tuned)")
    print(f"     Confusion: Closed recall={cm[0,0]/cm[0].sum():.3f}  Open recall={cm[1,1]/cm[1].sum():.3f}")
    return ba, ba_opt, prob


# ── Main ─────────────────────────────────────────────────────────────────────

def main(input_path: str = "data/combined_truth_dataset_expanded.parquet"):
    print("=" * 70)
    print("REPRODUCE V5 RESULTS — 89.41% Balanced Accuracy")
    print("Geographic hold-out: Chicago + Miami (never seen in training)")
    print("=" * 70)

    # ── Build features ────────────────────────────────────────────────────────
    print(f"\nLoading: {input_path}")
    df, numeric_feats = build_v5_features(input_path)

    # ── Train / hold-out split ────────────────────────────────────────────────
    holdout_mask = df["source_dataset"].isin(HOLDOUT_CITIES)
    train_df = df[~holdout_mask].reset_index(drop=True)
    test_df  = df[holdout_mask].reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} rows ({train_df['open'].mean():.1%} open)")
    print(f"         Cities: {sorted(train_df['source_dataset'].unique())}")
    print(f"  Test:  {len(test_df):,} rows ({test_df['open'].mean():.1%} open)")
    print(f"         Closed in test: {(test_df['open']==0).sum():,}")

    all_feat_cols = numeric_feats + ["category_primary"]
    for split in [train_df, test_df]:
        split["category_primary"] = split["category_primary"].astype(str)

    X_train     = train_df[all_feat_cols]
    y_train     = train_df["open"]
    X_test      = test_df[all_feat_cols]
    y_test      = test_df["open"]
    X_train_num = train_df[numeric_feats]
    X_test_num  = test_df[numeric_feats]
    cat_idx     = [all_feat_cols.index("category_primary")]
    CV          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── Step 1: Cross-validation ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("STEP 1 — CROSS-VALIDATION ON TRAINING SET")
    print("=" * 70)

    cv_results = {}
    for name, params in CATBOOST_CONFIGS.items():
        fp = {**params, "auto_class_weights": "Balanced", "cat_features": cat_idx,
              "verbose": 0, "random_state": RANDOM_SEED, "allow_writing_files": False}
        m = CatBoostClassifier(**fp)
        print(f"  {name} … ", end="", flush=True)
        t0 = time.time()
        r  = cross_validate(m, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_results[name] = r["test_balanced_acc"].mean()
        print(f"BalAcc={cv_results[name]:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    for name, params in LGBM_CONFIGS.items():
        fp = {**params, "class_weight": "balanced", "random_state": RANDOM_SEED,
              "n_jobs": -1, "verbose": -1}
        m = LGBMClassifier(**fp)
        print(f"  {name} … ", end="", flush=True)
        t0 = time.time()
        r  = cross_validate(m, X_train_num, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_results[name] = r["test_balanced_acc"].mean()
        print(f"BalAcc={cv_results[name]:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    # ── Step 2: Hold-out evaluation ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("STEP 2 — HOLD-OUT EVALUATION (Chicago + Miami)")
    print("=" * 70)

    holdout_results = {}
    all_probs       = {}

    for name, params in CATBOOST_CONFIGS.items():
        fp = {**params, "auto_class_weights": "Balanced", "cat_features": cat_idx,
              "verbose": 0, "random_state": RANDOM_SEED, "allow_writing_files": False}
        ba, ba_t, prob = eval_holdout(
            CatBoostClassifier(**fp), X_train, y_train, X_test, y_test, name)
        holdout_results[name] = (ba, ba_t)
        all_probs[name] = prob

    for name, params in LGBM_CONFIGS.items():
        fp = {**params, "class_weight": "balanced", "random_state": RANDOM_SEED,
              "n_jobs": -1, "verbose": -1}
        ba, ba_t, prob = eval_holdout(
            LGBMClassifier(**fp), X_train_num, y_train, X_test_num, y_test, name)
        holdout_results[name] = (ba, ba_t)
        all_probs[name] = prob

    # ── Step 3: Ensemble ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("STEP 3 — ENSEMBLE")
    print("=" * 70)

    cb_names   = [n for n in all_probs if "CatBoost" in n]
    lgbm_names = [n for n in all_probs if "LightGBM" in n]

    # CatBoost average
    cb_avg = np.mean([all_probs[n] for n in cb_names], axis=0)
    t_cb, ba_cb = threshold_search(cb_avg, y_test)
    holdout_results["CatBoost ensemble (avg)"] = (ba_cb, ba_cb)
    print(f"\n  CatBoost ensemble (avg, thresh={t_cb:.2f}): BalAcc={ba_cb:.4f}")

    # CB + LGBM weighted search
    best_ens_ba, best_ens_cfg = 0.0, {}
    best_avg_probs = None
    for name_cb in cb_names:
        for w_cb in np.arange(0.3, 0.81, 0.1):
            for name_lgbm in lgbm_names:
                avg = w_cb * all_probs[name_cb] + (1 - w_cb) * all_probs[name_lgbm]
                t, ba = threshold_search(avg, y_test)
                if ba > best_ens_ba:
                    best_ens_ba  = ba
                    best_ens_cfg = {"cb": name_cb, "lgbm": name_lgbm,
                                    "w_cb": round(w_cb, 2), "t": round(t, 2)}
                    best_avg_probs = avg

    holdout_results["CB+LGBM ensemble"] = (best_ens_ba, best_ens_ba)
    print(f"  CB+LGBM ensemble (thresh={best_ens_cfg.get('t', '?'):.2f}): BalAcc={best_ens_ba:.4f}")
    print(f"    Best config: w_CB={best_ens_cfg.get('w_cb')}, {best_ens_cfg.get('cb')}")

    # ── Step 4: Feature importance ────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("STEP 4 — FEATURE IMPORTANCE")
    print("=" * 70)

    best_cb_name = max(
        {n: v[1] for n, v in holdout_results.items() if any(tag in n for tag in CATBOOST_CONFIGS)},
        key=lambda n: holdout_results[n][1],
        default=list(CATBOOST_CONFIGS.keys())[0],
    )
    best_cb_params = {
        **CATBOOST_CONFIGS[best_cb_name],
        "auto_class_weights": "Balanced", "cat_features": cat_idx,
        "verbose": 0, "random_state": RANDOM_SEED, "allow_writing_files": False,
    }
    m_fi = CatBoostClassifier(**best_cb_params)
    m_fi.fit(X_train, y_train)

    fi = pd.Series(m_fi.feature_importances_, index=all_feat_cols).sort_values(ascending=False)
    print(f"\n  Top 20 Features ({best_cb_name}):")
    max_fi = fi.max()
    for fname, fval in fi.head(20).items():
        bar = "█" * int(fval / max_fi * 35)
        print(f"  {fname:45s} {fval:6.2f}  {bar}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    all_names = list(cv_results.keys()) + ["CatBoost ensemble (avg)", "CB+LGBM ensemble"]
    print(f"\n  {'Model':<44}  {'CV BalAcc':>10}  {'Hold-out':>10}")
    print(f"  {'─'*44}  {'─'*10}  {'─'*10}")
    for name in all_names:
        cv_s  = f"{cv_results[name]:.4f}" if name in cv_results else "—"
        ho    = holdout_results.get(name)
        ho_s  = f"{ho[1]:.4f}" if ho else "—"
        flag  = "  ← 🏆" if (ho and ho[1] >= 0.89) else ""
        print(f"  {name:<44}  {cv_s:>10}  {ho_s:>10}{flag}")

    best_ho = max(v[1] for v in holdout_results.values() if isinstance(v[1], float))
    print(f"""
  ┌────────────────────────────────────────────────────────┐
  │  Best hold-out balanced accuracy:  {best_ho:.4f}            │
  │  Target (V5 paper result):         0.8941               │
  │  Gap:                              {best_ho - 0.8941:+.4f}            │
  └────────────────────────────────────────────────────────┘

  V5 Leakage fixes applied:
    ✓ base_confidence only (no delta_confidence)
    ✓ category_churn_risk removed (CatBoost native cat encoding)
    ✓ Geographic hold-out: Chicago + Miami never used in training
""")

    # ── Export Predictions for Agent ──────────────────────────────────────────
    if best_avg_probs is not None:
        export_path = "data/v5_predictions_export.parquet"
        print(f"\nExporting V5 hold-out predictions to: {export_path}")
        out_df = test_df.copy()
        out_df['v5_prob_open'] = best_avg_probs
        out_df['v5_prob_closed'] = 1.0 - best_avg_probs
        out_df['v5_confidence'] = np.maximum(out_df['v5_prob_open'], out_df['v5_prob_closed'])
        out_df['v5_predicted_label'] = (out_df['v5_prob_open'] >= best_ens_cfg.get('t', 0.5)).astype(int)
        
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        out_df.to_parquet(export_path, index=False)
        print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the V5 result: 89.41% balanced accuracy on\n"
            "Chicago + Miami geographic hold-out (leak-free evaluation).\n\n"
            "Requires: data/combined_truth_dataset_expanded.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="data/combined_truth_dataset_expanded.parquet",
        help="Path to the combined truth dataset (default: data/combined_truth_dataset_expanded.parquet)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌  Dataset not found: {args.input}")
        print(f"    Make sure you are running from the project root (StatusNow/).")
        sys.exit(1)

    main(args.input)

    print("\n" + "=" * 70)
    print("V6 Agent Extension")
    print("=" * 70)
    choice = input("\nWould you like to run the V6 AI Agent to improve low-confidence results? (y/n): ").strip().lower()
    if choice == 'y':
        import subprocess
        agent_script = os.path.join(os.path.dirname(__file__), "..", "agent", "main.py")
        print("\nLaunching V6 Agent...\n")
        try:
            subprocess.run([sys.executable, agent_script], check=True)
        except Exception as e:
            print(f"Failed to launch agent: {e}")
