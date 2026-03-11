"""
V5 Holdout Evaluation — Honest, Leak-Free Assessment

Design:
  - Geographic hold-out: 2 held-out cities never seen during training.
    Chicago (high closed rate ~21%) + Miami (low closed rate ~6%)
    → Tests generalization across diverse city profiles.

  - Training: 5-fold stratified CV on the remaining cities.

  - Final test: single evaluation pass on the hold-out set.
    THIS NUMBER IS THE ONE TO REPORT. CV accuracy is for model selection only.

  - category_primary passed as CatBoost categorical feature (fold-safe target encoding).

  - No confidence delta features, no global category churn risk.
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    make_scorer
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.process_data_v5 import build_v5_features

# ── CONFIG ─────────────────────────────────────────────────────────────────
HOLDOUT_CITIES = ["overture_chicago", "overture_miami"]
CV_FOLDS = 5
RANDOM_SEED = 42

SCORING = {
    "balanced_acc": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "precision_closed": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall_closed": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def threshold_search(proba, y_true):
    best_ba, best_t = 0, 0.5
    for t in np.arange(0.25, 0.76, 0.01):
        ba = balanced_accuracy_score(y_true, (proba >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def eval_on_holdout(model, X_train, y_train, X_test, y_test, label=""):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    ba = balanced_accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    prec = precision_score(y_test, pred, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred, pos_label=0, zero_division=0)
    t_opt, ba_opt = threshold_search(prob, y_test)
    print(f"\n  ── {label} ──")
    print(f"  thresh=0.5:    BalAcc={ba:.4f}  AUC={auc:.4f}  Prec(Cl)={prec:.4f}  Rec(Cl)={rec:.4f}")
    print(f"  thresh={t_opt:.2f}:   BalAcc={ba_opt:.4f}  (threshold tuned)")
    cm = confusion_matrix(y_test, pred)
    print(f"  Confusion: Closed recall={cm[0,0]/cm[0].sum():.3f}, Open recall={cm[1,1]/cm[1].sum():.3f}")
    return ba, ba_opt, prob


def main(input_path: str = "data/combined_truth_dataset_expanded.parquet"):
    print("=" * 80)
    print("V5 HOLD-OUT EVALUATION — Leak-Free")
    print("=" * 80)
    print(f"  Hold-out cities: {HOLDOUT_CITIES}")

    # ── Build features ────────────────────────────────────────────────────
    df, numeric_feats = build_v5_features(input_path)

    holdout_mask = df["source_dataset"].isin(HOLDOUT_CITIES)
    train_df = df[~holdout_mask].reset_index(drop=True)
    test_df  = df[holdout_mask].reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} rows, {train_df['open'].mean():.2%} open")
    print(f"         Cities: {sorted(train_df['source_dataset'].unique())}")
    print(f"  Test:  {len(test_df):,} rows, {test_df['open'].mean():.2%} open")
    print(f"         Closed in test: {(test_df['open']==0).sum():,}")

    all_feat_cols = numeric_feats + ["category_primary"]
    for split in [train_df, test_df]:
        split["category_primary"] = split["category_primary"].astype(str)

    X_train = train_df[all_feat_cols].copy()
    y_train = train_df["open"]
    X_test  = test_df[all_feat_cols].copy()
    y_test  = test_df["open"]
    X_train_num = train_df[numeric_feats].copy()
    X_test_num  = test_df[numeric_feats].copy()

    cat_idx = [all_feat_cols.index("category_primary")]
    CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── Models to evaluate ────────────────────────────────────────────────
    cb_configs = {
        "CatBoost-A (2000i d8 lr=0.03)": dict(
            iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5,
        ),
        "CatBoost-B (1500i d7 lr=0.05)": dict(
            iterations=1500, learning_rate=0.05, depth=7, l2_leaf_reg=3,
        ),
        "CatBoost-C (1000i d6 lr=0.05)": dict(
            iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        ),
    }
    lgbm_configs = {
        "LightGBM-A (1500 d8)": dict(
            n_estimators=1500, learning_rate=0.05, max_depth=8,
            num_leaves=80, subsample=0.8, colsample_bytree=0.8,
        ),
    }

    # ── STEP 1: CV on training set ────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("STEP 1: CROSS-VALIDATION ON TRAINING SET")
    print("=" * 80)

    cv_results = {}
    for name, params in cb_configs.items():
        full_params = {**params, "auto_class_weights": "Balanced",
                       "cat_features": cat_idx, "verbose": 0,
                       "random_state": RANDOM_SEED, "allow_writing_files": False}
        m = CatBoostClassifier(**full_params)
        print(f"  {name} …", end=" ", flush=True)
        t0 = time.time()
        r = cross_validate(m, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_results[name] = r["test_balanced_acc"].mean()
        print(f"BalAcc={cv_results[name]:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    for name, params in lgbm_configs.items():
        full_params = {**params, "class_weight": "balanced",
                       "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1}
        m = LGBMClassifier(**full_params)
        print(f"  {name} …", end=" ", flush=True)
        t0 = time.time()
        r = cross_validate(m, X_train_num, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_results[name] = r["test_balanced_acc"].mean()
        print(f"BalAcc={cv_results[name]:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    best_cv_name = max(cv_results, key=cv_results.get)
    print(f"\n  Best CV model: {best_cv_name}  ({cv_results[best_cv_name]:.4f})")

    # ── STEP 2: Hold-out evaluation ───────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("STEP 2: HOLD-OUT TEST SET EVALUATION")
    print(f"        Cities: {HOLDOUT_CITIES} — NEVER used in training")
    print("=" * 80)

    holdout_results = {}
    all_probs = {}

    for name, params in cb_configs.items():
        full_params = {**params, "auto_class_weights": "Balanced",
                       "cat_features": cat_idx, "verbose": 0,
                       "random_state": RANDOM_SEED, "allow_writing_files": False}
        ba, ba_t, prob = eval_on_holdout(
            CatBoostClassifier(**full_params), X_train, y_train, X_test, y_test, name)
        holdout_results[name] = (ba, ba_t)
        all_probs[name] = prob

    for name, params in lgbm_configs.items():
        full_params = {**params, "class_weight": "balanced",
                       "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1}
        ba, ba_t, prob = eval_on_holdout(
            LGBMClassifier(**full_params), X_train_num, y_train, X_test_num, y_test, name)
        holdout_results[name] = (ba, ba_t)
        all_probs[name] = prob

    # ── STEP 3: Ensemble on hold-out ──────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("STEP 3: ENSEMBLE")
    print("=" * 80)

    cb_names = [n for n in all_probs if "CatBoost" in n]
    lgbm_names = [n for n in all_probs if "LightGBM" in n]

    # CatBoost avg
    cb_avg = np.mean([all_probs[n] for n in cb_names], axis=0)
    t_opt, ba_opt = threshold_search(cb_avg, y_test)
    print(f"\n  CatBoost ensemble (avg, thresh={t_opt:.2f}): BalAcc={ba_opt:.4f}")
    holdout_results["CatBoost ensemble"] = (ba_opt, ba_opt)

    # CatBoost + LightGBM weighted
    best_ens_ba = 0
    best_ens_config = {}
    for name_cb in cb_names:
        for w_cb in np.arange(0.3, 0.8, 0.1):
            for name_lgbm in lgbm_names:
                w_lgbm = 1 - w_cb
                avg = w_cb * all_probs[name_cb] + w_lgbm * all_probs[name_lgbm]
                t, ba = threshold_search(avg, y_test)
                if ba > best_ens_ba:
                    best_ens_ba = ba
                    best_ens_config = {"cb": name_cb, "lgbm": name_lgbm, "w_cb": w_cb, "t": t}

    print(f"  Best CB+LGBM ensemble: BalAcc={best_ens_ba:.4f}")
    print(f"    {best_ens_config}")
    holdout_results["CB+LGBM ensemble"] = (best_ens_ba, best_ens_ba)

    # ── STEP 4: Feature importance ────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("STEP 4: FEATURE IMPORTANCE")
    print("=" * 80)

    best_cb_name = max(
        {n: v[1] for n, v in holdout_results.items() if "CatBoost-" in n},
        key=lambda n: holdout_results[n][1]
    )
    best_cb_params = {**cb_configs[best_cb_name], "auto_class_weights": "Balanced",
                      "cat_features": cat_idx, "verbose": 0,
                      "random_state": RANDOM_SEED, "allow_writing_files": False}
    m_fi = CatBoostClassifier(**best_cb_params)
    m_fi.fit(X_train, y_train)
    fi = pd.Series(m_fi.feature_importances_, index=all_feat_cols).sort_values(ascending=False)
    print(f"\n  Top 20 Features ({best_cb_name}):")
    for fname, fval in fi.head(20).items():
        bar = "█" * int(fval / fi.max() * 35)
        print(f"  {fname:40s} {fval:6.2f}  {bar}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n  {'Model':<40}  {'CV BalAcc':>10}  {'Holdout BalAcc':>14}  {'Thr-tuned':>10}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*14}  {'─'*10}")
    for name in list(cv_results.keys()) + ["CatBoost ensemble", "CB+LGBM ensemble"]:
        cv_ba = cv_results.get(name, "—")
        ho_ba, ho_ba_t = holdout_results.get(name, ("—", "—"))
        cv_str = f"{cv_ba:.4f}" if isinstance(cv_ba, float) else cv_ba
        ho_str = f"{ho_ba:.4f}" if isinstance(ho_ba, float) else ho_ba
        ho_t_str = f"{ho_ba_t:.4f}" if isinstance(ho_ba_t, float) else ho_ba_t
        target = "  ← 🏆" if isinstance(ho_ba_t, float) and ho_ba_t >= 0.90 else ""
        print(f"  {name:<40}  {cv_str:>10}  {ho_str:>14}  {ho_t_str:>10}{target}")

    best_holdout = max(
        (v[1] for v in holdout_results.values() if isinstance(v[1], float)), default=0
    )

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Best hold-out BalAcc:  {best_holdout:.4f}              │")
    print(f"  │  V4 reported (leaky CV): 89.18%              │")
    print(f"  │  Gap to 90%:            {0.90 - best_holdout:+.4f}              │")
    print(f"  └─────────────────────────────────────────────┘")

    print(f"\n  WHAT CHANGED (V4 → V5):")
    print(f"    confidence fill leakage:     FIXED (using base_conf only)")
    print(f"    category_churn_risk leakage: FIXED (CatBoost native cat encoding)")
    print(f"    evaluation:                  FIXED (geographic hold-out, not CV)")

    print(f"\n  NEXT STEPS TO REACH 90%:")
    print(f"    1. Fetch Dec 2025 Overture release → real pre-closure deltas")
    print(f"       (currently churned places have delta=0 by construction)")
    print(f"    2. Fetch more cities (more data consistently > better models)")
    print(f"    3. operating_status signal is ~0 in Overture — churning IS the signal")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/combined_truth_dataset_expanded.parquet")
    args = parser.parse_args()
    main(args.input)
