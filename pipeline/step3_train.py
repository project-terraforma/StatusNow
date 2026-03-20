"""
Step 3 — Train the CB+LGBM Ensemble and Save Model Artifacts
=============================================================

What this script does
---------------------
1. Reads the feature dataset produced by step2.
2. Fits PCA (on training rows only) to produce recency_pca, then transforms
   hold-out rows with the already-fitted PCA — no test-into-training leakage.
3. Fits a LabelEncoder (on training rows only) for category_primary so that
   LightGBM can consume the category feature.
4. Optionally holds out cities specified via --holdout-cities for an honest
   geographic evaluation (same methodology as the V5 research paper).
5. Trains CatBoost (3 configs) and LightGBM using out-of-fold (OOF)
   cross-validation to collect per-model probability estimates.
6. Searches for the best ensemble weights (CB + LGBM) and decision threshold
   using OOF predictions on the TRAINING set — the hold-out set is never
   touched during optimisation.
7. Trains final models on the full training set.
8. Evaluates on the hold-out with the pre-determined (OOF-derived) weights and
   threshold — a clean, unbiased estimate of generalisation.
9. Saves all model artifacts to pipeline_output/models/:
     catboost_a.cbm, catboost_b.cbm, catboost_c.cbm
     lightgbm_a.pkl
     pca_recency.pkl          ← fitted PCA for inference
     label_encoder_cat.pkl    ← fitted LabelEncoder for inference
     ensemble_config.json     ← weights, threshold, feature list, accuracy

Output
------
  pipeline_output/models/  (model files + ensemble_config.json)

Usage
-----
  python pipeline/step3_train.py [options]

  Options:
    --input           FILE      Feature dataset (default: pipeline_output/02_features.parquet)
    --output-dir      DIR       Where to write model files (default: pipeline_output/models/)
    --holdout-cities  CITY ...  City names to hold out for evaluation.
                                 Example: --holdout-cities chicago miami
    --cv-folds        N         Number of CV folds (default: 5)
    --seed            N         Random seed (default: 42)
"""

import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ─── model configurations ────────────────────────────────────────────────────

CATBOOST_CONFIGS = {
    "CatBoost-A": dict(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5),
    "CatBoost-B": dict(iterations=1500, learning_rate=0.05, depth=7, l2_leaf_reg=3),
    "CatBoost-C": dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5),
}

LGBM_CONFIGS = {
    "LightGBM-A": dict(
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        num_leaves=80, subsample=0.8, colsample_bytree=0.8,
    ),
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _threshold_search(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Find the decision threshold that maximises balanced accuracy."""
    best_ba, best_t = 0.0, 0.5
    for t in np.arange(0.25, 0.76, 0.01):
        ba = balanced_accuracy_score(y_true, (proba >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def _print_bar(value: float, max_value: float, width: int = 35) -> str:
    n = int(value / max_value * width) if max_value > 0 else 0
    return "█" * n


# ─── main training function ──────────────────────────────────────────────────

def train(
    input_path:      str   = "pipeline_output/02_features.parquet",
    output_dir:      str   = "pipeline_output/models",
    holdout_cities:  "list[str] | None" = None,
    cv_folds:        int   = 5,
    seed:            int   = 42,
) -> dict:
    """
    Train the ensemble and save models.

    Parameters
    ----------
    input_path      : path to 02_features.parquet
    output_dir      : where to save model files
    holdout_cities  : list of city names to hold out.
                      If None, no geographic hold-out; only OOF accuracy reported.
    cv_folds        : number of stratified CV folds
    seed            : random seed for reproducibility

    Returns
    -------
    dict with keys: best_oof_bal_acc, best_holdout_bal_acc (if holdout), models_dir
    """
    print("\n" + "=" * 70)
    print("STEP 3 — TRAIN CB+LGBM ENSEMBLE")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_parquet(input_path)
    print(f"\n  Loaded {len(df):,} rows | {df.shape[1]} columns")

    # days_latest / days_avg are PCA inputs, not model features.
    # category_primary is handled natively by CatBoost and label-encoded for LGBM.
    non_feat_cols = {
        "open", "city", "release_pair", "release_date_base",
        "release_date_current", "release_index", "category_primary",
        "days_latest", "days_avg",   # ← PCA inputs, excluded from raw feature list
    }
    numeric_feats = [c for c in df.columns if c not in non_feat_cols]

    df["category_primary"] = df["category_primary"].astype(str)

    # ── Train / hold-out split ────────────────────────────────────────────────
    if holdout_cities:
        city_col = "city" if "city" in df.columns else "release_date_current"
        mask = df[city_col].isin(holdout_cities)
        train_df = df[~mask].reset_index(drop=True)
        test_df  = df[mask].reset_index(drop=True)
        print(f"\n  Hold-out cities:  {holdout_cities}  (filtering on '{city_col}')")
        print(f"  Train: {len(train_df):,} rows ({train_df['open'].mean():.1%} open)")
        print(f"  Test:  {len(test_df):,}  rows ({test_df['open'].mean():.1%} open)")
    else:
        train_df = df.reset_index(drop=True)
        test_df  = None
        print(f"\n  No hold-out cities — training on full dataset, OOF CV only.")
        print(f"  Train: {len(train_df):,} rows ({train_df['open'].mean():.1%} open)")

    # ── FIX: PCA fitted on training rows only ────────────────────────────────
    # recency_pca was intentionally deferred from step2 so PCA is never
    # contaminated by hold-out data.
    pca_cols = ["days_latest", "days_avg"]
    pca = PCA(n_components=1)
    train_rec = train_df[pca_cols].fillna(9999).values
    pca.fit(train_rec)

    train_df = train_df.copy()
    train_df["recency_pca"] = pca.transform(train_rec).flatten()

    if test_df is not None:
        test_df = test_df.copy()
        test_df["recency_pca"] = pca.transform(
            test_df[pca_cols].fillna(9999).values
        ).flatten()

    numeric_feats = numeric_feats + ["recency_pca"]

    # ── FIX: LabelEncoder for category_primary (fitted on training only) ─────
    # LightGBM cannot handle raw string categoricals; we label-encode on
    # training rows and map hold-out rows, using -1 for unseen categories.
    le = LabelEncoder()
    le.fit(train_df["category_primary"].astype(str))
    train_df["category_encoded"] = le.transform(
        train_df["category_primary"].astype(str)
    )
    if test_df is not None:
        def _safe_encode(val: str) -> int:
            return int(le.transform([val])[0]) if val in le.classes_ else -1
        test_df["category_encoded"] = (
            test_df["category_primary"].astype(str).apply(_safe_encode)
        )

    # Feature matrices
    all_feat_cols   = numeric_feats + ["category_primary"]
    cat_idx         = [all_feat_cols.index("category_primary")]
    lgbm_feat_cols  = numeric_feats + ["category_encoded"]   # LGBM gets encoded cat

    X_train      = train_df[all_feat_cols].copy()
    y_train      = train_df["open"].values
    X_train_lgbm = train_df[lgbm_feat_cols].copy()

    if test_df is not None and len(test_df) > 0:
        X_test      = test_df[all_feat_cols].copy()
        y_test      = test_df["open"].values
        X_test_lgbm = test_df[lgbm_feat_cols].copy()
    else:
        X_test = y_test = X_test_lgbm = None

    CV = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # ── Step A: OOF cross-validation ─────────────────────────────────────────
    # cross_val_predict returns out-of-fold probabilities fitted entirely on
    # training data.  These OOF probs are used for ensemble weight + threshold
    # search so that the hold-out set is never touched during optimisation.
    print("\n" + "─" * 70)
    print("  A. OOF CROSS-VALIDATION (training set — hold-out never seen)")
    print("─" * 70)

    oof_probs:    dict[str, np.ndarray] = {}
    oof_thresholds: dict[str, float]   = {}
    cv_results:   dict[str, float]     = {}

    for name, params in CATBOOST_CONFIGS.items():
        full_params = {
            **params,
            "auto_class_weights": "Balanced",
            "cat_features": cat_idx,
            "verbose": 0,
            "random_state": seed,
            "allow_writing_files": False,
        }
        m = CatBoostClassifier(**full_params)
        print(f"  {name} OOF … ", end="", flush=True)
        t0 = time.time()
        # n_jobs=1: let CatBoost use its own internal threading per fold
        oof = cross_val_predict(
            m, X_train, y_train, cv=CV, method="predict_proba", n_jobs=1
        )[:, 1]
        oof_probs[name] = oof
        t_oof, ba_oof = _threshold_search(oof, y_train)
        auc_oof = roc_auc_score(y_train, oof)
        oof_thresholds[name] = t_oof
        cv_results[name] = ba_oof
        print(
            f"OOF BalAcc={ba_oof:.4f}  AUC={auc_oof:.4f}  "
            f"thresh={t_oof:.2f}  ({time.time()-t0:.0f}s)"
        )

    for name, params in LGBM_CONFIGS.items():
        full_params = {
            **params,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        m = LGBMClassifier(**full_params)
        print(f"  {name} OOF … ", end="", flush=True)
        t0 = time.time()
        oof = cross_val_predict(
            m, X_train_lgbm, y_train, cv=CV, method="predict_proba", n_jobs=1
        )[:, 1]
        oof_probs[name] = oof
        t_oof, ba_oof = _threshold_search(oof, y_train)
        auc_oof = roc_auc_score(y_train, oof)
        oof_thresholds[name] = t_oof
        cv_results[name] = ba_oof
        print(
            f"OOF BalAcc={ba_oof:.4f}  AUC={auc_oof:.4f}  "
            f"thresh={t_oof:.2f}  ({time.time()-t0:.0f}s)"
        )

    best_cv = max(cv_results, key=cv_results.get)
    print(f"\n  Best OOF: {best_cv}  ({cv_results[best_cv]:.4f})")

    # ── Step B: Ensemble weight + threshold search on OOF (y_train) ──────────
    # FIX: ensemble weights and threshold are chosen using OOF predictions on
    # y_train.  The hold-out set plays NO role here — reported hold-out numbers
    # are therefore a clean, unbiased evaluation.
    print("\n" + "─" * 70)
    print("  B. ENSEMBLE SEARCH (OOF predictions on training set)")
    print("─" * 70)

    cb_names   = [n for n in oof_probs if "CatBoost"  in n]
    lgbm_names = [n for n in oof_probs if "LightGBM"  in n]

    best_ens_ba  = 0.0
    best_ens_cfg: dict = {}

    for name_cb in cb_names:
        for w_cb in np.arange(0.3, 0.81, 0.1):
            for name_lgbm in lgbm_names:
                w_lgbm = 1.0 - w_cb
                avg = w_cb * oof_probs[name_cb] + w_lgbm * oof_probs[name_lgbm]
                t, ba = _threshold_search(avg, y_train)   # ← y_train, NOT y_test
                if ba > best_ens_ba:
                    best_ens_ba  = ba
                    best_ens_cfg = {
                        "cb_model":   name_cb,
                        "lgbm_model": name_lgbm,
                        "w_cb":       round(float(w_cb), 2),
                        "w_lgbm":     round(float(w_lgbm), 2),
                        "threshold":  round(float(t), 2),
                    }

    print(f"  Best OOF ensemble BalAcc: {best_ens_ba:.4f}")
    print(f"  Config: {best_ens_cfg}")

    # ── Step C: Train final models on full training set ───────────────────────
    print("\n" + "─" * 70)
    print("  C. FINAL MODEL TRAINING (full training set)")
    print("─" * 70)

    trained_cb:   dict[str, CatBoostClassifier] = {}
    trained_lgbm: dict[str, LGBMClassifier]     = {}
    holdout_probs: dict[str, np.ndarray]         = {}

    for name, params in CATBOOST_CONFIGS.items():
        full_params = {
            **params,
            "auto_class_weights": "Balanced",
            "cat_features": cat_idx,
            "verbose": 0,
            "random_state": seed,
            "allow_writing_files": False,
        }
        print(f"  Training {name} … ", end="", flush=True)
        t0 = time.time()
        m = CatBoostClassifier(**full_params)
        m.fit(X_train, y_train)
        trained_cb[name] = m
        print(f"done ({time.time()-t0:.0f}s)")

        if X_test is not None:
            holdout_probs[name] = m.predict_proba(X_test)[:, 1]

    for name, params in LGBM_CONFIGS.items():
        full_params = {
            **params,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        print(f"  Training {name} … ", end="", flush=True)
        t0 = time.time()
        m = LGBMClassifier(**full_params)
        m.fit(X_train_lgbm, y_train)
        trained_lgbm[name] = m
        print(f"done ({time.time()-t0:.0f}s)")

        if X_test is not None:
            holdout_probs[name] = m.predict_proba(X_test_lgbm)[:, 1]

    # ── Step D: Hold-out evaluation (pre-determined config, no tuning) ────────
    print("\n" + "─" * 70)
    print("  D. HOLD-OUT EVALUATION (pre-determined OOF weights — unbiased)")
    print("─" * 70)

    holdout_results: dict[str, float] = {}

    if X_test is not None and holdout_probs:
        # Individual models — use each model's OOF-derived threshold
        for name, prob in holdout_probs.items():
            thresh = oof_thresholds.get(name, 0.5)
            pred   = (prob >= thresh).astype(int)
            ba     = balanced_accuracy_score(y_test, pred)
            holdout_results[name] = ba
            print(f"  {name:<25}  BalAcc={ba:.4f}  (OOF thresh={thresh:.2f})")

        # Ensemble — use OOF-derived weights + threshold (no search on hold-out)
        name_cb    = best_ens_cfg["cb_model"]
        name_lgbm  = best_ens_cfg["lgbm_model"]
        w_cb       = best_ens_cfg["w_cb"]
        w_lgbm     = best_ens_cfg["w_lgbm"]
        threshold  = best_ens_cfg["threshold"]

        avg  = w_cb * holdout_probs[name_cb] + w_lgbm * holdout_probs[name_lgbm]
        pred = (avg >= threshold).astype(int)
        ba_ens = balanced_accuracy_score(y_test, pred)
        holdout_results["CB+LGBM ensemble"] = ba_ens
        print(
            f"\n  CB+LGBM ensemble   BalAcc={ba_ens:.4f}  "
            f"(w_cb={w_cb}, w_lgbm={w_lgbm}, thresh={threshold})"
        )
    else:
        print("  No hold-out set — skipping hold-out evaluation.")

    # ── Step E: Feature importance ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  E. FEATURE IMPORTANCE (CatBoost-C)")
    print("─" * 70)

    m_fi = trained_cb.get("CatBoost-C") or list(trained_cb.values())[0]
    fi = pd.Series(
        m_fi.feature_importances_,
        index=all_feat_cols,
    ).sort_values(ascending=False)

    max_fi = fi.max()
    print(f"\n  Top 20 features:")
    for fname, fval in fi.head(20).items():
        bar = _print_bar(fval, max_fi)
        print(f"    {fname:45s} {fval:6.2f}  {bar}")

    # ── Step F: Save models + artifacts ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  F. SAVING MODELS")
    print("─" * 70)

    os.makedirs(output_dir, exist_ok=True)

    for name, model in trained_cb.items():
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".cbm"
        path  = os.path.join(output_dir, fname)
        model.save_model(path)
        print(f"  Saved: {path}")

    for name, model in trained_lgbm.items():
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".pkl"
        path  = os.path.join(output_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {path}")

    # Save PCA and LabelEncoder for inference-time use
    pca_path = os.path.join(output_dir, "pca_recency.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  Saved: {pca_path}")

    le_path = os.path.join(output_dir, "label_encoder_cat.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  Saved: {le_path}")

    ensemble_config = {
        "numeric_features":     numeric_feats,
        "all_features":         all_feat_cols,
        "lgbm_features":        lgbm_feat_cols,
        "cat_features_idx":     cat_idx,
        "category_classes":     le.classes_.tolist(),
        "oof_cv_results":       {k: float(v) for k, v in cv_results.items()},
        "oof_thresholds":       {k: float(v) for k, v in oof_thresholds.items()},
        "oof_ensemble_bal_acc": float(best_ens_ba),
        "holdout_ensemble":     best_ens_cfg,
        "best_holdout_bal_acc": (
            float(holdout_results["CB+LGBM ensemble"])
            if "CB+LGBM ensemble" in holdout_results else None
        ),
        "holdout_individual":   {k: float(v) for k, v in holdout_results.items()},
        "feature_importance":   {k: float(v) for k, v in fi.items()},
    }

    config_path = os.path.join(output_dir, "ensemble_config.json")
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    all_names = list(cv_results.keys()) + ["CB+LGBM ensemble"]
    print(f"\n  {'Model':<40}  {'OOF BalAcc':>10}  {'Hold-out':>10}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*10}")

    for name in all_names:
        oof_s = f"{cv_results[name]:.4f}" if name in cv_results else "—"
        ho    = holdout_results.get(name)
        ho_s  = f"{ho:.4f}" if ho is not None else "—"
        flag  = "  ← best" if (ho is not None and ho >= max(holdout_results.values(), default=0)) else ""
        print(f"  {name:<40}  {oof_s:>10}  {ho_s:>10}{flag}")

    best_oof = max(cv_results.values())
    print(f"\n  Best OOF balanced accuracy:     {best_oof:.4f}")
    if holdout_results:
        best_ho = max(holdout_results.values())
        print(f"  Best hold-out balanced acc:     {best_ho:.4f}  ← honest (OOF-tuned config)")

    print(f"\n  Models saved to: {output_dir}/")

    return {
        "models_dir":           output_dir,
        "best_oof_bal_acc":     best_oof,
        "best_holdout_bal_acc": max(holdout_results.values()) if holdout_results else None,
    }


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: Train the CB+LGBM ensemble and save model artifacts.\n"
            "Run after step2_feature_engineering.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="pipeline_output/02_features.parquet",
        help="Feature dataset (default: pipeline_output/02_features.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_output/models",
        help="Directory for saved model files (default: pipeline_output/models/)",
    )
    parser.add_argument(
        "--holdout-cities",
        nargs="+",
        default=None,
        metavar="CITY",
        help=(
            "City names to hold out for geographic evaluation. "
            "E.g.: --holdout-cities chicago miami"
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    train(
        input_path=args.input,
        output_dir=args.output_dir,
        holdout_cities=args.holdout_cities,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
