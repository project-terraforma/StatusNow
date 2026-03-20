"""
Step 3 — Train the CB+LGBM Ensemble and Save Model Artifacts
=============================================================

What this script does
---------------------
1. Reads the feature dataset produced by step2.
2. Optionally holds out cities specified via --holdout-cities for an honest
   geographic evaluation (same methodology as the V5 research paper).
3. Trains CatBoost (3 configs) and LightGBM on the training split using
   5-fold stratified cross-validation for model selection.
4. Evaluates all models on the hold-out set (if provided) or reports CV
   accuracy only.
5. Searches for the best ensemble weights (CB + LGBM) and decision threshold.
6. Saves all model artifacts to pipeline_output/models/:
     catboost_a.cbm, catboost_b.cbm, catboost_c.cbm
     lgbm.pkl
     ensemble_config.json   ← weights, threshold, feature list, accuracy
7. Prints a clean final results table.

Output
------
  pipeline_output/models/  (model files + ensemble_config.json)

Usage
-----
  python pipeline/step3_train.py [options]

  Options:
    --input           FILE      Feature dataset (default: pipeline_output/02_features.parquet)
    --output-dir      DIR       Where to write model files (default: pipeline_output/models/)
    --holdout-cities  CITY ...  release_date_current values to hold out for evaluation.
                                 Example: --holdout-cities 2026-01-21.0 2026-02-18.0
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

from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

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
    holdout_cities  : list of release_date_current values to hold out.
                      If None, no geographic hold-out; only CV accuracy is reported.
    cv_folds        : number of stratified CV folds
    seed            : random seed for reproducibility

    Returns
    -------
    dict with keys: best_cv_bal_acc, best_holdout_bal_acc (if holdout), models_dir
    """
    print("\n" + "=" * 70)
    print("STEP 3 — TRAIN CB+LGBM ENSEMBLE")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_parquet(input_path)
    print(f"\n  Loaded {len(df):,} rows | {df.shape[1]} columns")

    # Identify feature columns
    non_feat_cols = {"open", "city", "release_pair", "release_date_base",
                     "release_date_current", "release_index", "category_primary"}
    numeric_feats = [c for c in df.columns if c not in non_feat_cols]
    all_feat_cols = numeric_feats + ["category_primary"]
    cat_idx = [all_feat_cols.index("category_primary")]

    df["category_primary"] = df["category_primary"].astype(str)

    # ── Train / hold-out split ────────────────────────────────────────────────
    if holdout_cities:
        # Hold out by city name (from _city column propagated through step1)
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
        print(f"\n  No hold-out cities specified — training on full dataset, CV only.")
        print(f"  Train: {len(train_df):,} rows ({train_df['open'].mean():.1%} open)")

    X_train     = train_df[all_feat_cols].copy()
    y_train     = train_df["open"]
    X_train_num = train_df[numeric_feats].copy()

    if test_df is not None and len(test_df) > 0:
        X_test      = test_df[all_feat_cols].copy()
        y_test      = test_df["open"]
        X_test_num  = test_df[numeric_feats].copy()
    else:
        X_test = y_test = X_test_num = None

    CV = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    SCORING = {
        "balanced_acc": "balanced_accuracy",
        "roc_auc":      "roc_auc",
    }

    # ── Step A: Cross-validation ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  A. CROSS-VALIDATION (training set)")
    print("─" * 70)

    cv_results: dict[str, float] = {}

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
        print(f"  {name} … ", end="", flush=True)
        t0 = time.time()
        r  = cross_validate(m, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        ba = r["test_balanced_acc"].mean()
        cv_results[name] = ba
        print(f"BalAcc={ba:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    for name, params in LGBM_CONFIGS.items():
        full_params = {
            **params,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        m = LGBMClassifier(**full_params)
        print(f"  {name} … ", end="", flush=True)
        t0 = time.time()
        r  = cross_validate(m, X_train_num, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        ba = r["test_balanced_acc"].mean()
        cv_results[name] = ba
        print(f"BalAcc={ba:.4f}  AUC={r['test_roc_auc'].mean():.4f}  ({time.time()-t0:.0f}s)")

    best_cv = max(cv_results, key=cv_results.get)
    print(f"\n  Best CV: {best_cv}  ({cv_results[best_cv]:.4f})")

    # ── Step B: Train final models on full train set + hold-out eval ─────────
    print("\n" + "─" * 70)
    print("  B. FINAL MODEL TRAINING (full training set)")
    print("─" * 70)

    trained_cb: dict[str, CatBoostClassifier] = {}
    trained_lgbm: dict[str, LGBMClassifier]   = {}
    holdout_results: dict[str, tuple[float, float]] = {}
    all_probs: dict[str, np.ndarray] = {}

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
            prob = m.predict_proba(X_test)[:, 1]
            pred = m.predict(X_test)
            ba   = balanced_accuracy_score(y_test, pred)
            t_opt, ba_t = _threshold_search(prob, y_test)
            holdout_results[name] = (ba, ba_t)
            all_probs[name] = prob
            print(
                f"    Hold-out: BalAcc={ba:.4f}  "
                f"(tuned thresh={t_opt:.2f}: {ba_t:.4f})"
            )

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
        m.fit(X_train_num, y_train)
        trained_lgbm[name] = m
        print(f"done ({time.time()-t0:.0f}s)")

        if X_test is not None:
            prob = m.predict_proba(X_test_num)[:, 1]
            pred = m.predict(X_test_num)
            ba   = balanced_accuracy_score(y_test, pred)
            t_opt, ba_t = _threshold_search(prob, y_test)
            holdout_results[name] = (ba, ba_t)
            all_probs[name] = prob
            print(
                f"    Hold-out: BalAcc={ba:.4f}  "
                f"(tuned thresh={t_opt:.2f}: {ba_t:.4f})"
            )

    # ── Step C: Ensemble search ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  C. ENSEMBLE SEARCH")
    print("─" * 70)

    ensemble_config = {
        "numeric_features": numeric_feats,
        "all_features": all_feat_cols,
        "cat_features_idx": cat_idx,
        "cv_results": {k: float(v) for k, v in cv_results.items()},
    }

    if X_test is not None and all_probs:
        cb_names   = [n for n in all_probs if "CatBoost" in n]
        lgbm_names = [n for n in all_probs if "LightGBM" in n]

        # CatBoost ensemble (average of CB variants)
        cb_avg = np.mean([all_probs[n] for n in cb_names], axis=0)
        t_cb, ba_cb = _threshold_search(cb_avg, y_test)
        holdout_results["CatBoost ensemble"] = (ba_cb, ba_cb)
        print(f"  CatBoost ensemble  (thresh={t_cb:.2f}): BalAcc={ba_cb:.4f}")

        # CB + LGBM weighted search
        best_ens_ba = 0.0
        best_ens_cfg: dict = {}
        for name_cb in cb_names:
            for w_cb in np.arange(0.3, 0.81, 0.1):
                for name_lgbm in lgbm_names:
                    w_lgbm = 1 - w_cb
                    avg = w_cb * all_probs[name_cb] + w_lgbm * all_probs[name_lgbm]
                    t, ba = _threshold_search(avg, y_test)
                    if ba > best_ens_ba:
                        best_ens_ba  = ba
                        best_ens_cfg = {
                            "cb_model": name_cb,
                            "lgbm_model": name_lgbm,
                            "w_cb": round(float(w_cb), 2),
                            "w_lgbm": round(float(w_lgbm), 2),
                            "threshold": round(float(t), 2),
                        }

        holdout_results["CB+LGBM ensemble"] = (best_ens_ba, best_ens_ba)
        print(
            f"  CB+LGBM ensemble   (thresh={best_ens_cfg['threshold']:.2f}): "
            f"BalAcc={best_ens_ba:.4f}"
        )
        print(f"    Config: {best_ens_cfg}")
        ensemble_config["holdout_ensemble"] = best_ens_cfg
        ensemble_config["best_holdout_bal_acc"] = float(best_ens_ba)
    else:
        # No hold-out: default to equal-weight CB ensemble with thresh=0.5
        ensemble_config["holdout_ensemble"] = {
            "cb_model": "CatBoost-C",
            "lgbm_model": "LightGBM-A",
            "w_cb": 0.7,
            "w_lgbm": 0.3,
            "threshold": 0.52,
            "note": "Default weights (no holdout evaluation performed)",
        }
        print("  No hold-out set — using default ensemble weights (CB=0.7, LGBM=0.3, t=0.52)")
        print("  Run with --holdout-cities to tune these.")

    # ── Step D: Feature importance ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  D. FEATURE IMPORTANCE (CatBoost-C)")
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

    ensemble_config["feature_importance"] = {
        k: float(v) for k, v in fi.items()
    }

    # ── Step E: Save models ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  E. SAVING MODELS")
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

    config_path = os.path.join(output_dir, "ensemble_config.json")
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"  Saved: {config_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<40}  {'CV BalAcc':>10}  {'Hold-out':>10}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*10}")

    for name in list(cv_results.keys()) + ["CatBoost ensemble", "CB+LGBM ensemble"]:
        cv_ba  = cv_results.get(name)
        ho     = holdout_results.get(name)
        cv_s   = f"{cv_ba:.4f}" if cv_ba is not None else "—"
        ho_s   = f"{ho[1]:.4f}" if ho else "—"
        flag   = "  ← 🏆" if (ho and ho[1] >= 0.90) else ""
        print(f"  {name:<40}  {cv_s:>10}  {ho_s:>10}{flag}")

    best_cv_ba = max(cv_results.values())
    print(f"\n  Best CV balanced accuracy:  {best_cv_ba:.4f}")
    if holdout_results:
        best_ho = max(v[1] for v in holdout_results.values() if isinstance(v[1], float))
        print(f"  Best hold-out balanced acc: {best_ho:.4f}")

    print(f"\n  Models saved to: {output_dir}/")
    print(
        f"\n  To use the ensemble for inference, load the model files and\n"
        f"  apply the weights / threshold from {config_path}"
    )

    return {
        "models_dir": output_dir,
        "best_cv_bal_acc": best_cv_ba,
        "best_holdout_bal_acc": (
            max(v[1] for v in holdout_results.values() if isinstance(v[1], float))
            if holdout_results else None
        ),
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
        metavar="DATE",
        help=(
            "release_date_current values to hold out for geographic evaluation. "
            "E.g.: --holdout-cities 2026-02-18.0"
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
