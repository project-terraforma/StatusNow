"""
V5 Train Best Model — CB+LGBM Ensemble (w=0.7, t=0.52)
=======================================================

Trains only the known best V5 configuration without running CV or comparing
alternative models. Faster than v5_full_benchmark.py — good for generating
predictions to feed into the V6 agent.

What this script does
---------------------
1. Loads the gold standard dataset and engineers V5 features
2. Splits into train (10 cities) / hold-out (Chicago + Miami)
3. Trains CatBoost-C (1000i, d6, lr=0.05) + LightGBM-A (1500, d8)
4. Applies best-known ensemble weights: w_CB=0.7, threshold=0.52
5. Prints hold-out metrics and feature importance
6. Exports predictions to data/v5_predictions_export.parquet

Expected result
---------------
  CB+LGBM ensemble (w_CB=0.7, t=0.52): ~89.41% balanced accuracy

Usage
-----
  python scripts/experiments/v5_train_best.py

  Options:
    --input         FILE   Path to the truth dataset (default: data/combined_truth_dataset_expanded.parquet)
    --output        FILE   Where to save predictions  (default: data/v5_predictions_export.parquet)
    --no-confidence        Drop all base_confidence features before training
"""

import argparse
import sys
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from research.process_data_v5 import build_v5_features


# ── Best-known configuration ──────────────────────────────────────────────────

HOLDOUT_CITIES = ["overture_chicago", "overture_miami"]
RANDOM_SEED    = 42

BEST_CB_PARAMS = dict(
    iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
    auto_class_weights="Balanced", verbose=0,
    random_state=RANDOM_SEED, allow_writing_files=False,
)

BEST_LGBM_PARAMS = dict(
    n_estimators=1500, learning_rate=0.05, max_depth=8,
    num_leaves=80, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)

W_CB   = 0.7
W_LGBM = 0.3
THRESH = 0.52


# ── Main ─────────────────────────────────────────────────────────────────────

CONFIDENCE_FEATURES = [
    "base_conf", "base_conf_sq", "base_conf_x_stale",
    "loss_x_low_conf", "stale_x_low_conf",
]


def main(input_path: str, output_path: str, model_dir: str = None,
         no_confidence: bool = False):
    mode = "NO-CONFIDENCE" if no_confidence else "full feature set"
    print("=" * 70)
    print(f"V5 TRAIN BEST MODEL — CB+LGBM Ensemble (w=0.7, t=0.52)  [{mode}]")
    print("Hold-out: Chicago + Miami")
    print("=" * 70)

    # ── Features ──────────────────────────────────────────────────────────────
    print(f"\nLoading: {input_path}")
    df, numeric_feats = build_v5_features(input_path)

    if no_confidence:
        dropped = [f for f in CONFIDENCE_FEATURES if f in numeric_feats]
        numeric_feats = [f for f in numeric_feats if f not in CONFIDENCE_FEATURES]
        print(f"\n  Dropped {len(dropped)} confidence features: {dropped}")

    holdout_mask = df["source_dataset"].isin(HOLDOUT_CITIES)
    train_df     = df[~holdout_mask].reset_index(drop=True)
    test_df      = df[holdout_mask].reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} rows  ({train_df['open'].mean():.1%} open)")
    print(f"  Test:  {len(test_df):,} rows  ({test_df['open'].mean():.1%} open)")
    print(f"  Features: {len(numeric_feats)} numeric + category_primary")

    all_feat_cols = numeric_feats + ["category_primary"]
    for split in [train_df, test_df]:
        split["category_primary"] = split["category_primary"].astype(str)

    cat_idx  = [all_feat_cols.index("category_primary")]
    X_train  = train_df[all_feat_cols]
    y_train  = train_df["open"]
    X_test   = test_df[all_feat_cols]
    y_test   = test_df["open"]
    X_train_num = train_df[numeric_feats]
    X_test_num  = test_df[numeric_feats]

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nTraining CatBoost-C … ", end="", flush=True)
    cb = CatBoostClassifier(**BEST_CB_PARAMS, cat_features=cat_idx)
    cb.fit(X_train, y_train)
    cb_prob = cb.predict_proba(X_test)[:, 1]
    print("done")

    print("Training LightGBM-A … ", end="", flush=True)
    lgbm = LGBMClassifier(**BEST_LGBM_PARAMS)
    lgbm.fit(X_train_num, y_train)
    lgbm_prob = lgbm.predict_proba(X_test_num)[:, 1]
    print("done")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    avg_prob = W_CB * cb_prob + W_LGBM * lgbm_prob
    pred     = (avg_prob >= THRESH).astype(int)

    ba   = balanced_accuracy_score(y_test, pred)
    auc  = roc_auc_score(y_test, avg_prob)
    prec = precision_score(y_test, pred, pos_label=0, zero_division=0)
    rec  = recall_score(y_test, pred, pos_label=0, zero_division=0)
    cm   = confusion_matrix(y_test, pred)

    print(f"""
Results (hold-out: Chicago + Miami)
────────────────────────────────────
  Balanced Accuracy : {ba:.4f}
  AUC               : {auc:.4f}
  Closed Precision  : {prec:.4f}
  Closed Recall     : {rec:.4f}
  Open Recall       : {cm[1, 1] / cm[1].sum():.4f}
  Confusion matrix  : {cm.tolist()}
""")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.Series(cb.feature_importances_, index=all_feat_cols).sort_values(ascending=False)
    print("Top 20 Features (CatBoost-C):")
    max_fi = fi.max()
    for fname, fval in fi.head(20).items():
        bar = "█" * int(fval / max_fi * 35)
        print(f"  {fname:45s} {fval:6.2f}  {bar}")

    # ── Save models ───────────────────────────────────────────────────────────
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        cb.save_model(os.path.join(model_dir, "catboost_c.cbm"))
        with open(os.path.join(model_dir, "lightgbm_a.pkl"), "wb") as f:
            pickle.dump(lgbm, f)
        with open(os.path.join(model_dir, "ensemble_config.json"), "w") as f:
            json.dump({"w_cb": W_CB, "w_lgbm": W_LGBM, "threshold": THRESH,
                       "catboost_model": "catboost_c.cbm",
                       "lightgbm_model": "lightgbm_a.pkl"}, f, indent=2)
        print(f"\nModels saved to: {model_dir}/")
        print(f"  catboost_c.cbm, lightgbm_a.pkl, ensemble_config.json")

    # ── Export predictions ────────────────────────────────────────────────────
    out_df = test_df.copy()
    out_df["v5_prob_open"]       = avg_prob
    out_df["v5_prob_closed"]     = 1.0 - avg_prob
    out_df["v5_confidence"]      = np.maximum(avg_prob, 1.0 - avg_prob)
    out_df["v5_predicted_label"] = pred

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"\nPredictions exported to: {output_path}")
    print(f"  {len(out_df):,} rows  |  "
          f"{(out_df['v5_confidence'] < 0.65).sum()} low-confidence POIs "
          f"(candidates for V6 agent)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the known best V5 model (CatBoost-C + LightGBM-A ensemble)\n"
            "and export predictions. Skips CV and multi-model comparison.\n\n"
            "Requires: data/combined_truth_dataset_expanded.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="data/combined_truth_dataset_expanded.parquet",
        help="Path to the combined truth dataset",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/v5_predictions_export.parquet",
        help="Where to save the hold-out predictions parquet",
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="models",
        help="Directory to save trained model files (default: models/). Pass empty string to skip.",
    )
    parser.add_argument(
        "--no-confidence", action="store_true",
        help="Drop all base_confidence features (base_conf, base_conf_sq, "
             "base_conf_x_stale, loss_x_low_conf, stale_x_low_conf) before training.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌  Dataset not found: {args.input}")
        print(f"    Run from the project root (StatusNow/).")
        sys.exit(1)

    main(args.input, args.output, model_dir=args.model_dir or None,
         no_confidence=args.no_confidence)
