"""
Experiment: Predictive Label Training (R2-oracle labels, R0→R1 features only)
==============================================================================

Hypothesis to test
------------------
In the current pipeline, labels are defined by R2 presence and features are
also drawn from R1/R2 data (pair 1).  The concern: the model might be learning
"does this place have R2-era characteristics?" rather than genuine closure
signals.

This experiment completely excludes R2 from the feature window.

Setup
-----
  Features : R0→R1 comparison (Jan→Feb) — same feature engineering as usual
  Labels   : R2 oracle (Mar)
               label=0 (Closed) : present in R0 AND R1, absent from R2  (HQC)
               label=1 (Open)   : present in R1 AND in R2

Rationale: forces the model to predict FUTURE outcomes from PAST features,
more closely mirroring production use.  If accuracy drops significantly vs
the current pipeline, the model was learning R2-specific artifacts.
If accuracy holds, the signals are genuinely predictive.

Output
------
Prints balanced accuracy, AUC, precision, recall on Chicago + Miami hold-out.
Nothing is saved / pushed.
"""

import json
import os
import pickle
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.step2_feature_engineering import build_features

# ─── config ──────────────────────────────────────────────────────────────────

RELEASES_DIR   = "overture_releases"
OUTPUT_RAW     = "pipeline_output/exp_predictive_raw.parquet"
OUTPUT_FEATS   = "pipeline_output/exp_predictive_feats.parquet"
HOLDOUT_CITIES = {"chicago", "miami"}
TARGET_OPEN    = 0.6
SEED           = 42

# ─── step 1: build R0→R1 pair, label from R2 ─────────────────────────────────

import re, os as _os
RELEASE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}\.\d+)")

def _discover():
    entries = []
    for f in os.listdir(RELEASES_DIR):
        if not f.endswith(".parquet"):
            continue
        m = RELEASE_PATTERN.match(f)
        if m:
            entries.append((m.group(1), os.path.join(RELEASES_DIR, f)))
    entries.sort(key=lambda x: x[0])
    return entries

releases = _discover()
assert len(releases) >= 3, "Need at least 3 releases for this experiment"

tag_r0, path_r0 = releases[0]
tag_r1, path_r1 = releases[1]
tag_r2, path_r2 = releases[2]

print(f"R0 (features base):  {tag_r0}")
print(f"R1 (features curr):  {tag_r1}")
print(f"R2 (label oracle):   {tag_r2}  ← excluded from features entirely")
print()

# Build R0→R1 full outer join (identical SQL to step1)
print("Building R0→R1 feature pair …")
con = duckdb.connect()
query = f"""
    WITH base AS (
        SELECT
            id,
            CAST(names      AS VARCHAR) AS names,
            CAST(categories AS VARCHAR) AS categories,
            CAST(websites   AS VARCHAR) AS websites,
            CAST(socials    AS VARCHAR) AS socials,
            CAST(phones     AS VARCHAR) AS phones,
            CAST(emails     AS VARCHAR) AS emails,
            CAST(addresses  AS VARCHAR) AS addresses,
            CAST(brand      AS VARCHAR) AS brand,
            CAST(sources    AS VARCHAR) AS sources,
            CAST(confidence AS DOUBLE)  AS confidence,
            try_cast(operating_status AS VARCHAR) AS operating_status,
            CAST(_city      AS VARCHAR) AS city
        FROM read_parquet('{path_r0}')
    ),
    curr AS (
        SELECT
            id,
            CAST(names      AS VARCHAR) AS names,
            CAST(categories AS VARCHAR) AS categories,
            CAST(websites   AS VARCHAR) AS websites,
            CAST(socials    AS VARCHAR) AS socials,
            CAST(phones     AS VARCHAR) AS phones,
            CAST(emails     AS VARCHAR) AS emails,
            CAST(addresses  AS VARCHAR) AS addresses,
            CAST(brand      AS VARCHAR) AS brand,
            CAST(sources    AS VARCHAR) AS sources,
            CAST(confidence AS DOUBLE)  AS confidence,
            try_cast(operating_status AS VARCHAR) AS operating_status,
            CAST(_city      AS VARCHAR) AS city
        FROM read_parquet('{path_r1}')
    ),
    r2_ids AS (
        SELECT id FROM read_parquet('{path_r2}')
    )
    SELECT
        COALESCE(curr.id,         base.id)         AS id,
        COALESCE(curr.names,      base.names)       AS names,
        COALESCE(curr.categories, base.categories)  AS categories,
        COALESCE(curr.websites,   base.websites)    AS websites,
        COALESCE(curr.socials,    base.socials)     AS socials,
        COALESCE(curr.phones,     base.phones)      AS phones,
        COALESCE(curr.emails,     base.emails)      AS emails,
        COALESCE(curr.addresses,  base.addresses)   AS addresses,
        COALESCE(curr.brand,      base.brand)       AS brand,
        COALESCE(curr.sources,    base.sources)     AS sources,
        curr.confidence                             AS confidence,
        COALESCE(curr.city,       base.city)        AS city,

        base.id         AS base_id,
        base.names      AS base_names,
        base.categories AS base_categories,
        base.websites   AS base_websites,
        base.socials    AS base_socials,
        base.phones     AS base_phones,
        base.emails     AS base_emails,
        base.addresses  AS base_addresses,
        base.brand      AS base_brand,
        base.sources    AS base_sources,
        base.confidence AS base_confidence,

        -- Label from R2 oracle (NOT from R1 presence)
        CASE
            WHEN r2_ids.id IS NOT NULL THEN 1   -- present in R2 → Open
            ELSE 0                               -- absent from R2 → Closed
        END AS label

    FROM base
    FULL OUTER JOIN curr   ON base.id = curr.id
    LEFT  JOIN r2_ids      ON COALESCE(curr.id, base.id) = r2_ids.id
    -- Only keep places that exist in R1 (curr); dropping R0-only places
    -- because they already churned before the feature window ends.
    WHERE curr.id IS NOT NULL
"""

df = con.execute(query).df()
con.close()

n_open   = (df["label"] == 1).sum()
n_closed = (df["label"] == 0).sum()
print(f"Raw from R0→R1 (R1-present only): {len(df):,}  open={n_open:,}  closed={n_closed:,}")

# Keep only HQC closed (also present in R0 = base_id not null)
hqc_mask   = df["base_id"].notna()
closed_mask = df["label"] == 0
df = df[~closed_mask | (closed_mask & hqc_mask)].copy()

n_open   = (df["label"] == 1).sum()
n_closed = (df["label"] == 0).sum()
print(f"After HQC filter:                  {len(df):,}  open={n_open:,}  closed={n_closed:,}")

# Add required metadata columns for step2 compatibility
df["release_pair"]         = f"{tag_r0}→{tag_r1}"
df["release_index"]        = 0
df["release_date_base"]    = tag_r0
df["release_date_current"] = tag_r1   # ← features measured at R1; reference date = R1

# Global 60/40 rebalance (keep all HQC closed, sample open)
n_open_target = int(round(n_closed * TARGET_OPEN / (1.0 - TARGET_OPEN)))
print(f"\nRebalancing to 60/40: {n_closed:,} closed → {n_open_target:,} open target")
if n_open > n_open_target:
    open_df   = df[df["label"] == 1].sample(n=n_open_target, random_state=SEED)
    closed_df = df[df["label"] == 0]
    df        = pd.concat([open_df, closed_df]).reset_index(drop=True)

print(f"Final raw dataset:    {len(df):,}  open={int((df['label']==1).sum()):,} ({df['label'].mean():.1%})")

os.makedirs("pipeline_output", exist_ok=True)
df.to_parquet(OUTPUT_RAW, index=False)

# ─── step 2: feature engineering ─────────────────────────────────────────────

print("\n" + "─"*60)
df_feats, numeric_feats = build_features(input_path=OUTPUT_RAW, output_path=OUTPUT_FEATS)

# ─── step 3: train & evaluate ────────────────────────────────────────────────

print("\n" + "─"*60)
print("TRAINING & EVALUATION")
print("─"*60)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

df_feats = pd.read_parquet(OUTPUT_FEATS)

non_feat = {"open", "city", "release_pair", "release_date_base",
            "release_date_current", "release_index", "category_primary",
            "days_latest", "days_avg"}
numeric_feats = [c for c in df_feats.columns if c not in non_feat]

# Train/hold-out split by city
mask       = df_feats["city"].isin(HOLDOUT_CITIES)
train_df   = df_feats[~mask].reset_index(drop=True)
test_df    = df_feats[mask].reset_index(drop=True)

print(f"\nTrain: {len(train_df):,}  ({train_df['open'].mean():.1%} open)")
print(f"Test:  {len(test_df):,}   ({test_df['open'].mean():.1%} open)  [{', '.join(HOLDOUT_CITIES)}]")

# PCA on train only
pca = PCA(n_components=1)
pca.fit(train_df[["days_latest","days_avg"]].fillna(9999))
train_df["recency_pca"] = pca.transform(train_df[["days_latest","days_avg"]].fillna(9999)).flatten()
test_df["recency_pca"]  = pca.transform(test_df[["days_latest","days_avg"]].fillna(9999)).flatten()
numeric_feats = numeric_feats + ["recency_pca"]

# LabelEncoder on train only
le = LabelEncoder()
le.fit(train_df["category_primary"].astype(str))
train_df["category_encoded"] = le.transform(train_df["category_primary"].astype(str))
test_df["category_encoded"]  = test_df["category_primary"].astype(str).apply(
    lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1)

all_feat_cols  = numeric_feats + ["category_primary"]
lgbm_feat_cols = numeric_feats + ["category_encoded"]
cat_idx        = [all_feat_cols.index("category_primary")]

X_train      = train_df[all_feat_cols]
y_train      = train_df["open"].values
X_train_lgbm = train_df[lgbm_feat_cols]
X_test       = test_df[all_feat_cols]
y_test       = test_df["open"].values
X_test_lgbm  = test_df[lgbm_feat_cols]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# OOF for best model config (CatBoost-C + LightGBM-A)
print("\nOOF cross-validation …")
cb_params = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
                 auto_class_weights="Balanced", cat_features=cat_idx,
                 verbose=0, random_state=SEED, allow_writing_files=False)
lgbm_params = dict(n_estimators=1500, learning_rate=0.05, max_depth=8,
                   num_leaves=80, subsample=0.8, colsample_bytree=0.8,
                   class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)

oof_cb   = cross_val_predict(CatBoostClassifier(**cb_params),
                              X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:,1]
oof_lgbm = cross_val_predict(LGBMClassifier(**lgbm_params),
                              X_train_lgbm, y_train, cv=CV, method="predict_proba", n_jobs=1)[:,1]

# Ensemble search on OOF
best_ba, best_w, best_t = 0.0, 0.7, 0.5
for w in np.arange(0.3, 0.81, 0.1):
    avg = w * oof_cb + (1-w) * oof_lgbm
    for t in np.arange(0.25, 0.76, 0.01):
        ba = balanced_accuracy_score(y_train, (avg >= t).astype(int))
        if ba > best_ba:
            best_ba, best_w, best_t = ba, w, t

print(f"Best OOF: BalAcc={best_ba:.4f}  w_cb={best_w:.1f}  thresh={best_t:.2f}")

# Train final models
print("\nTraining final models …")
cb_final   = CatBoostClassifier(**cb_params); cb_final.fit(X_train, y_train)
lgbm_final = LGBMClassifier(**lgbm_params);   lgbm_final.fit(X_train_lgbm, y_train)

# Hold-out evaluation with pre-determined config
p_cb   = cb_final.predict_proba(X_test)[:,1]
p_lgbm = lgbm_final.predict_proba(X_test_lgbm)[:,1]
avg    = best_w * p_cb + (1-best_w) * p_lgbm
pred   = (avg >= best_t).astype(int)

print("\n" + "="*60)
print("HOLD-OUT RESULTS (Chicago + Miami)")
print("="*60)
print(f"  Balanced Accuracy : {balanced_accuracy_score(y_test, pred):.4f}")
print(f"  AUC               : {roc_auc_score(y_test, avg):.4f}")
print(f"  Precision (Open)  : {precision_score(y_test, pred, pos_label=1):.4f}")
print(f"  Recall    (Open)  : {recall_score(y_test, pred, pos_label=1):.4f}")
print(f"  Precision (Closed): {precision_score(y_test, pred, pos_label=0):.4f}")
print(f"  Recall    (Closed): {recall_score(y_test, pred, pos_label=0):.4f}")
print(f"\n  Ensemble: w_cb={best_w:.1f}  w_lgbm={1-best_w:.1f}  thresh={best_t:.2f}")
print(f"  Train/test: {len(train_df):,} / {len(test_df):,}")
cm = confusion_matrix(y_test, pred)
print(f"\n  Confusion matrix (rows=actual, cols=pred):")
print(f"              Pred Closed  Pred Open")
print(f"  Act Closed  {cm[0,0]:>10,}  {cm[0,1]:>9,}")
print(f"  Act Open    {cm[1,0]:>10,}  {cm[1,1]:>9,}")
print()
print("  (No R2 data was used in any feature — labels only)")
