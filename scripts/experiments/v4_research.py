"""
V4 Research Pipeline — Systematic approach to 90% Balanced Accuracy

Experiment tracks:
1. Error analysis / diagnostic
2. V4 feature engineering (identity changes, richer interactions)
3. Hyperparameter optimization (Optuna)
4. Multi-model ensemble (stacking)  
5. Threshold tuning
6. Dataset expansion (if needed)
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
import os
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (
    balanced_accuracy_score, make_scorer, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CURRENT_DATE = datetime(2026, 2, 4)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {
    "roc_auc": "roc_auc",
    "balanced_acc": "balanced_accuracy",
    "precision_closed": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall_closed": make_scorer(recall_score, pos_label=0, zero_division=0),
}

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def parse_json_safe(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    if isinstance(x, (list, dict)):
        return x
    return None

def get_len(x):
    v = parse_json_safe(x)
    if v is None:
        return 0
    return len(v) if isinstance(v, list) else 0

def get_primary_name(x):
    v = parse_json_safe(x)
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("primary", "").lower().strip()
    return None

def get_primary_cat(x):
    v = parse_json_safe(x)
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("primary", None)
    return None

def get_recency_stats(x):
    data = parse_json_safe(x)
    if not isinstance(data, list) or len(data) == 0:
        return pd.Series([9999, 9999, 9999])
    dates = []
    for item in data:
        if isinstance(item, dict):
            ut = item.get("update_time")
            if ut:
                try:
                    dt = datetime.strptime(ut.split("T")[0], "%Y-%m-%d")
                    dates.append((CURRENT_DATE - dt).days)
                except:
                    pass
    if not dates:
        return pd.Series([9999, 9999, 9999])
    return pd.Series([min(dates), max(dates), sum(dates) / len(dates)])

def get_source_datasets(x):
    data = parse_json_safe(x)
    if isinstance(data, list):
        return [str(i.get("dataset", "")).lower() for i in data if isinstance(i, dict)]
    return []

def get_address_locality(x):
    data = parse_json_safe(x)
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if isinstance(item, dict):
            return f"{item.get('locality', '')}_{item.get('postcode', '')}".lower()
    return None

def get_website_domain(x):
    data = parse_json_safe(x)
    if isinstance(data, list) and len(data) > 0:
        url = str(data[0])
        domain = url.replace("http://", "").replace("https://", "").replace("www.", "")
        return domain.split("/")[0].split("?")[0].lower()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# V4 FEATURE ENGINEERING (comprehensive)
# ═══════════════════════════════════════════════════════════════════════════
def build_v4_features(input_path: str) -> pd.DataFrame:
    print("=" * 80)
    print("V4+ FEATURE ENGINEERING")
    print("=" * 80)

    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows from {input_path}")

    # ── 1. SOURCE FEATURES ────────────────────────────────────────────────
    print("  source features …")
    df["num_sources"] = df["sources"].apply(get_len)
    df["base_num_sources"] = df["base_sources"].apply(get_len)
    df["delta_sources"] = df["num_sources"] - df["base_num_sources"]
    df["source_list"] = df["sources"].apply(get_source_datasets)
    df["source_has_msft"] = df["source_list"].apply(
        lambda x: 1 if ("microsoft" in x or "msft" in x) else 0
    )
    df["is_cross_verified"] = (df["num_sources"] > 1).astype(int)

    recency_cols = df["sources"].apply(get_recency_stats)
    df["days_since_latest_update"] = recency_cols.iloc[:, 0]
    df["days_since_oldest_update"] = recency_cols.iloc[:, 1]
    df["avg_days_since_update"] = recency_cols.iloc[:, 2]

    # ── 2. DIGITAL PRESENCE ───────────────────────────────────────────────
    print("  digital presence …")
    df["num_websites"] = df["websites"].apply(get_len)
    df["num_socials"] = df["socials"].apply(get_len)
    df["num_phones"] = df["phones"].apply(get_len)
    df["has_website"] = (df["num_websites"] > 0).astype(int)
    df["has_social"] = (df["num_socials"] > 0).astype(int)
    df["has_phone"] = (df["num_phones"] > 0).astype(int)

    def has_platform(x, platform):
        data = parse_json_safe(x)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and platform in item.lower():
                    return 1
        return 0

    df["has_facebook"] = df["socials"].apply(lambda x: has_platform(x, "facebook.com"))
    df["has_instagram"] = df["socials"].apply(lambda x: has_platform(x, "instagram.com"))
    df["has_yelp"] = df["socials"].apply(lambda x: has_platform(x, "yelp.com"))

    def email_count(x):
        if pd.isna(x) if not isinstance(x, (list, dict)) else False:
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        return get_len(x)

    df["num_emails"] = df["emails"].apply(email_count)
    df["contact_depth"] = df["num_websites"] + df["num_socials"] + df["num_emails"]
    
    # Total digital signals (presence sum)
    df["total_digital_presence"] = (
        df["has_website"] + df["has_social"] + df["has_phone"] +
        df["has_facebook"] + df["has_instagram"] + df["has_yelp"]
    )

    # ── 3. BRAND ──────────────────────────────────────────────────────────
    def check_brand(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        if isinstance(x, str):
            return 0 if x.strip() in ("", "null", "[]") else 1
        return 1

    df["is_brand"] = df["brand"].apply(check_brand)

    # ── 4. CONFIDENCE ─────────────────────────────────────────────────────
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["base_confidence"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["confidence"] = df["confidence"].fillna(df["base_confidence"])

    # ── 5. CATEGORIES ─────────────────────────────────────────────────────
    print("  categories …")
    df["category_primary"] = df["categories"].apply(get_primary_cat)
    df["cat_is_unknown"] = (df["category_primary"].isna() | (df["category_primary"] == "unknown")).astype(int)

    top_cats = df["category_primary"].value_counts().nlargest(25).index.tolist()
    for drop_val in ["unknown", None]:
        if drop_val in top_cats:
            top_cats.remove(drop_val)
    df["category_simple"] = df["category_primary"].apply(lambda x: x if x in top_cats else "other")
    dummies = pd.get_dummies(df["category_simple"], prefix="cat")

    # Category churn risk (CV-safe: use label-based stats)
    cat_churn = df.groupby("category_primary")["label"].agg(["mean", "count"])
    cat_churn["churn_rate"] = 1 - cat_churn["mean"]
    reliable = cat_churn[cat_churn["count"] >= 10]["churn_rate"].to_dict()
    median_churn = cat_churn[cat_churn["count"] >= 10]["churn_rate"].median()
    df["category_churn_risk"] = df["category_primary"].map(reliable).fillna(median_churn)

    # ── 6. DELTA FEATURES ─────────────────────────────────────────────────
    print("  delta features …")
    df["delta_confidence"] = df["confidence"] - df["base_confidence"]

    for cur, base in [("websites", "base_websites"), ("socials", "base_socials"), ("phones", "base_phones")]:
        df[f"delta_num_{cur}"] = df[cur].apply(get_len) - df[base].apply(get_len)

    df["has_lost_website"] = (df["delta_num_websites"] < 0).astype(int)
    df["has_gained_website"] = (df["delta_num_websites"] > 0).astype(int)
    df["has_gained_social"] = (df["delta_num_socials"] > 0).astype(int)
    df["has_lost_social"] = (df["delta_num_socials"] < 0).astype(int)
    df["has_lost_phone"] = (df["delta_num_phones"] < 0).astype(int)
    df["has_gained_phone"] = (df["delta_num_phones"] > 0).astype(int)
    df["delta_total_contact"] = df["delta_num_websites"] + df["delta_num_socials"] + df["delta_num_phones"]
    df["has_any_loss"] = (
        (df["delta_num_websites"] < 0) | (df["delta_num_socials"] < 0) | (df["delta_num_phones"] < 0)
    ).astype(int)
    df["has_any_gain"] = (
        (df["delta_num_websites"] > 0) | (df["delta_num_socials"] > 0) | (df["delta_num_phones"] > 0)
    ).astype(int)
    df["has_complete_loss"] = (
        (df["delta_num_websites"] < 0) & (df["delta_num_socials"] < 0)
    ).astype(int)
    df["num_loss_types"] = df["has_lost_website"] + df["has_lost_social"] + df["has_lost_phone"]
    df["num_gain_types"] = df["has_gained_website"] + df["has_gained_social"] + df["has_gained_phone"]

    # ── 7. INTERACTION FEATURES ───────────────────────────────────────────
    print("  interaction features …")
    df["recency_x_loss"] = df["days_since_latest_update"] * df["has_any_loss"]
    df["recency_x_social_loss"] = df["days_since_latest_update"] * df["has_lost_social"]
    df["zombie_score"] = df["num_sources"] / (df["avg_days_since_update"] + 1)
    df["decay_velocity"] = df["delta_total_contact"] / (df["avg_days_since_update"] + 1)
    df["confidence_momentum"] = df["delta_confidence"] / (df["avg_days_since_update"] + 1)

    # ── 8. PCA ON RECENCY ─────────────────────────────────────────────────
    rec_feats = df[["days_since_latest_update", "avg_days_since_update"]].fillna(9999)
    pca = PCA(n_components=1)
    df["recency_pca"] = pca.fit_transform(rec_feats).flatten()

    # ── 9. DIGITAL CONGRUENCE ─────────────────────────────────────────────
    def congruence(row):
        w = parse_json_safe(row["websites"])
        s = parse_json_safe(row["socials"])
        if not w or not s or not isinstance(w, list) or not isinstance(s, list):
            return 0
        domain = str(w[0]).replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0].split(".")[0].lower()
        for soc in s:
            if isinstance(soc, str) and domain in soc.lower():
                return 1
        return 0

    df["digital_congruence"] = df.apply(congruence, axis=1)

    # ── 10. V3 RECENCY DECAY ─────────────────────────────────────────────
    print("  recency decay …")
    days = df["days_since_latest_update"].clip(upper=9999)
    df["log_days_since_update"] = np.log1p(days)
    df["is_stale_3mo"] = (days > 90).astype(int)
    df["is_stale_6mo"] = (days > 180).astype(int)
    df["is_stale_1yr"] = (days > 365).astype(int)
    df["is_stale_2yr"] = (days > 730).astype(int)
    df["recency_bucket"] = pd.cut(
        days, bins=[-1, 90, 365, 730, 99999], labels=[0, 1, 2, 3]
    ).astype(int)

    # ── 11. V3 BRAND-AWARE ────────────────────────────────────────────────
    df["brand_x_stale"] = df["is_brand"] * df["is_stale_1yr"]
    df["nonbrand_stale_risk"] = (1 - df["is_brand"]) * df["is_stale_6mo"]

    # ══════════════════════════════════════════════════════════════════════
    # V4 NEW FEATURES
    # ══════════════════════════════════════════════════════════════════════
    print("  [V4] identity change features …")

    # Name change
    df["curr_name"] = df["names"].apply(get_primary_name)
    df["base_name"] = df["base_names"].apply(get_primary_name)
    df["name_changed"] = (
        (df["curr_name"] != df["base_name"]) &
        df["curr_name"].notna() & df["base_name"].notna()
    ).astype(int)
    df["name_length"] = df["curr_name"].apply(lambda x: len(x) if x else 0)

    # Category change
    df["base_cat_primary"] = df["base_categories"].apply(get_primary_cat)
    df["cat_changed"] = (
        (df["category_primary"] != df["base_cat_primary"]) &
        df["category_primary"].notna() & df["base_cat_primary"].notna()
    ).astype(int)

    # Source loss
    df["has_lost_sources"] = (df["delta_sources"] < 0).astype(int)

    # Address change
    df["curr_address_key"] = df["addresses"].apply(get_address_locality)
    df["base_address_key"] = df["base_addresses"].apply(get_address_locality)
    df["address_changed"] = (
        (df["curr_address_key"] != df["base_address_key"]) &
        df["curr_address_key"].notna() & df["base_address_key"].notna()
    ).astype(int)

    # Website domain change
    df["curr_domain"] = df["websites"].apply(get_website_domain)
    df["base_domain"] = df["base_websites"].apply(get_website_domain)
    df["website_domain_changed"] = (
        (df["curr_domain"] != df["base_domain"]) &
        df["curr_domain"].notna() & df["base_domain"].notna()
    ).astype(int)

    # Composite identity change
    df["identity_change_score"] = df["name_changed"] + df["cat_changed"] + df["address_changed"]
    
    # Contact loss severity (weighted)
    df["contact_loss_severity"] = (
        df["has_lost_website"] * 2 +
        df["has_lost_social"] * 1.5 +
        df["has_lost_phone"] * 1
    )

    # V4 Interaction features
    df["brand_x_name_change"] = df["is_brand"] * df["name_changed"]
    df["nonbrand_x_name_change"] = (1 - df["is_brand"]) * df["name_changed"]
    df["recency_x_name_change"] = days * df["name_changed"]
    df["recency_x_cat_change"] = days * df["cat_changed"]
    df["source_loss_x_stale"] = df["has_lost_sources"] * df["is_stale_6mo"]
    df["loss_x_low_confidence"] = df["has_any_loss"] * (df["confidence"] < 0.5).astype(int)
    df["stale_x_low_confidence"] = df["is_stale_1yr"] * (df["confidence"] < 0.5).astype(int)
    df["stale_x_loss_x_nonbrand"] = df["is_stale_6mo"] * df["has_any_loss"] * (1 - df["is_brand"])
    
    # Source dataset
    if "source_dataset" in df.columns:
        df["is_overture_data"] = (df["source_dataset"].str.contains("overture", na=False)).astype(int)
    else:
        df["is_overture_data"] = 0

    # Recency spread (difference between oldest and newest source update)
    df["recency_spread"] = df["days_since_oldest_update"] - df["days_since_latest_update"]
    df["recency_spread"] = df["recency_spread"].clip(lower=0)

    # Confidence squared (nonlinearity)
    df["confidence_sq"] = df["confidence"] ** 2
    
    # Log num_sources 
    df["log_num_sources"] = np.log1p(df["num_sources"])

    # ── ASSEMBLE FINAL FEATURE SET ────────────────────────────────────────
    print("  assembling features …")
    df_ml = pd.concat([df, dummies], axis=1)

    feature_cols = [
        # Core metadata
        "confidence", "confidence_sq", "is_brand", "num_sources", "log_num_sources",
        "source_has_msft", "is_cross_verified",
        # Digital presence
        "has_website", "has_social", "has_phone", "contact_depth",
        "has_facebook", "has_instagram", "has_yelp",
        "total_digital_presence", "num_websites", "num_socials",
        # Category
        "cat_is_unknown", "category_churn_risk",
        # Delta
        "delta_confidence", "delta_num_websites", "delta_num_socials", "delta_num_phones",
        "delta_total_contact", "delta_sources",
        "has_gained_social", "has_lost_social", "has_any_loss", "has_any_gain",
        "has_lost_website", "has_gained_website", "has_lost_phone", "has_gained_phone",
        "has_complete_loss", "num_loss_types", "num_gain_types",
        # Recency
        "recency_pca",
        "log_days_since_update", "is_stale_3mo", "is_stale_6mo", "is_stale_1yr", "is_stale_2yr",
        "recency_bucket", "recency_spread",
        # Interactions (V2)
        "recency_x_loss", "recency_x_social_loss", "zombie_score",
        "decay_velocity", "confidence_momentum",
        # Congruence
        "digital_congruence",
        # Brand-aware (V3)
        "brand_x_stale", "nonbrand_stale_risk",
        # V4: Identity changes
        "name_changed", "cat_changed", "address_changed", "website_domain_changed",
        "identity_change_score", "name_length",
        # V4: Loss signals
        "has_lost_sources", "contact_loss_severity",
        # V4: Complex interactions
        "brand_x_name_change", "nonbrand_x_name_change",
        "recency_x_name_change", "recency_x_cat_change",
        "source_loss_x_stale", "loss_x_low_confidence",
        "stale_x_low_confidence", "stale_x_loss_x_nonbrand",
        # Context
        "is_overture_data",
    ] + list(dummies.columns)

    # Remove any features that don't exist
    feature_cols = [f for f in feature_cols if f in df_ml.columns]

    final_df = df_ml[feature_cols + ["label"]].copy()
    final_df = final_df.dropna(subset=["label"])
    final_df.rename(columns={"label": "open"}, inplace=True)
    final_df["open"] = final_df["open"].astype(int)

    # Fill NaN in features
    for col in feature_cols:
        if final_df[col].isna().any():
            final_df[col] = final_df[col].fillna(0)

    print(f"\n  Shape:         {final_df.shape}")
    print(f"  Class Balance: {final_df['open'].mean():.2%} Open / {1 - final_df['open'].mean():.2%} Closed")
    print(f"  Total Features: {len(feature_cols)}")

    return final_df


# ═══════════════════════════════════════════════════════════════════════════
# CV RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_cv(X, y, models_dict, label=""):
    results = {}
    for name, model in models_dict.items():
        print(f"    {name} …", end=" ", flush=True)
        t0 = time.time()
        cv_res = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            "Time(s)": round(elapsed, 1),
            "BalAcc": cv_res["test_balanced_acc"].mean(),
            "ROC_AUC": cv_res["test_roc_auc"].mean(),
            "Prec(Cl)": cv_res["test_precision_closed"].mean(),
            "Rec(Cl)": cv_res["test_recall_closed"].mean(),
        }
        print(f"BalAcc={results[name]['BalAcc']:.4f}  ROC={results[name]['ROC_AUC']:.4f}")
    df_r = pd.DataFrame(results).T
    print(f"\n{'─'*70}")
    print(f" {label}")
    print(f"{'─'*70}")
    print(df_r.round(4).to_string())
    return df_r


# ═══════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def error_analysis(X, y):
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    model = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=7,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    preds = cross_val_predict(model, X, y, cv=CV, n_jobs=-1)
    proba = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)

    ba = balanced_accuracy_score(y, preds)
    print(f"\n  Overall BalAcc: {ba:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y, preds)
    print(f"    Predicted:    Closed    Open")
    print(f"    Act Closed: {cm[0,0]:>7d}  {cm[0,1]:>7d}   (Recall={cm[0,0]/(cm[0,0]+cm[0,1]):.3f})")
    print(f"    Act Open:   {cm[1,0]:>7d}  {cm[1,1]:>7d}   (Recall={cm[1,1]/(cm[1,0]+cm[1,1]):.3f})")

    # Analyze errors by feature groups
    errors = preds != y.values
    print(f"\n  Total errors: {errors.sum()} / {len(y)} ({errors.mean():.1%})")

    # By brand
    if "is_brand" in X.columns:
        brand_mask = X["is_brand"] == 1
        brand_ba = balanced_accuracy_score(y[brand_mask], preds[brand_mask]) if brand_mask.sum() > 30 else float("nan")
        nonbrand_ba = balanced_accuracy_score(y[~brand_mask], preds[~brand_mask]) if (~brand_mask).sum() > 30 else float("nan")
        print(f"\n  Brand BalAcc:     {brand_ba:.4f}  (n={brand_mask.sum()})")
        print(f"  Non-brand BalAcc: {nonbrand_ba:.4f}  (n={(~brand_mask).sum()})")

    # By recency bucket
    if "recency_bucket" in X.columns:
        for bucket in sorted(X["recency_bucket"].unique()):
            mask = X["recency_bucket"] == bucket
            if mask.sum() > 30:
                bk_ba = balanced_accuracy_score(y[mask], preds[mask])
                print(f"  Recency bucket {bucket}: BalAcc={bk_ba:.4f} (n={mask.sum()})")

    # By staleness
    if "is_stale_6mo" in X.columns:
        stale = X["is_stale_6mo"] == 1
        fresh = X["is_stale_6mo"] == 0
        if stale.sum() > 30:
            print(f"\n  Fresh (<6mo): BalAcc={balanced_accuracy_score(y[fresh], preds[fresh]):.4f} (n={fresh.sum()})")
            print(f"  Stale (>6mo): BalAcc={balanced_accuracy_score(y[stale], preds[stale]):.4f} (n={stale.sum()})")

    # By confidence level
    if "confidence" in X.columns:
        low_conf = X["confidence"] < 0.5
        high_conf = X["confidence"] >= 0.8
        mid_conf = (~low_conf) & (~high_conf)
        for label_name, mask in [("Low conf (<0.5)", low_conf), ("Mid conf", mid_conf), ("High conf (≥0.8)", high_conf)]:
            if mask.sum() > 30:
                print(f"  {label_name}: BalAcc={balanced_accuracy_score(y[mask], preds[mask]):.4f} (n={mask.sum()})")

    # Hardest samples: model is confident but wrong
    label_prob = np.where(y == 1, proba[:, 1], proba[:, 0])
    hard_mask = (label_prob < 0.3) & errors
    print(f"\n  Hardest errors (model >70% confident, wrong): {hard_mask.sum()}")

    # Feature importance
    model.fit(X, y)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Top 15 Feature Importances:")
    for fname, fval in fi.head(15).items():
        print(f"    {fname:40s} {fval:.4f}")

    return ba, fi


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
def run_hpo(X, y, n_trials=50):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 80)
    print(f"OPTUNA HPO ({n_trials} trials)")
    print("=" * 80)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_state": 42,
            "allow_writing_files": False,
        }
        model = CatBoostClassifier(**params)
        scores = cross_validate(model, X, y, cv=CV, scoring={"balanced_acc": "balanced_accuracy"}, n_jobs=-1)
        return scores["test_balanced_acc"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best BalAcc: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def run_lgbm_hpo(X, y, n_trials=50):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 80)
    print(f"LIGHTGBM HPO ({n_trials} trials)")
    print("=" * 80)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_validate(model, X, y, cv=CV, scoring={"balanced_acc": "balanced_accuracy"}, n_jobs=-1)
        return scores["test_balanced_acc"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best BalAcc: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


# ═══════════════════════════════════════════════════════════════════════════
# STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════
def stacking_ensemble(X, y, cb_params=None, lgbm_params=None):
    print("\n" + "=" * 80)
    print("STACKING ENSEMBLE")
    print("=" * 80)

    cb_p = cb_params or {
        "iterations": 800, "learning_rate": 0.05, "depth": 7,
        "l2_leaf_reg": 5, "auto_class_weights": "Balanced",
        "verbose": 0, "random_state": 42, "allow_writing_files": False,
    }
    if "auto_class_weights" not in cb_p:
        cb_p["auto_class_weights"] = "Balanced"
    cb_p["verbose"] = 0
    cb_p["random_state"] = 42
    cb_p["allow_writing_files"] = False

    base_models = {
        "CatBoost": CatBoostClassifier(**cb_p),
        "XGBoost": XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 1).sum() / (y == 0).sum(),
            eval_metric="auc", random_state=42, n_jobs=-1,
        ),
    }

    lgbm_p = lgbm_params or {
        "n_estimators": 800, "learning_rate": 0.05, "max_depth": 7,
        "num_leaves": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    if "class_weight" not in lgbm_p:
        lgbm_p["class_weight"] = "balanced"
    lgbm_p["random_state"] = 42
    lgbm_p["n_jobs"] = -1
    lgbm_p["verbose"] = -1
    base_models["LightGBM"] = LGBMClassifier(**lgbm_p)

    # Generate OOF predictions
    print(f"  Generating OOF predictions for {len(base_models)} base models …")
    oof_preds = np.zeros((len(X), len(base_models)))
    individual_scores = {}

    for i, (name, model) in enumerate(base_models.items()):
        print(f"    {name} OOF …", end=" ", flush=True)
        oof_prob = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        oof_preds[:, i] = oof_prob
        ba = balanced_accuracy_score(y, (oof_prob >= 0.5).astype(int))
        individual_scores[name] = ba
        print(f"BalAcc={ba:.4f}")

    # Meta-learner
    print(f"\n  Training meta-learner (LogisticRegression) …")
    meta = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    meta_scores = cross_validate(meta, oof_preds, y, cv=CV, scoring={"balanced_acc": "balanced_accuracy"})
    meta_ba = meta_scores["test_balanced_acc"].mean()
    print(f"  Stacking BalAcc: {meta_ba:.4f}")

    # Threshold tuning  
    meta_fitted = LogisticRegression(C=1.0, class_weight="balanced", random_state=42).fit(oof_preds, y)
    meta_oof_prob = cross_val_predict(meta_fitted, oof_preds, y, cv=CV, method="predict_proba")[:, 1]

    best_thresh, best_ba = 0.5, 0
    for t in np.arange(0.25, 0.75, 0.01):
        ba = balanced_accuracy_score(y, (meta_oof_prob >= t).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_thresh = t
    print(f"  Threshold-tuned: thresh={best_thresh:.2f} → BalAcc={best_ba:.4f}")

    # Simple averaging ensemble
    avg_prob = oof_preds.mean(axis=1)
    avg_ba = 0
    avg_thresh = 0.5
    for t in np.arange(0.25, 0.75, 0.01):
        ba = balanced_accuracy_score(y, (avg_prob >= t).astype(int))
        if ba > avg_ba:
            avg_ba = ba
            avg_thresh = t
    print(f"  Averaging ensemble: thresh={avg_thresh:.2f} → BalAcc={avg_ba:.4f}")

    # Weighted averaging (optimize weights)
    from itertools import product as iprod
    best_w_ba = 0
    best_weights = None
    for w1, w2, w3 in iprod(np.arange(0.1, 0.9, 0.1), repeat=3):
        wsum = w1 + w2 + w3
        w_prob = (w1 * oof_preds[:, 0] + w2 * oof_preds[:, 1] + w3 * oof_preds[:, 2]) / wsum
        for t in np.arange(0.3, 0.7, 0.05):
            ba = balanced_accuracy_score(y, (w_prob >= t).astype(int))
            if ba > best_w_ba:
                best_w_ba = ba
                best_weights = (w1/wsum, w2/wsum, w3/wsum)
    print(f"  Weighted ensemble: weights={best_weights} → BalAcc={best_w_ba:.4f}")

    return {
        "individual": individual_scores,
        "stacking": meta_ba,
        "stacking_threshold": best_ba,
        "averaging": avg_ba,
        "weighted": best_w_ba,
        "best": max(meta_ba, best_ba, avg_ba, best_w_ba),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main(input_path="data/combined_truth_dataset.parquet"):
    print("=" * 80)
    print("V4 RESEARCH PIPELINE — Target: 90% Balanced Accuracy")
    print("=" * 80)
    print(f"Input: {input_path}\n")

    # ── Step 1: Build V4 Features ─────────────────────────────────────────
    df_v4 = build_v4_features(input_path)
    X = df_v4.drop(columns=["open"])
    y = df_v4["open"]

    # ── Step 2: Quick baseline comparison ─────────────────────────────────
    print("\n\n▶ STEP 1: V3 BASELINE (reproduce)")
    v3_path = "data/processed_for_ml_testing.parquet"
    if os.path.exists(v3_path):
        df_v3 = pd.read_parquet(v3_path)
        X_v3 = df_v3.drop(columns=["open"])
        y_v3 = df_v3["open"]
        r_v3 = run_cv(X_v3, y_v3, {
            "CatBoost (V3 baseline)": CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                verbose=0, auto_class_weights="Balanced",
                random_state=42, allow_writing_files=False,
            ),
        }, "V3 Baseline")
    else:
        print("  V3 baseline file not found, skipping.")
        r_v3 = None

    # ── Step 3: V4 Features + Default Models ──────────────────────────────
    print("\n\n▶ STEP 2: V4 FEATURES + DEFAULT MODELS")
    default_models = {
        "CatBoost": CatBoostClassifier(
            iterations=800, learning_rate=0.05, depth=7,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 1).sum() / (y == 0).sum(),
            eval_metric="auc", random_state=42, n_jobs=-1,
        ),
    }
    r_v4 = run_cv(X, y, default_models, "V4 Features + Default Models")

    # ── Step 4: Error Analysis ────────────────────────────────────────────
    print("\n\n▶ STEP 3: ERROR ANALYSIS")
    baseline_ba, feature_importance = error_analysis(X, y)

    # ── Step 5: HPO ───────────────────────────────────────────────────────
    print("\n\n▶ STEP 4: CATBOOST HPO")
    cb_best_params, cb_best_score = run_hpo(X, y, n_trials=60)

    print("\n\n▶ STEP 5: LIGHTGBM HPO")
    lgbm_best_params, lgbm_best_score = run_hpo(X, y, n_trials=60) if False else (None, None)
    lgbm_best_params, lgbm_best_score = run_lgbm_hpo(X, y, n_trials=60)

    # ── Step 6: Try HPO models ────────────────────────────────────────────
    print("\n\n▶ STEP 6: HPO MODELS VALIDATED")
    hpo_models = {}
    if cb_best_params:
        full_cb = {**cb_best_params, "auto_class_weights": "Balanced", "verbose": 0,
                   "random_state": 42, "allow_writing_files": False}
        hpo_models["CatBoost (HPO)"] = CatBoostClassifier(**full_cb)
    if lgbm_best_params:
        full_lgbm = {**lgbm_best_params, "class_weight": "balanced", "random_state": 42,
                     "n_jobs": -1, "verbose": -1}
        hpo_models["LightGBM (HPO)"] = LGBMClassifier(**full_lgbm)
    if hpo_models:
        r_hpo = run_cv(X, y, hpo_models, "HPO Models")

    # ── Step 7: Stacking Ensemble ─────────────────────────────────────────
    print("\n\n▶ STEP 7: STACKING ENSEMBLE")
    ensemble_results = stacking_ensemble(X, y, cb_params=cb_best_params, lgbm_params=lgbm_best_params)

    # ── Step 8: Try on larger dataset ─────────────────────────────────────
    large_path = "data/combined_truth_dataset_all.parquet"
    r_large = None
    if os.path.exists(large_path):
        print("\n\n▶ STEP 8: LARGER DATASET (18.6k)")
        df_v4_large = build_v4_features(large_path)
        X_large = df_v4_large.drop(columns=["open"])
        y_large = df_v4_large["open"]
        
        large_models = {}
        if cb_best_params:
            full_cb = {**cb_best_params, "auto_class_weights": "Balanced", "verbose": 0,
                       "random_state": 42, "allow_writing_files": False}
            large_models["CatBoost (HPO)"] = CatBoostClassifier(**full_cb)
        if lgbm_best_params:
            full_lgbm = {**lgbm_best_params, "class_weight": "balanced", "random_state": 42,
                         "n_jobs": -1, "verbose": -1}
            large_models["LightGBM (HPO)"] = LGBMClassifier(**full_lgbm)
        if large_models:
            r_large = run_cv(X_large, y_large, large_models, "HPO on 18.6k dataset")
        
        # Ensemble on large
        print("\n  Stacking on large dataset …")
        ensemble_large = stacking_ensemble(X_large, y_large, cb_params=cb_best_params, lgbm_params=lgbm_best_params)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    results_summary = {}
    if r_v3 is not None:
        results_summary["V3 Baseline (12k)"] = r_v3["BalAcc"].max()
    results_summary["V4 CatBoost (12k)"] = r_v4["BalAcc"].iloc[0] if len(r_v4) > 0 else 0
    results_summary["V4 LightGBM (12k)"] = r_v4["BalAcc"].iloc[1] if len(r_v4) > 1 else 0
    if cb_best_score:
        results_summary["CatBoost HPO (12k)"] = cb_best_score
    if lgbm_best_score:
        results_summary["LightGBM HPO (12k)"] = lgbm_best_score
    results_summary["Ensemble Best (12k)"] = ensemble_results["best"]
    if r_large is not None:
        results_summary["HPO Best (18.6k)"] = r_large["BalAcc"].max()

    for name, score in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
        marker = " 🏆" if score >= 0.90 else ""
        print(f"  {name:35s}  {score:.4f}{marker}")

    best_achieved = max(results_summary.values())
    print(f"\n  Best achieved: {best_achieved:.4f}")
    print(f"  Target:        0.9000")
    if best_achieved >= 0.90:
        print(f"\n  🏆 TARGET ACHIEVED!")
    else:
        print(f"\n  Gap: {0.90 - best_achieved:.4f}")
        print(f"  → Next step: Expand dataset to more cities")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/combined_truth_dataset.parquet")
    args = parser.parse_args()
    main(args.input)
