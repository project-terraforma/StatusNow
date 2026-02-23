"""
train_sf_licenses.py -- Train Random Forest & GBDT on SF data with new feature engineering.

Pipeline:
  1. Load enriched combined_truth_dataset_all.parquet (SF subset)
  2. Engineer V3 features + V4 new features (digital darkness, name change, zip churn)
  3. Optionally add license features
  4. Train RF and GBDT with 5-fold stratified CV
  5. 3-way comparison: V3 baseline / +V4 features / +V4+license
"""

import os
import sys
import io
import json
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    make_scorer, precision_score, recall_score,
    classification_report, balanced_accuracy_score,
    roc_auc_score, f1_score
)
from sklearn.decomposition import PCA

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT_FILE = "data/combined_truth_dataset_all.parquet"
REFERENCE_DATE = datetime(2026, 2, 23)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "precision_closed": make_scorer(precision_score, pos_label=0),
    "recall_closed": make_scorer(recall_score, pos_label=0),
    "balanced_acc": "balanced_accuracy",
}


# ── helpers ────────────────────────────────────────────────────────────────
def _parse_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x) if x.strip() else []
        except Exception:
            return []
    if isinstance(x, (list, dict)):
        return x
    return []


def _get_len(x):
    parsed = _parse_json(x)
    return len(parsed) if isinstance(parsed, list) else 0


def _is_present(x):
    return 1 if _get_len(x) > 0 else 0


# ── feature engineering (V3 pipeline) ──────────────────────────────────────
def engineer_features(df):
    """Apply V3 feature engineering pipeline to raw truth data."""
    print("  Engineering features ...")

    # Source features
    def _source_datasets(x):
        data = _parse_json(x)
        if isinstance(data, list):
            return [str(i.get("dataset", "")).lower() for i in data if isinstance(i, dict)]
        return []

    def _recency_stats(x):
        data = _parse_json(x)
        if not isinstance(data, list) or len(data) == 0:
            return pd.Series([9999, 9999, 9999])
        dates = []
        for item in data:
            if isinstance(item, dict):
                ut = item.get("update_time")
                if ut:
                    try:
                        dt = datetime.strptime(ut.split("T")[0], "%Y-%m-%d")
                        dates.append((REFERENCE_DATE - dt).days)
                    except Exception:
                        pass
        if not dates:
            return pd.Series([9999, 9999, 9999])
        return pd.Series([min(dates), max(dates), sum(dates) / len(dates)])

    df["num_sources"] = df["sources"].apply(_get_len)
    source_list = df["sources"].apply(_source_datasets)
    df["source_has_msft"] = source_list.apply(
        lambda x: 1 if ("microsoft" in x or "msft" in x) else 0
    )
    df["is_cross_verified"] = (df["num_sources"] > 1).astype(int)
    df[["days_since_latest_update", "days_since_oldest_update", "avg_days_since_update"]] = (
        df["sources"].apply(_recency_stats)
    )

    # Digital presence
    df["has_website"] = df["websites"].apply(_is_present)
    df["has_social"] = df["socials"].apply(_is_present)
    df["has_phone"] = df["phones"].apply(_is_present)

    def _has_platform(x, platform):
        data = _parse_json(x)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and platform in item.lower():
                    return 1
        return 0

    df["has_facebook"] = df["socials"].apply(lambda x: _has_platform(x, "facebook.com"))
    df["len_socials"] = df["socials"].apply(_get_len)

    def _email_count(x):
        if pd.isna(x): return 0
        if isinstance(x, (int, float)): return int(x)
        return _get_len(x)

    df["contact_depth"] = (
        df["websites"].apply(_get_len) + df["len_socials"] + df["emails"].apply(_email_count)
    )

    # Brand
    def _check_brand(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return 0
        if isinstance(x, str): return 0 if x.strip() in ("", "null", "[]") else 1
        return 1

    df["is_brand"] = df["brand"].apply(_check_brand)

    # Confidence
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["base_confidence"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["confidence"] = df["confidence"].fillna(df["base_confidence"])

    # Categories
    def _primary_cat(x):
        data = _parse_json(x)
        return data.get("primary", "unknown") if isinstance(data, dict) else "unknown"

    df["category_primary"] = df["categories"].apply(_primary_cat)
    df["cat_is_unknown"] = (df["category_primary"] == "unknown").astype(int)

    top_cats = df["category_primary"].value_counts().nlargest(20).index.tolist()
    if "unknown" in top_cats:
        top_cats.remove("unknown")
    df["category_simple"] = df["category_primary"].apply(lambda x: x if x in top_cats else "other")
    dummies = pd.get_dummies(df["category_simple"], prefix="cat")

    # Delta features
    df["base_confidence"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["delta_confidence"] = df["confidence"] - df["base_confidence"]

    for cur, base in [("websites", "base_websites"), ("socials", "base_socials"), ("phones", "base_phones")]:
        df[f"delta_num_{cur}"] = df[cur].apply(_get_len) - df[base].apply(_get_len)

    df["has_lost_website"] = (df["delta_num_websites"] < 0).astype(int)
    df["has_gained_social"] = (df["delta_num_socials"] > 0).astype(int)
    df["has_lost_social"] = (df["delta_num_socials"] < 0).astype(int)
    df["delta_total_contact"] = df["delta_num_websites"] + df["delta_num_socials"] + df["delta_num_phones"]
    df["has_any_loss"] = (
        (df["delta_num_websites"] < 0) | (df["delta_num_socials"] < 0) | (df["delta_num_phones"] < 0)
    ).astype(int)

    # Interaction features
    df["recency_x_loss"] = df["days_since_latest_update"] * df["has_any_loss"]
    df["recency_x_social_loss"] = df["days_since_latest_update"] * df["has_lost_social"]
    df["zombie_score"] = df["num_sources"] / (df["avg_days_since_update"] + 1)
    df["decay_velocity"] = df["delta_total_contact"] / (df["avg_days_since_update"] + 1)
    df["confidence_momentum"] = df["delta_confidence"] / (df["avg_days_since_update"] + 1)

    # Category churn risk
    cat_churn = df.groupby("category_primary")["label"].agg(["mean", "count"])
    cat_churn["churn_rate"] = 1 - cat_churn["mean"]
    reliable = cat_churn[cat_churn["count"] >= 10]["churn_rate"].to_dict()
    median_churn = cat_churn[cat_churn["count"] >= 10]["churn_rate"].median()
    df["category_churn_risk"] = df["category_primary"].map(reliable).fillna(median_churn)

    # Digital congruence
    def _congruence(row):
        w = _parse_json(row["websites"])
        s = _parse_json(row["socials"])
        if not w or not s: return 0
        domain = str(w[0]).replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0].split(".")[0].lower()
        for soc in s:
            if isinstance(soc, str) and domain in soc.lower(): return 1
        return 0

    df["digital_congruence"] = df.apply(_congruence, axis=1)

    # PCA on recency
    rec_feats = df[["days_since_latest_update", "avg_days_since_update"]].fillna(9999)
    pca = PCA(n_components=1)
    df["recency_pca"] = pca.fit_transform(rec_feats).flatten()

    # V3 recency decay
    days = df["days_since_latest_update"].clip(upper=9999)
    df["log_days_since_update"] = np.log1p(days)
    df["is_stale_6mo"] = (days > 180).astype(int)
    df["is_stale_1yr"] = (days > 365).astype(int)
    df["is_stale_2yr"] = (days > 730).astype(int)
    df["recency_bucket"] = pd.cut(
        days, bins=[-1, 90, 365, 730, 99999], labels=[0, 1, 2, 3]
    ).astype(int)

    # V3 brand-aware
    df["brand_x_stale"] = df["is_brand"] * df["is_stale_1yr"]
    df["nonbrand_stale_risk"] = (1 - df["is_brand"]) * df["is_stale_6mo"]

    # ══════════════════════════════════════════════════════════════════════
    # V4 NEW FEATURES
    # ══════════════════════════════════════════════════════════════════════

    # -- 1. Digital darkness / full presence compound features --
    print("  [V4] digital darkness ...")
    df["no_digital_presence"] = (
        (df["has_website"] == 0) & (df["has_social"] == 0) & (df["has_phone"] == 0)
    ).astype(int)
    df["has_all_contact"] = (
        (df["has_website"] == 1) & (df["has_social"] == 1) & (df["has_phone"] == 1)
    ).astype(int)

    # -- 2. Name-change detection --
    print("  [V4] name-change detection ...")
    def _get_primary_name(x):
        data = _parse_json(x)
        if isinstance(data, dict):
            return str(data.get("primary", "")).lower().strip()
        return ""

    curr_name = df["names"].apply(_get_primary_name)
    base_name = df["base_names"].apply(_get_primary_name)
    # Name changed = both exist and differ
    df["name_changed"] = (
        (curr_name.str.len() > 0) & (base_name.str.len() > 0) & (curr_name != base_name)
    ).astype(int)

    # -- 3. Zip-level churn rate --
    print("  [V4] zip-level churn rate ...")
    def _extract_zip(addr_field):
        data = _parse_json(addr_field)
        if isinstance(data, list) and len(data) > 0:
            z = str(data[0].get("postcode", "")).strip()
            m = re.match(r"(\d{5})", z)
            return m.group(1) if m else ""
        return ""

    df["_zip"] = df["addresses"].apply(_extract_zip)
    zip_churn = df[df["_zip"] != ""].groupby("_zip")["label"].agg(["mean", "count"])
    zip_churn["churn_rate"] = 1 - zip_churn["mean"]
    reliable_zips = zip_churn[zip_churn["count"] >= 5]["churn_rate"].to_dict()
    median_zip_churn = zip_churn[zip_churn["count"] >= 5]["churn_rate"].median()
    df["zip_churn_rate"] = df["_zip"].map(reliable_zips).fillna(median_zip_churn)

    n_dark = df["no_digital_presence"].sum()
    n_changed = df["name_changed"].sum()
    n_zips = len(reliable_zips)
    print(f"    Digital darkness: {n_dark} rows ({n_dark/len(df):.1%})")
    print(f"    Name changed:    {n_changed} rows ({n_changed/len(df):.1%})")
    print(f"    Zip churn:       {n_zips} zips with >= 5 samples")

    # ── Assemble feature sets ─────────────────────────────────────────────
    base_features = [
        "confidence", "is_brand", "num_sources", "source_has_msft", "is_cross_verified",
        "has_website", "has_social", "has_phone", "contact_depth",
        "cat_is_unknown", "category_churn_risk",
        "delta_confidence", "delta_num_socials", "delta_total_contact",
        "has_gained_social", "has_lost_social", "has_any_loss",
        "recency_pca",
        "recency_x_loss", "recency_x_social_loss", "zombie_score",
        "decay_velocity", "confidence_momentum",
        "digital_congruence",
        "log_days_since_update", "is_stale_6mo", "is_stale_1yr", "is_stale_2yr", "recency_bucket",
        "brand_x_stale", "nonbrand_stale_risk",
    ] + list(dummies.columns)

    new_v4_features = ["no_digital_presence", "has_all_contact", "name_changed", "zip_churn_rate"]

    # License features
    license_features = []
    for col in ["license_active", "days_to_license_expiry", "license_age_days", "license_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            license_features.append(col)
    if "license_count" in df.columns:
        df["has_license"] = (df["license_count"] > 0).astype(int)
        license_features.append("has_license")

    df_ml = pd.concat([df, dummies], axis=1)
    y = df_ml["label"].astype(int)

    X_base = df_ml[base_features].copy()
    X_v4 = df_ml[base_features + new_v4_features].copy()
    X_full = df_ml[base_features + new_v4_features + license_features].copy()

    return X_base, X_v4, X_full, y, base_features, new_v4_features, license_features


# ── training ───────────────────────────────────────────────────────────────
def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "GBDT": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=42,
        ),
    }


def run_cv(X, y, label=""):
    models = make_models()
    results = {}
    for name, model in models.items():
        print(f"    Training {name} ...", flush=True)
        t0 = time.time()
        cv_res = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            "Time (s)": round(elapsed, 1),
            "ROC AUC": cv_res["test_roc_auc"].mean(),
            "F1 Macro": cv_res["test_f1_macro"].mean(),
            "Prec(Closed)": cv_res["test_precision_closed"].mean(),
            "Recall(Closed)": cv_res["test_recall_closed"].mean(),
            "Balanced Acc": cv_res["test_balanced_acc"].mean(),
        }
        print(f"      done in {elapsed:.1f}s  |  ROC AUC={results[name]['ROC AUC']:.4f}")

    df_r = pd.DataFrame(results).T
    print(f"\n  {'-'*70}")
    print(f"  {label}")
    print(f"  {'-'*70}")
    print(df_r.round(4).to_string())
    return df_r


NEW_FEAT_SET = {"no_digital_presence", "has_all_contact", "name_changed", "zip_churn_rate"}
LIC_FEAT_SET = {"license_active", "days_to_license_expiry", "license_age_days",
                "license_count", "has_license"}


def feature_importance_report(X, y, top_n=20):
    """Train a single RF on all data and show feature importance."""
    print(f"\n  Feature importance (top {top_n}) ...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"  {'Feature':<35s}  {'Importance':>10s}")
    print(f"  {'-'*47}")
    for feat, val in imp.head(top_n).items():
        if feat in NEW_FEAT_SET:
            marker = " [NEW]"
        elif feat in LIC_FEAT_SET:
            marker = " [LIC]"
        else:
            marker = ""
        print(f"  {feat:<35s}  {val:>10.4f}{marker}")
    print(f"\n  [NEW] = V4 new feature  |  [LIC] = license feature")
    return imp


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  RF & GBDT Training -- SF Data + V4 Features + License")
    print("=" * 70)

    # Load enriched truth dataset
    print(f"\n  Loading {INPUT_FILE} ...")
    df_all = pd.read_parquet(INPUT_FILE)
    print(f"  Total rows: {len(df_all):,}")

    # Filter to SF only
    df_sf = df_all[df_all["source_dataset"] == "overture_sf"].copy().reset_index(drop=True)
    print(f"  SF rows:   {len(df_sf):,}")
    print(f"  Label dist: {df_sf['label'].value_counts().to_dict()}")

    if "license_count" in df_sf.columns:
        matched = (df_sf["license_count"] > 0).sum()
        print(f"  License matched: {matched:,} / {len(df_sf):,} ({matched/len(df_sf):.1%})")

    # Engineer features
    X_base, X_v4, X_full, y, base_feats, v4_feats, lic_feats = engineer_features(df_sf)

    print(f"\n  V3 base features: {len(base_feats)}")
    print(f"  V4 new features:  {len(v4_feats)}  {v4_feats}")
    print(f"  License features: {len(lic_feats)}  {lic_feats}")
    print(f"  Total features:   {X_full.shape[1]}")
    print(f"  Samples:          {len(y):,}")
    print(f"  Class balance:    {y.mean():.1%} Open")

    # ── A. V3 Baseline ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  A. V3 BASELINE")
    print(f"{'='*70}")
    r_base = run_cv(X_base, y, label="V3 Baseline")

    # ── B. V3 + V4 new features ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  B. V3 + V4 NEW FEATURES (darkness, name change, zip churn)")
    print(f"{'='*70}")
    r_v4 = run_cv(X_v4, y, label="V3 + V4 (digital darkness, name change, zip churn)")

    # ── C. V3 + V4 + License ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  C. V3 + V4 + LICENSE FEATURES")
    print(f"{'='*70}")
    r_full = run_cv(X_full, y, label="V3 + V4 + License Features")

    # ── D. Feature importance ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  D. FEATURE IMPORTANCE (RF on full feature set)")
    print(f"{'='*70}")
    imp = feature_importance_report(X_full, y, top_n=20)

    # ── COMPARISON ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")

    for metric in ["ROC AUC", "Balanced Acc", "F1 Macro", "Prec(Closed)", "Recall(Closed)"]:
        b_rf = r_base.loc["Random Forest", metric]
        v4_rf = r_v4.loc["Random Forest", metric]
        f_rf = r_full.loc["Random Forest", metric]
        b_gb = r_base.loc["GBDT", metric]
        v4_gb = r_v4.loc["GBDT", metric]
        f_gb = r_full.loc["GBDT", metric]
        print(f"\n  {metric}:")
        print(f"    RF:   {b_rf:.4f} -> {v4_rf:.4f} -> {f_rf:.4f}  (V3->V4: {v4_rf-b_rf:+.4f}, V4->Full: {f_rf-v4_rf:+.4f})")
        print(f"    GBDT: {b_gb:.4f} -> {v4_gb:.4f} -> {f_gb:.4f}  (V3->V4: {v4_gb-b_gb:+.4f}, V4->Full: {f_gb-v4_gb:+.4f})")

    # Best overall
    all_results = pd.concat([r_base, r_v4, r_full], keys=["V3 Base", "V3+V4", "V3+V4+Lic"])
    best_idx = all_results["Balanced Acc"].idxmax()
    best_val = all_results["Balanced Acc"].max()
    best_auc_idx = all_results["ROC AUC"].idxmax()
    best_auc = all_results["ROC AUC"].max()
    print(f"\n  Best Balanced Acc: {best_idx} = {best_val:.4f}")
    print(f"  Best ROC AUC:     {best_auc_idx} = {best_auc:.4f}")

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
