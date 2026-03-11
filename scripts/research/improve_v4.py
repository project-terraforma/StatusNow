"""
V4 Improvement Pipeline — Target: 90% Balanced Accuracy

Strategy:
1. New features: name_changed, cat_changed, delta_sources, phone_loss, address_changed
2. CatBoost hyperparameter tuning with Optuna
3. LightGBM addition
4. Ensemble: CatBoost + XGBoost + LightGBM with stacking
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import balanced_accuracy_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Try LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

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
        except Exception:
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


def get_recency_stats(x, current_date):
    from datetime import datetime
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
                    dates.append((current_date - dt).days)
                except Exception:
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
# V4 FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def build_v4_features(input_path: str = "data/combined_truth_dataset.parquet") -> pd.DataFrame:
    from datetime import datetime
    CURRENT_DATE = datetime(2026, 2, 4)

    print("=" * 80)
    print("V4 FEATURE ENGINEERING")
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

    recency_cols = df["sources"].apply(lambda x: get_recency_stats(x, CURRENT_DATE))
    df["days_since_latest_update"] = recency_cols.iloc[:, 0]
    df["days_since_oldest_update"] = recency_cols.iloc[:, 1]
    df["avg_days_since_update"] = recency_cols.iloc[:, 2]

    # ── 2. DIGITAL PRESENCE ───────────────────────────────────────────────
    print("  digital presence …")
    df["has_website"] = (df["websites"].apply(get_len) > 0).astype(int)
    df["has_social"] = (df["socials"].apply(get_len) > 0).astype(int)
    df["has_phone"] = (df["phones"].apply(get_len) > 0).astype(int)
    df["len_socials"] = df["socials"].apply(get_len)

    def has_platform(x, platform):
        data = parse_json_safe(x)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and platform in item.lower():
                    return 1
        return 0

    df["has_facebook"] = df["socials"].apply(lambda x: has_platform(x, "facebook.com"))

    def email_count(x):
        if pd.isna(x) if not isinstance(x, (list, dict)) else False:
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        return get_len(x)

    df["contact_depth"] = (
        df["websites"].apply(get_len) + df["len_socials"] + df["emails"].apply(email_count)
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
    # Leakage fix: use base_confidence for churned places
    df["confidence"] = df["confidence"].fillna(df["base_confidence"])

    # ── 5. CATEGORIES ─────────────────────────────────────────────────────
    print("  categories …")
    df["category_primary"] = df["categories"].apply(get_primary_cat)
    df["cat_is_unknown"] = (df["category_primary"].isna() | (df["category_primary"] == "unknown")).astype(int)

    top_cats = df["category_primary"].value_counts().nlargest(20).index.tolist()
    for drop_val in ["unknown", None]:
        if drop_val in top_cats:
            top_cats.remove(drop_val)
    df["category_simple"] = df["category_primary"].apply(lambda x: x if x in top_cats else "other")
    dummies = pd.get_dummies(df["category_simple"], prefix="cat")

    # Category churn risk (computed from data)
    cat_churn = df.groupby("category_primary")["label"].agg(["mean", "count"])
    cat_churn["churn_rate"] = 1 - cat_churn["mean"]
    reliable = cat_churn[cat_churn["count"] >= 10]["churn_rate"].to_dict()
    median_churn = cat_churn[cat_churn["count"] >= 10]["churn_rate"].median()
    df["category_churn_risk"] = df["category_primary"].map(reliable).fillna(median_churn)

    # ── 6. DELTA FEATURES ─────────────────────────────────────────────────
    print("  delta features …")
    df["base_confidence_val"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["delta_confidence"] = df["confidence"] - df["base_confidence_val"]

    for cur, base in [("websites", "base_websites"), ("socials", "base_socials"), ("phones", "base_phones")]:
        df[f"delta_num_{cur}"] = df[cur].apply(get_len) - df[base].apply(get_len)

    df["has_lost_website"] = (df["delta_num_websites"] < 0).astype(int)
    df["has_gained_social"] = (df["delta_num_socials"] > 0).astype(int)
    df["has_lost_social"] = (df["delta_num_socials"] < 0).astype(int)
    df["has_lost_phone"] = (df["delta_num_phones"] < 0).astype(int)
    df["delta_total_contact"] = df["delta_num_websites"] + df["delta_num_socials"] + df["delta_num_phones"]
    df["has_any_loss"] = (
        (df["delta_num_websites"] < 0) | (df["delta_num_socials"] < 0) | (df["delta_num_phones"] < 0)
    ).astype(int)
    df["has_complete_loss"] = (
        (df["delta_num_websites"] < 0) & (df["delta_num_socials"] < 0)
    ).astype(int)

    # ── 7. V2 INTERACTION FEATURES ────────────────────────────────────────
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

    # ── 9. V2 DIGITAL CONGRUENCE ─────────────────────────────────────────
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

    # ── 10. V3 RECENCY DECAY NON-LINEARITY ────────────────────────────────
    print("  [V3] recency decay …")
    days = df["days_since_latest_update"].clip(upper=9999)
    df["log_days_since_update"] = np.log1p(days)
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
    # ── V4 NEW FEATURES ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    print("  [V4] NEW FEATURES …")

    # Name change signal (26% closed vs 12% open)
    df["curr_name"] = df["names"].apply(get_primary_name)
    df["base_name"] = df["base_names"].apply(get_primary_name)
    df["name_changed"] = (
        (df["curr_name"] != df["base_name"]) &
        df["curr_name"].notna() & df["base_name"].notna()
    ).astype(int)
    # Name complexity: longer names may signal a rebrand
    df["name_length"] = df["curr_name"].apply(lambda x: len(x) if x else 0)
    df["base_name_length"] = df["base_name"].apply(lambda x: len(x) if x else 0)
    df["name_length_delta"] = df["name_length"] - df["base_name_length"]

    # Category change signal (24% closed vs 15% open)
    df["base_cat_primary"] = df["base_categories"].apply(get_primary_cat)
    df["cat_changed"] = (
        (df["category_primary"] != df["base_cat_primary"]) &
        df["category_primary"].notna() & df["base_cat_primary"].notna()
    ).astype(int)

    # Source count delta (already computed: delta_sources)
    # Losing sources is a closure signal
    df["has_lost_sources"] = (df["delta_sources"] < 0).astype(int)

    # Address change
    df["curr_address_key"] = df["addresses"].apply(get_address_locality)
    df["base_address_key"] = df["base_addresses"].apply(get_address_locality)
    df["address_changed"] = (
        (df["curr_address_key"] != df["base_address_key"]) &
        df["curr_address_key"].notna() & df["base_address_key"].notna()
    ).astype(int)

    # Website domain change (stronger than just presence change)
    df["curr_website_domain"] = df["websites"].apply(get_website_domain)
    df["base_website_domain"] = df["base_websites"].apply(get_website_domain)
    df["website_domain_changed"] = (
        (df["curr_website_domain"] != df["base_website_domain"]) &
        df["curr_website_domain"].notna() & df["base_website_domain"].notna()
    ).astype(int)

    # Total identity changes (compound signal)
    df["identity_change_score"] = (
        df["name_changed"] + df["cat_changed"] + df["address_changed"]
    )

    # Has source dataset (NYC vs season2) as feature
    df["is_overture_nyc"] = (df["source_dataset"] == "overture_nyc").astype(int)

    # Brand x name_change: Brand that changes name is unusual (potential closure/transition)
    df["brand_x_name_change"] = df["is_brand"] * df["name_changed"]
    df["nonbrand_x_name_change"] = (1 - df["is_brand"]) * df["name_changed"]

    # Multi-contact loss strength
    df["contact_loss_severity"] = (
        df["has_lost_website"] * 2 +
        df["has_lost_social"] * 1.5 +
        df["has_lost_phone"] * 1
    )

    # Recency x name change: recent name changes are more suspicious
    df["recency_x_name_change"] = df["days_since_latest_update"] * df["name_changed"]
    df["recency_x_cat_change"] = df["days_since_latest_update"] * df["cat_changed"]

    # Source diversity (was using multiple data sources)
    df["source_loss_x_stale"] = df["has_lost_sources"] * df["is_stale_6mo"]

    print(f"  [V4] Added features: name_changed, cat_changed, address_changed, "
          f"website_domain_changed, identity_change_score, has_lost_sources, "
          f"contact_loss_severity + interactions")

    # ── ASSEMBLE FINAL FEATURE SET ────────────────────────────────────────
    print("  assembling final features …")
    df_ml = pd.concat([df, dummies], axis=1)

    v3_features = [
        # Core metadata
        "confidence", "is_brand", "num_sources", "source_has_msft", "is_cross_verified",
        # Digital presence
        "has_website", "has_social", "has_phone", "contact_depth",
        # Category
        "cat_is_unknown", "category_churn_risk",
        # Delta
        "delta_confidence", "delta_num_socials", "delta_total_contact",
        "has_gained_social", "has_lost_social", "has_any_loss",
        # Recency (PCA)
        "recency_pca",
        # V2 interactions
        "recency_x_loss", "recency_x_social_loss", "zombie_score",
        "decay_velocity", "confidence_momentum",
        # V2 congruence
        "digital_congruence",
        # V3 recency decay
        "log_days_since_update", "is_stale_6mo", "is_stale_1yr", "is_stale_2yr", "recency_bucket",
        # V3 brand-aware
        "brand_x_stale", "nonbrand_stale_risk",
    ]

    v4_features = [
        # NEW: Identity change signals
        "name_changed", "cat_changed", "address_changed", "website_domain_changed",
        "identity_change_score",
        # NEW: Loss signals
        "has_lost_phone", "has_lost_website", "has_complete_loss",
        "has_lost_sources", "delta_sources",
        "contact_loss_severity",
        # NEW: Name signals
        "name_length", "name_length_delta",
        # NEW: Interactions
        "brand_x_name_change", "nonbrand_x_name_change",
        "recency_x_name_change", "recency_x_cat_change",
        "source_loss_x_stale",
        # Dataset source
        "is_overture_nyc",
    ]

    all_features = v3_features + v4_features + list(dummies.columns)

    # Make sure all features exist
    missing = [f for f in all_features if f not in df_ml.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
        all_features = [f for f in all_features if f in df_ml.columns]

    final_df = df_ml[all_features + ["label"]].copy()
    final_df = final_df.dropna(subset=["label"])
    final_df.rename(columns={"label": "open"}, inplace=True)
    final_df["open"] = final_df["open"].astype(int)

    print(f"\n  Shape:         {final_df.shape}")
    print(f"  Class Balance: {final_df['open'].mean():.2%} Open")
    print(f"  V3 features:   {len(v3_features)} + {len(dummies.columns)} cat dummies")
    print(f"  V4 NEW:        {len(v4_features)}")
    print(f"  Total:         {len(all_features)}")

    return final_df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS
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
        print(f"BalAcc={results[name]['BalAcc']:.4f}")
    df_r = pd.DataFrame(results).T
    print(f"\n{'─'*70}")
    print(f" {label}")
    print(f"{'─'*70}")
    print(df_r.round(4).to_string())
    return df_r


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════════════════════
def catboost_hpo(X, y, n_trials=30):
    """Optuna-based HPO for CatBoost."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. Skipping HPO.")
        return None, None

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_state": 42,
            "allow_writing_files": False,
        }
        model = CatBoostClassifier(**params)
        scores = cross_validate(model, X, y, cv=CV, scoring={"balanced_acc": "balanced_accuracy"}, n_jobs=-1)
        return scores["test_balanced_acc"].mean()

    print(f"\n  Running Optuna HPO ({n_trials} trials) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    print(f"  Best BalAcc: {best_score:.4f}")
    print(f"  Best params: {best_params}")
    return best_params, best_score


# ═══════════════════════════════════════════════════════════════════════════
# STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════
def stacking_ensemble(X, y, best_catboost_params=None):
    """Build a stacking ensemble with CatBoost, XGBoost, LightGBM as base learners."""
    print("\n" + "=" * 80)
    print("STACKING ENSEMBLE")
    print("=" * 80)

    # Base models
    cb_params = best_catboost_params or {
        "iterations": 800, "learning_rate": 0.05, "depth": 7,
        "l2_leaf_reg": 5, "auto_class_weights": "Balanced",
        "verbose": 0, "random_state": 42, "allow_writing_files": False,
    }
    base_models = {
        "CatBoost": CatBoostClassifier(**cb_params),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 1).sum() / (y == 0).sum(),
            eval_metric="auc", random_state=42, n_jobs=-1,
        ),
    }

    if HAS_LGBM:
        base_models["LightGBM"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )

    # Generate out-of-fold predictions for meta-learner
    print(f"  Generating OOF predictions for {len(base_models)} base models …")
    oof_preds = np.zeros((len(X), len(base_models)))
    model_names = list(base_models.keys())

    for i, (name, model) in enumerate(base_models.items()):
        print(f"    {name} OOF …", end=" ", flush=True)
        oof_prob = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        oof_preds[:, i] = oof_prob
        ba = balanced_accuracy_score(y, (oof_prob >= 0.5).astype(int))
        print(f"BalAcc={ba:.4f}")

    # Meta-learner: Logistic Regression on OOF predictions
    print(f"\n  Training meta-learner (LogisticRegression) …")
    meta = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    meta_scores = cross_validate(meta, oof_preds, y, cv=CV, scoring={"balanced_acc": "balanced_accuracy"})
    meta_bal_acc = meta_scores["test_balanced_acc"].mean()
    print(f"  Meta-learner BalAcc: {meta_bal_acc:.4f}")

    # Also try optimal threshold on OOF
    from sklearn.linear_model import LogisticRegression as LR
    thresholds = np.arange(0.3, 0.7, 0.02)
    best_thresh = 0.5
    best_ba = 0
    meta_fitted = LR(C=1.0, class_weight="balanced", random_state=42).fit(oof_preds, y)
    meta_oof_prob = cross_val_predict(meta_fitted, oof_preds, y, cv=CV, method="predict_proba")[:, 1]
    for t in thresholds:
        ba = balanced_accuracy_score(y, (meta_oof_prob >= t).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_thresh = t
    print(f"  Threshold tuned: {best_thresh:.2f} → BalAcc={best_ba:.4f}")

    return meta_bal_acc, best_ba, model_names, oof_preds


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main(input_path: str = "data/combined_truth_dataset.parquet"):
    print("=" * 80)
    print("V4 IMPROVEMENT PIPELINE — Target: 90% Balanced Accuracy")
    print("=" * 80)
    print(f"\nInput: {input_path}")

    # ── Step 1: Build V4 Features ─────────────────────────────────────────
    df_v4 = build_v4_features(input_path)
    X_v4 = df_v4.drop(columns=["open"])
    y_v4 = df_v4["open"]

    # Also load V3 features for comparison
    v3_path = "data/processed_for_ml_testing.parquet"
    df_v3 = pd.read_parquet(v3_path)
    X_v3 = df_v3.drop(columns=["open"])
    y_v3 = df_v3["open"]

    print(f"\n  V3 features: {X_v3.shape[1]}")
    print(f"  V4 features: {X_v4.shape[1]} (+{X_v4.shape[1] - X_v3.shape[1]} new)")

    # ── Step 2: Baseline Models ───────────────────────────────────────────
    print("\n\n▶ A. V3 BASELINE (reproduced)")
    baseline_models = {
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        ),
    }
    r_baseline = run_cv(X_v3, y_v3, baseline_models, "V3 Baseline (12k, 52 features)")

    # ── Step 3: V4 Features ───────────────────────────────────────────────
    print("\n\n▶ B. V4 FEATURES (new signals)")
    v4_models = {
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        ),
    }
    r_v4 = run_cv(X_v4, y_v4, v4_models, "V4 Features")

    # ── Step 4: Larger Dataset ────────────────────────────────────────────
    print("\n\n▶ C. V4 FEATURES + LARGER DATASET (18.6k)")
    large_path = "data/combined_truth_dataset_all.parquet"
    import os
    if os.path.exists(large_path):
        df_v4_large = build_v4_features(large_path)
        X_v4l = df_v4_large.drop(columns=["open"])
        y_v4l = df_v4_large["open"]
        print(f"  Large dataset: {X_v4l.shape[0]} samples, {X_v4l.shape[1]} features")
        r_v4_large = run_cv(X_v4l, y_v4l, v4_models, "V4 Features + 18.6k dataset")
    else:
        print("  (large dataset not found, skipping)")
        r_v4_large = None

    # ── Step 5: CatBoost HPO ──────────────────────────────────────────────
    print("\n\n▶ D. CATBOOST HYPERPARAMETER OPTIMIZATION")
    best_params, best_hpo_score = catboost_hpo(X_v4, y_v4, n_trials=40)

    if best_params:
        best_params_full = {**best_params, "auto_class_weights": "Balanced",
                            "verbose": 0, "random_state": 42, "allow_writing_files": False}
        hpo_models = {"CatBoost (HPO)": CatBoostClassifier(**best_params_full)}
        r_hpo = run_cv(X_v4, y_v4, hpo_models, "CatBoost HPO + V4 Features")

    # ── Step 6: All models on V4 ──────────────────────────────────────────
    print("\n\n▶ E. ALL MODELS ON V4 FEATURES")
    all_models = {
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            verbose=0, auto_class_weights="Balanced",
            random_state=42, allow_writing_files=False,
        ),
        "XGBoost (tuned)": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_v4 == 1).sum() / (y_v4 == 0).sum(),
            eval_metric="auc", random_state=42, n_jobs=-1,
        ),
    }
    if HAS_LGBM:
        all_models["LightGBM"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    r_allmodels = run_cv(X_v4, y_v4, all_models, "All Models on V4 Features")

    # ── Step 7: HPO Model on All Data ─────────────────────────────────────
    if best_params and r_v4_large is not None:
        print("\n\n▶ F. BEST HPO MODEL ON LARGE DATASET")
        hpo_large_models = {"CatBoost (HPO, 18.6k)": CatBoostClassifier(**best_params_full)}
        r_hpo_large = run_cv(X_v4l, y_v4l, hpo_large_models, "HPO CatBoost + V4 + 18.6k")

    # ── Step 8: Stacking Ensemble ─────────────────────────────────────────
    print("\n\n▶ G. STACKING ENSEMBLE")
    meta_ba, meta_ba_tuned, base_names, oof_preds = stacking_ensemble(
        X_v4, y_v4, best_catboost_params=best_params_full if best_params else None
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    baseline_ba = r_baseline["BalAcc"].max()
    v4_ba = r_v4["BalAcc"].max()
    print(f"\n  V3 Baseline (12k, 52 feat):     {baseline_ba:.4f}")
    print(f"  V4 Features (12k, +new):        {v4_ba:.4f}  Δ {v4_ba - baseline_ba:+.4f}")
    if r_v4_large is not None:
        print(f"  V4 Features (18.6k):            {r_v4_large['BalAcc'].max():.4f}  Δ {r_v4_large['BalAcc'].max() - baseline_ba:+.4f}")
    if best_hpo_score:
        print(f"  CatBoost HPO + V4 (12k):        {best_hpo_score:.4f}  Δ {best_hpo_score - baseline_ba:+.4f}")
    all_ba = r_allmodels["BalAcc"].max()
    print(f"  Best Single Model (V4):         {all_ba:.4f}  Δ {all_ba - baseline_ba:+.4f}")
    print(f"  Stacking Ensemble (V4):         {meta_ba:.4f}  Δ {meta_ba - baseline_ba:+.4f}")
    print(f"  Stacking + Threshold (V4):      {meta_ba_tuned:.4f}  Δ {meta_ba_tuned - baseline_ba:+.4f}")

    TARGET = 0.90
    best_achieved = max(v4_ba, all_ba, meta_ba, meta_ba_tuned,
                        best_hpo_score if best_hpo_score else 0)
    if r_v4_large is not None:
        best_achieved = max(best_achieved, r_v4_large["BalAcc"].max())

    print(f"\n  Target:    {TARGET:.2%}")
    print(f"  Best:      {best_achieved:.4f}")
    if best_achieved >= TARGET:
        print(f"\n  🏆 TARGET ACHIEVED! {best_achieved:.4f} >= {TARGET:.2%}")
    else:
        gap = TARGET - best_achieved
        print(f"\n  Gap to target: {gap:.4f} ({gap:.2%})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/combined_truth_dataset.parquet")
    args = parser.parse_args()
    main(args.input)
