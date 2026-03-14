"""
V5 Feature Pipeline — Leak-Free

Key changes from V4:
  1. Confidence: Use base_confidence ONLY (Jan 2026 quality score, always populated).
     Drop delta_confidence + confidence_momentum — both are 0 for all 93.7% of churned
     closed places (because fill made confidence = base_confidence, so delta = 0).
  2. category_churn_risk: REMOVED. Was computed globally from all labels → 0.50 correlation
     with target via leakage. Replaced by passing category_primary as a raw CatBoost
     categorical feature so the model learns encoding per fold internally.
  3. Geographic hold-out support: returns source_dataset column so caller can split by city.

Why delta features (has_gained_social etc.) are KEPT:
  - True, churned places (93.7% of closed) always have delta=0 due to COALESCE.
  - But non-churned open/closed places have real deltas from the Feb–Jan comparison.
  - These features still carry signal for explicitly-closed places.
  - Future fix: 3rd release (Dec 2025) would give leak-free pre-closure deltas.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA


CURRENT_DATE = datetime(2026, 2, 4)


# ── helpers ────────────────────────────────────────────────────────────────
def _parse(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def _len(x):
    v = _parse(x)
    return len(v) if isinstance(v, list) else 0


def _primary(x, key="primary"):
    v = _parse(x)
    if isinstance(v, dict):
        return v.get(key, "").lower().strip() if key == "primary" else v.get(key)
    return None


def _recency(x):
    data = _parse(x)
    if not isinstance(data, list) or not data:
        return (9999, 9999, 9999)
    days = []
    for item in data:
        if isinstance(item, dict):
            ut = item.get("update_time")
            if ut:
                try:
                    dt = datetime.strptime(ut.split("T")[0], "%Y-%m-%d")
                    days.append((CURRENT_DATE - dt).days)
                except Exception:
                    pass
    if not days:
        return (9999, 9999, 9999)
    return (min(days), max(days), sum(days) / len(days))


def _source_names(x):
    data = _parse(x)
    if isinstance(data, list):
        return [str(i.get("dataset", "")).lower() for i in data if isinstance(i, dict)]
    return []


def _website_domain(x):
    data = _parse(x)
    if isinstance(data, list) and data:
        url = str(data[0]).replace("http://", "").replace("https://", "").replace("www.", "")
        return url.split("/")[0].split("?")[0].lower()
    return None


def _address_key(x):
    data = _parse(x)
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            return f"{item.get('locality','')}_{item.get('postcode','')}".lower()
    return None


def _email_count(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return int(x)
    return _len(x)


# ── main pipeline ──────────────────────────────────────────────────────────
def build_v5_features(input_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with all features + 'open' target + 'source_dataset' column.
    The 'category_primary' column (raw string) is also included so CatBoost can use
    it as a native categorical feature — avoids global target-encoding leakage.
    """
    print("=" * 80)
    print("V5 FEATURE PIPELINE (Leak-Free)")
    print("=" * 80)

    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows | columns: {list(df.columns)}")

    # ── 1. SOURCES / RECENCY ──────────────────────────────────────────────
    print("  sources & recency …")
    df["num_sources"] = df["sources"].apply(_len)
    df["base_num_sources"] = df["base_sources"].apply(_len)
    df["delta_sources"] = df["num_sources"] - df["base_num_sources"]
    df["has_lost_sources"] = (df["delta_sources"] < 0).astype(int)
    df["source_list"] = df["sources"].apply(_source_names)
    df["source_has_msft"] = df["source_list"].apply(
        lambda x: 1 if any(s in x for s in ("microsoft", "msft")) else 0
    )
    df["is_cross_verified"] = (df["num_sources"] > 1).astype(int)
    df["log_num_sources"] = np.log1p(df["num_sources"])

    rec = df["sources"].apply(_recency)
    df["days_latest"] = rec.apply(lambda x: x[0])
    df["days_oldest"] = rec.apply(lambda x: x[1])
    df["days_avg"] = rec.apply(lambda x: x[2])

    # ── 2. DIGITAL PRESENCE ───────────────────────────────────────────────
    print("  digital presence …")
    df["num_websites"] = df["websites"].apply(_len)
    df["num_socials"] = df["socials"].apply(_len)
    df["num_phones"] = df["phones"].apply(_len)
    df["has_website"] = (df["num_websites"] > 0).astype(int)
    df["has_social"] = (df["num_socials"] > 0).astype(int)
    df["has_phone"] = (df["num_phones"] > 0).astype(int)

    def has_platform(x, p):
        data = _parse(x)
        if isinstance(data, list):
            return 1 if any(isinstance(s, str) and p in s.lower() for s in data) else 0
        return 0

    df["has_facebook"] = df["socials"].apply(lambda x: has_platform(x, "facebook.com"))
    df["has_instagram"] = df["socials"].apply(lambda x: has_platform(x, "instagram.com"))
    df["has_yelp"] = df["socials"].apply(lambda x: has_platform(x, "yelp.com"))

    df["num_emails"] = df["emails"].apply(_email_count)
    df["contact_depth"] = df["num_websites"] + df["num_socials"] + df["num_emails"]
    df["total_digital"] = df["has_website"] + df["has_social"] + df["has_phone"] + \
                          df["has_facebook"] + df["has_instagram"] + df["has_yelp"]

    # ── 3. BRAND ──────────────────────────────────────────────────────────
    df["is_brand"] = df["brand"].apply(
        lambda x: 0 if (x is None or (isinstance(x, float) and np.isnan(x))
                        or str(x).strip() in ("", "null", "[]")) else 1
    )

    # ── 4. CONFIDENCE — V5: base_confidence ONLY (no leaky delta) ─────────
    print("  confidence (base only — V5 leak fix) …")
    df["base_conf"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    # base_conf is always populated (Jan 2026 quality score for ALL places).
    # We do NOT use current confidence (null for 93.7% of closed = leaky).
    df["base_conf_sq"] = df["base_conf"] ** 2
    df["base_conf_x_stale"] = df["base_conf"]  # will be overwritten after staleness computed

    # ── 5. CATEGORIES ─────────────────────────────────────────────────────
    print("  categories …")
    df["category_primary"] = df["categories"].apply(_primary)
    # Keep raw string for CatBoost native encoding — no global churn risk precomputation
    df["category_primary"] = df["category_primary"].fillna("unknown")
    df["cat_is_unknown"] = (df["category_primary"] == "unknown").astype(int)

    # ── 6. DELTA FEATURES ─────────────────────────────────────────────────
    print("  delta features …")
    # NOTE: For churned places (93.7% of closed), COALESCE makes current = previous,
    # so all deltas are 0. This is the nature of 2-release datasets.
    # These features still carry real signal for the 6.3% explicitly-closed places.
    for cur, base in [("websites", "base_websites"),
                      ("socials", "base_socials"),
                      ("phones", "base_phones")]:
        df[f"delta_{cur}"] = df[cur].apply(_len) - df[base].apply(_len)

    df["has_lost_website"] = (df["delta_websites"] < 0).astype(int)
    df["has_gained_website"] = (df["delta_websites"] > 0).astype(int)
    df["has_lost_social"] = (df["delta_socials"] < 0).astype(int)
    df["has_gained_social"] = (df["delta_socials"] > 0).astype(int)
    df["has_lost_phone"] = (df["delta_phones"] < 0).astype(int)
    df["has_gained_phone"] = (df["delta_phones"] > 0).astype(int)
    df["delta_total"] = df["delta_websites"] + df["delta_socials"] + df["delta_phones"]
    df["has_any_loss"] = (
        (df["delta_websites"] < 0) | (df["delta_socials"] < 0) | (df["delta_phones"] < 0)
    ).astype(int)
    df["has_any_gain"] = (
        (df["delta_websites"] > 0) | (df["delta_socials"] > 0) | (df["delta_phones"] > 0)
    ).astype(int)
    df["num_loss_types"] = df["has_lost_website"] + df["has_lost_social"] + df["has_lost_phone"]
    df["num_gain_types"] = df["has_gained_website"] + df["has_gained_social"] + df["has_gained_phone"]
    df["has_complete_loss"] = ((df["delta_websites"] < 0) & (df["delta_socials"] < 0)).astype(int)
    df["contact_loss_severity"] = (
        df["has_lost_website"] * 2 + df["has_lost_social"] * 1.5 + df["has_lost_phone"] * 1
    )

    # ── 7. IDENTITY CHANGES ───────────────────────────────────────────────
    print("  identity changes …")
    df["curr_name"] = df["names"].apply(lambda x: _primary(x))
    df["base_name"] = df["base_names"].apply(lambda x: _primary(x))
    df["name_changed"] = (
        (df["curr_name"] != df["base_name"]) &
        df["curr_name"].notna() & df["base_name"].notna()
    ).astype(int)
    df["name_length"] = df["curr_name"].apply(lambda x: len(x) if x else 0)
    df["base_name_length"] = df["base_name"].apply(lambda x: len(x) if x else 0)
    df["name_length_delta"] = df["name_length"] - df["base_name_length"]

    df["base_cat"] = df["base_categories"].apply(_primary)
    df["cat_changed"] = (
        (df["category_primary"] != df["base_cat"]) &
        df["category_primary"].notna() & df["base_cat"].notna() &
        (df["category_primary"] != "unknown") & (df["base_cat"] != "unknown")
    ).astype(int)

    df["curr_domain"] = df["websites"].apply(_website_domain)
    df["base_domain"] = df["base_websites"].apply(_website_domain)
    df["website_domain_changed"] = (
        (df["curr_domain"] != df["base_domain"]) &
        df["curr_domain"].notna() & df["base_domain"].notna()
    ).astype(int)

    df["curr_addr"] = df["addresses"].apply(_address_key)
    df["base_addr"] = df["base_addresses"].apply(_address_key)
    df["address_changed"] = (
        (df["curr_addr"] != df["base_addr"]) &
        df["curr_addr"].notna() & df["base_addr"].notna()
    ).astype(int)

    df["identity_change_score"] = (
        df["name_changed"] + df["cat_changed"] + df["address_changed"]
    )

    # ── 8. RECENCY FEATURES ───────────────────────────────────────────────
    print("  recency decay …")
    days = df["days_latest"].clip(upper=9999)
    df["log_days"] = np.log1p(days)
    df["is_stale_3mo"] = (days > 90).astype(int)
    df["is_stale_6mo"] = (days > 180).astype(int)
    df["is_stale_1yr"] = (days > 365).astype(int)
    df["is_stale_2yr"] = (days > 730).astype(int)
    df["recency_bucket"] = pd.cut(
        days, bins=[-1, 90, 365, 730, 99999], labels=[0, 1, 2, 3]
    ).astype(int)
    df["recency_spread"] = (df["days_oldest"] - df["days_latest"]).clip(lower=0)

    rec_mat = df[["days_latest", "days_avg"]].fillna(9999)
    pca = PCA(n_components=1)
    df["recency_pca"] = pca.fit_transform(rec_mat).flatten()

    # Overwrite after staleness is available
    df["base_conf_x_stale"] = df["base_conf"] * df["is_stale_1yr"]

    # ── 9. INTERACTION FEATURES ───────────────────────────────────────────
    print("  interactions …")
    df["zombie_score"] = df["num_sources"] / (df["days_avg"] + 1)
    df["decay_velocity"] = df["delta_total"] / (df["days_avg"] + 1)
    df["recency_x_loss"] = days * df["has_any_loss"]
    df["recency_x_social_loss"] = days * df["has_lost_social"]

    df["brand_x_stale"] = df["is_brand"] * df["is_stale_1yr"]
    df["nonbrand_stale_risk"] = (1 - df["is_brand"]) * df["is_stale_6mo"]
    df["brand_x_name_change"] = df["is_brand"] * df["name_changed"]
    df["nonbrand_x_name_change"] = (1 - df["is_brand"]) * df["name_changed"]
    df["recency_x_name_change"] = days * df["name_changed"]
    df["recency_x_cat_change"] = days * df["cat_changed"]
    df["source_loss_x_stale"] = df["has_lost_sources"] * df["is_stale_6mo"]
    df["stale_x_loss_x_nonbrand"] = df["is_stale_6mo"] * df["has_any_loss"] * (1 - df["is_brand"])
    df["loss_x_low_conf"] = df["has_any_loss"] * (df["base_conf"] < 0.5).astype(int)
    df["stale_x_low_conf"] = df["is_stale_1yr"] * (df["base_conf"] < 0.5).astype(int)
    df["multi_signal_risk"] = (
        df["has_any_loss"] + df["is_stale_1yr"] + df["name_changed"] + df["cat_changed"]
    )

    # Digital congruence
    def congruence(row):
        w = _parse(row["websites"])
        s = _parse(row["socials"])
        if not w or not s or not isinstance(w, list) or not isinstance(s, list):
            return 0
        domain = str(w[0]).replace("http://","").replace("https://","").replace("www.","") \
                          .split("/")[0].split(".")[0].lower()
        return 1 if any(isinstance(x, str) and domain in x.lower() for x in s) else 0

    df["digital_congruence"] = df.apply(congruence, axis=1)

    # Source context
    if "source_dataset" in df.columns:
        df["is_overture_data"] = (df["source_dataset"].str.contains("overture", na=False)).astype(int)
    else:
        df["is_overture_data"] = 0

    # ── 10. ASSEMBLE ──────────────────────────────────────────────────────
    print("  assembling …")

    numeric_features = [
        # Confidence (base only — leak-free)
        "base_conf", "base_conf_sq", "base_conf_x_stale",
        # Brand & sources
        "is_brand", "num_sources", "log_num_sources", "source_has_msft", "is_cross_verified",
        # Digital presence
        "has_website", "has_social", "has_phone", "contact_depth",
        "has_facebook", "has_instagram", "has_yelp",
        "total_digital", "num_websites", "num_socials",
        # Category
        "cat_is_unknown",
        # Delta features
        "delta_websites", "delta_socials", "delta_phones",
        "delta_total", "delta_sources",
        "has_gained_social", "has_lost_social", "has_any_loss", "has_any_gain",
        "has_lost_website", "has_gained_website", "has_lost_phone", "has_gained_phone",
        "has_complete_loss", "num_loss_types", "num_gain_types",
        "contact_loss_severity",
        # Recency
        "recency_pca", "log_days", "is_stale_3mo", "is_stale_6mo",
        "is_stale_1yr", "is_stale_2yr", "recency_bucket", "recency_spread",
        # Interactions
        "zombie_score", "decay_velocity", "recency_x_loss", "recency_x_social_loss",
        "digital_congruence",
        "brand_x_stale", "nonbrand_stale_risk",
        "brand_x_name_change", "nonbrand_x_name_change",
        "recency_x_name_change", "recency_x_cat_change",
        "source_loss_x_stale", "stale_x_loss_x_nonbrand",
        "loss_x_low_conf", "stale_x_low_conf",
        # Identity changes
        "name_changed", "cat_changed", "website_domain_changed", "address_changed",
        "identity_change_score", "name_length", "name_length_delta",
        "has_lost_sources", "multi_signal_risk",
        # Context
        "is_overture_data",
    ]

    # Keep only features that were actually created
    numeric_features = [f for f in numeric_features if f in df.columns]

    # category_primary is kept as raw string for CatBoost categorical encoding
    keep_cols = numeric_features + [
        "category_primary", "source_dataset", "label",
        "id", "base_id", "names", "base_names", "categories", "base_categories", "addresses", "base_addresses"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    final = df[keep_cols].copy()
    final = final.dropna(subset=["label"])
    final.rename(columns={"label": "open"}, inplace=True)
    final["open"] = final["open"].astype(int)

    # Fill remaining NaN in numeric features
    for col in numeric_features:
        if col in final.columns and final[col].isna().any():
            final[col] = final[col].fillna(0)

    n_numeric = len(numeric_features)
    print(f"\n  Shape:           {final.shape}")
    print(f"  Open rate:       {final['open'].mean():.2%}")
    print(f"  Numeric feats:   {n_numeric}")
    print(f"  + category_primary as CatBoost cat feature (no global target encoding)")
    print(f"\n  V5 LEAK FIXES:")
    print(f"    ✓ confidence → base_conf only (Jan 2026, always populated)")
    print(f"    ✓ delta_confidence + confidence_momentum REMOVED")
    print(f"    ✓ category_churn_risk REMOVED (replaced by native CatBoost encoding)")
    print(f"    ✓ source_dataset retained for geographic hold-out splitting")

    return final, numeric_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/combined_truth_dataset_expanded.parquet")
    parser.add_argument("--output", "-o", default="data/processed_v5.parquet")
    args = parser.parse_args()

    df_out, _ = build_v5_features(args.input)
    df_out.to_parquet(args.output)
    print(f"\n  Saved → {args.output}")
