"""
Process Data V3 — builds on V2 with:
  • Recency decay non-linearity (log-transform + staleness bins)
  • All V2 features preserved (interactions, category churn, PCA, congruence)
  • Optional Project C merge
"""

import duckdb
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA


# ── helpers ────────────────────────────────────────────────────────────────
def _parse_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            if x.strip() == "":
                return []
            return json.loads(x)
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


# ── main pipeline ─────────────────────────────────────────────────────────
def process_data_v3(use_project_c_merge: bool = False, input_path: str = "data/Season 2 Samples 3k Project Updated.parquet", output_path: str = "data/processed_for_ml_v3.parquet"):
    import os
    from datetime import datetime

    CURRENT_DATE = datetime(2026, 2, 4)

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────
    if use_project_c_merge:
        print("=" * 80)
        print("MERGING DATASETS")
        print("=" * 80)
        con = duckdb.connect()
        s2 = con.execute(
            "SELECT * FROM read_parquet('data/Season 2 Samples 3k Project Updated.parquet')"
        ).df()
        pc = con.execute(
            "SELECT * FROM read_parquet('data/Project C Samples.parquet')"
        ).df()

        if "open" in pc.columns and "label" not in pc.columns:
            pc["label"] = pc["open"]
            pc = pc.drop(columns=["open", "geometry", "bbox", "type", "version"], errors="ignore")

        for col in [c for c in s2.columns if c.startswith("base_")]:
            if col not in pc.columns:
                pc[col] = None

        df = pd.concat([s2, pc], ignore_index=True)
        print(f"✅ Merged: Season 2 ({len(s2)}) + Project C ({len(pc)}) = {len(df)} total rows")
    else:
        parquet_file = input_path
        if not os.path.exists(parquet_file):
            print(f"Error: '{parquet_file}' not found.")
            return
        print(f"Reading '{parquet_file}'...")
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").df()

    print("Engineering features …")

    # ── 2. SOURCE FEATURES ────────────────────────────────────────────────
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
                        dates.append((CURRENT_DATE - dt).days)
                    except Exception:
                        pass
        if not dates:
            return pd.Series([9999, 9999, 9999])
        return pd.Series([min(dates), max(dates), sum(dates) / len(dates)])

    print("  source features …")
    df["num_sources"] = df["sources"].apply(_get_len)
    df["source_list"] = df["sources"].apply(_source_datasets)
    df["source_has_msft"] = df["source_list"].apply(
        lambda x: 1 if ("microsoft" in x or "msft" in x) else 0
    )
    df["is_cross_verified"] = (df["num_sources"] > 1).astype(int)
    df[["days_since_latest_update", "days_since_oldest_update", "avg_days_since_update"]] = (
        df["sources"].apply(_recency_stats)
    )

    # ── 3. DIGITAL PRESENCE ───────────────────────────────────────────────
    print("  digital presence …")
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
        if pd.isna(x):
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        return _get_len(x)

    df["contact_depth"] = (
        df["websites"].apply(_get_len) + df["len_socials"] + df["emails"].apply(_email_count)
    )

    # ── 4. BRAND ──────────────────────────────────────────────────────────
    def _check_brand(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        if isinstance(x, str):
            return 0 if x.strip() in ("", "null", "[]") else 1
        return 1

    df["is_brand"] = df["brand"].apply(_check_brand)

    # ── 5. CONFIDENCE ─────────────────────────────────────────────────────
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["base_confidence"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    
    # LEAKAGE FIX: If confidence is missing (churned place), use base_confidence.
    # This prevents the model from learning "Confidence=0 means Closed".
    df["confidence"] = df["confidence"].fillna(df["base_confidence"])
    
    # ── 6. CATEGORIES ─────────────────────────────────────────────────────
    print("  categories …")

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

    # ── 7. DELTA FEATURES ─────────────────────────────────────────────────
    print("  delta features …")
    df["base_confidence"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["delta_confidence"] = df["confidence"] - df["base_confidence"]

    for col_pair in [("websites", "base_websites"), ("socials", "base_socials"), ("phones", "base_phones")]:
        cur, base = col_pair
        short = cur  # websites, socials, phones
        df[f"delta_num_{short}"] = df[cur].apply(_get_len) - df[base].apply(_get_len)

    df["has_lost_website"] = (df["delta_num_websites"] < 0).astype(int)
    df["has_gained_social"] = (df["delta_num_socials"] > 0).astype(int)
    df["has_lost_social"] = (df["delta_num_socials"] < 0).astype(int)
    df["delta_total_contact"] = df["delta_num_websites"] + df["delta_num_socials"] + df["delta_num_phones"]
    df["has_any_loss"] = (
        (df["delta_num_websites"] < 0) | (df["delta_num_socials"] < 0) | (df["delta_num_phones"] < 0)
    ).astype(int)

    # ── 8. V2 INTERACTION FEATURES ────────────────────────────────────────
    print("  interaction features …")
    df["recency_x_loss"] = df["days_since_latest_update"] * df["has_any_loss"]
    df["recency_x_social_loss"] = df["days_since_latest_update"] * df["has_lost_social"]
    df["zombie_score"] = df["num_sources"] / (df["avg_days_since_update"] + 1)
    df["decay_velocity"] = df["delta_total_contact"] / (df["avg_days_since_update"] + 1)
    df["confidence_momentum"] = df["delta_confidence"] / (df["avg_days_since_update"] + 1)

    # ── 9. V2 CATEGORY CHURN RISK ─────────────────────────────────────────
    print("  category churn risk …")
    cat_churn = df.groupby("category_primary")["label"].agg(["mean", "count"])
    cat_churn["churn_rate"] = 1 - cat_churn["mean"]
    reliable = cat_churn[cat_churn["count"] >= 10]["churn_rate"].to_dict()
    median_churn = cat_churn[cat_churn["count"] >= 10]["churn_rate"].median()
    df["category_churn_risk"] = df["category_primary"].map(reliable).fillna(median_churn)

    # ── 10. V2 DIGITAL CONGRUENCE ─────────────────────────────────────────
    print("  digital congruence …")

    def _congruence(row):
        w = _parse_json(row["websites"])
        s = _parse_json(row["socials"])
        if not w or not s:
            return 0
        domain = str(w[0]).replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0].split(".")[0].lower()
        for soc in s:
            if isinstance(soc, str) and domain in soc.lower():
                return 1
        return 0

    df["digital_congruence"] = df.apply(_congruence, axis=1)

    # ── 11. V2 PCA ON RECENCY ─────────────────────────────────────────────
    print("  PCA on recency …")
    rec_feats = df[["days_since_latest_update", "avg_days_since_update"]].fillna(9999)
    pca = PCA(n_components=1)
    df["recency_pca"] = pca.fit_transform(rec_feats).flatten()
    print(f"    explained variance: {pca.explained_variance_ratio_[0]:.2%}")

    # ══════════════════════════════════════════════════════════════════════
    # ── V3 NEW FEATURES ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    # ── 12. RECENCY DECAY NON-LINEARITY ───────────────────────────────────
    print("  [V3] recency decay binning …")

    days = df["days_since_latest_update"].clip(upper=9999)

    # Log-transform (captures diminishing marginal effect of aging)
    df["log_days_since_update"] = np.log1p(days)

    # Staleness bins — the "cliff" where businesses typically disappear
    df["is_stale_6mo"] = (days > 180).astype(int)
    df["is_stale_1yr"] = (days > 365).astype(int)
    df["is_stale_2yr"] = (days > 730).astype(int)

    # Recency bucket (ordinal: 0=fresh, 1=aging, 2=stale, 3=dead)
    df["recency_bucket"] = pd.cut(
        days,
        bins=[-1, 90, 365, 730, 99999],
        labels=[0, 1, 2, 3],
    ).astype(int)

    print(f"    Staleness distribution:")
    print(f"      Fresh (<6mo):   {(~df['is_stale_6mo'].astype(bool)).sum():>5d} ({(~df['is_stale_6mo'].astype(bool)).mean():.1%})")
    print(f"      Stale 6mo-1yr:  {((df['is_stale_6mo']==1) & (df['is_stale_1yr']==0)).sum():>5d}")
    print(f"      Stale 1yr-2yr:  {((df['is_stale_1yr']==1) & (df['is_stale_2yr']==0)).sum():>5d}")
    print(f"      Dead (>2yr):    {df['is_stale_2yr'].sum():>5d}")

    # ── 13. BRAND-AWARE INTERACTION ───────────────────────────────────────
    print("  [V3] brand-aware features …")

    # Brands tolerate stale data better — a Starbucks with a 1-year-old update is fine
    df["brand_x_stale"] = df["is_brand"] * df["is_stale_1yr"]
    # Non-brand + stale = high risk
    df["nonbrand_stale_risk"] = (1 - df["is_brand"]) * df["is_stale_6mo"]

    # ══════════════════════════════════════════════════════════════════════
    # ── ASSEMBLE FINAL FEATURE SET ────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    print("  assembling final features …")
    df_ml = pd.concat([df, dummies], axis=1)

    feature_cols = [
        # Core metadata
        "confidence", "is_brand", "num_sources", "source_has_msft", "is_cross_verified",
        # Digital presence
        "has_website", "has_social", "has_phone", "contact_depth",
        # Category
        "cat_is_unknown", "category_churn_risk",
        # Delta
        "delta_confidence", "delta_num_socials", "delta_total_contact",
        "has_gained_social", "has_lost_social", "has_any_loss",
        # Recency (PCA replaces raw pair)
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
    ] + list(dummies.columns)

    target_col = "label"
    final_df = df_ml[feature_cols + [target_col]].copy()
    final_df = final_df.dropna(subset=[target_col])
    final_df.rename(columns={"label": "open"}, inplace=True)
    final_df["open"] = final_df["open"].astype(int)

    print("\n" + "=" * 80)
    print("V3 FINAL DATASET")
    print("=" * 80)
    print(f"  Shape:          {final_df.shape}")
    print(f"  Class Balance:  {final_df['open'].mean():.2%} Open")
    print(f"  Feature Count:  {len(feature_cols)}")
    v3_new = ["log_days_since_update", "is_stale_6mo", "is_stale_1yr",
              "is_stale_2yr", "recency_bucket", "brand_x_stale", "nonbrand_stale_risk"]
    print(f"  V3 New Features ({len(v3_new)}): {', '.join(v3_new)}")

    out = output_path
    print(f"\n  Saving → '{out}' …")
    final_df.to_parquet(out)
    print("  Done!")
    return final_df


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process data for ML V3")
    parser.add_argument("--input", "-i", type=str, default="data/Season 2 Samples 3k Project Updated.parquet", help="Input parquet file")
    parser.add_argument("--output", "-o", type=str, default="data/processed_for_ml_v3.parquet", help="Output parquet file")
    parser.add_argument("--merge", "-m", action="store_true", help="Merge with Project C")
    
    args = parser.parse_args()

    if args.merge:
        print("\n🔗  MERGE MODE: Season 2 + Project C")
    else:
        print(f"\n📊  PROCESSING: {args.input}")
    
    # Modify process_data_v3 signature to accept paths? 
    # Or start modifying the function itself.
    # The function process_data_v3 inside uses hardcoded paths in the original code. 
    # I should change the function definition as well.
    # But wait, replace_file_content is for a chunk. 
    # I need to modify the function start too. 
    # Let's do this in 2 steps or just one MultiReplace. 
    # Actually, I can just modify the __main__ block to call the function with new args, 
    # BUT I need to update the function signature first.
    
    # Let's just update the main block here, but it won't work unless I update the function.
    # So I should use MultiReplace.
    process_data_v3(use_project_c_merge=args.merge, input_path=args.input, output_path=args.output)
