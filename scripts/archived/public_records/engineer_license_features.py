"""
Engineer ML features from matched public records.
Takes the match results from match_records.py and creates
features that can be merged into the V5 training pipeline.
"""

import pandas as pd
import numpy as np
import json
import os
import argparse

OUTPUT_DIR = "data/public_records"

# Categories that typically require a license
LICENSE_REQUIRED_CATEGORIES = {
    # Food service
    "restaurant", "pizza_restaurant", "fast_food_restaurant", "coffee_shop",
    "bakery", "deli", "food_stand", "food_truck",
    # Alcohol
    "bar", "nightclub", "brewery", "winery", "wine_bar", "pub", "beer_garden",
    "liquor_store",
    # Personal care
    "hair_salon", "beauty_salon", "nail_salon", "barber_shop", "spa",
    "tattoo_parlor",
    # Health/professional
    "doctor", "dentist", "pharmacy", "veterinarian", "chiropractor",
    "optometrist", "physical_therapy",
    # Hospitality
    "hotel", "motel", "bed_and_breakfast", "hostel",
    # Automotive
    "gas_station", "automotive_repair", "car_wash", "auto_dealership",
    # Retail + specialty
    "convenience_store", "grocery_store", "supermarket",
    "pawn_shop", "tobacco_store",
    # Childcare
    "daycare", "child_care",
    # Finance
    "bank_credit_union", "banks",
    # Real estate / insurance
    "real_estate_agent", "insurance_agency",
}


def build_features(dataset_path, match_path=None, output_path=None):
    """
    Build license-based features for each POI.
    
    Args:
        dataset_path: Path to the StatusNow dataset
        match_path: Path to match results from match_records.py
        output_path: Where to save the feature-augmented dataset
    """
    if match_path is None:
        match_path = os.path.join(OUTPUT_DIR, "poi_matches.parquet")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "license_features.parquet")

    # Load original dataset for category info
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} POIs from {dataset_path}")

    # Extract categories and labels
    poi_info = []
    for _, row in df.iterrows():
        try:
            cats = json.loads(row["categories"]) if isinstance(row["categories"], str) else row["categories"]
            primary_cat = cats.get("primary", "") if isinstance(cats, dict) else ""
        except:
            primary_cat = ""
        
        try:
            addrs = json.loads(row["addresses"]) if isinstance(row["addresses"], str) else row["addresses"]
            if isinstance(addrs, list) and len(addrs) > 0:
                country = addrs[0].get("country", "")
            else:
                country = ""
        except:
            country = ""

        poi_info.append({
            "poi_id": row["id"],
            "poi_category": primary_cat,
            "poi_label": row.get("label", None),
            "poi_country": country,
        })

    poi_df = pd.DataFrame(poi_info)

    # Load matches
    if not os.path.exists(match_path):
        print(f"❌ Match file not found: {match_path}")
        print("  Run match_records.py first!")
        return None

    matches = pd.read_parquet(match_path)
    print(f"Loaded {len(matches):,} matches from {match_path}")

    # Build per-POI feature columns
    features = poi_df[["poi_id", "poi_category", "poi_label", "poi_country"]].copy()

    # --- Binary signals ---

    # 1. has_any_match: POI matched to at least one public record
    matched_ids = set(matches["poi_id"].unique())
    features["has_license_match"] = features["poi_id"].isin(matched_ids).astype(int)

    # 2. has_high_confidence_match
    high_ids = set(matches[matches["match_confidence"] == "HIGH"]["poi_id"].unique())
    features["has_high_confidence_match"] = features["poi_id"].isin(high_ids).astype(int)

    # 3. Per-source match flags
    for source in matches["match_source"].unique():
        source_ids = set(matches[matches["match_source"] == source]["poi_id"].unique())
        col = f"has_{source}_match"
        features[col] = features["poi_id"].isin(source_ids).astype(int)

    # 4. License status signals (from DCWP and SLA)
    for source in ["nyc_dcwp", "nyc_sla"]:
        source_matches = matches[matches["match_source"] == source]
        if len(source_matches) == 0:
            continue

        # Active license
        active_col = [c for c in source_matches.columns if "active" in c.lower() or "status" in c.lower()]
        if active_col:
            active_ids = set(source_matches[
                source_matches[active_col[0]].astype(str).str.lower().isin(["true", "1", "active"])
            ]["poi_id"].unique())
            features[f"has_active_{source}_license"] = features["poi_id"].isin(active_ids).astype(int)

        # Expired license
        expiry_cols = [c for c in source_matches.columns if "expir" in c.lower() or "end" in c.lower()]
        if expiry_cols:
            expired_ids = set()
            for exp_col in expiry_cols:
                try:
                    exp_dates = pd.to_datetime(source_matches[exp_col], errors="coerce")
                    expired_mask = exp_dates < pd.Timestamp.now()
                    expired_ids.update(source_matches[expired_mask]["poi_id"].unique())
                except:
                    pass
            features[f"has_expired_{source}_license"] = features["poi_id"].isin(expired_ids).astype(int)

    # 5. DOHMH inspection signals
    dohmh_matches = matches[matches["match_source"] == "nyc_dohmh"]
    if len(dohmh_matches) > 0:
        inspected_ids = set(dohmh_matches["poi_id"].unique())
        features["has_health_inspection"] = features["poi_id"].isin(inspected_ids).astype(int)

        # Grade signal
        grade_col = [c for c in dohmh_matches.columns if "grade" in c.lower()]
        if grade_col:
            a_grade_ids = set(dohmh_matches[dohmh_matches[grade_col[0]] == "A"]["poi_id"].unique())
            features["has_grade_a"] = features["poi_id"].isin(a_grade_ids).astype(int)

    # 6. SF business closure signal
    sf_matches = matches[matches["match_source"] == "sf_business"]
    if len(sf_matches) > 0:
        closed_col = [c for c in sf_matches.columns if "closed" in c.lower() or "end" in c.lower()]
        if closed_col:
            closed_ids = set(sf_matches[
                sf_matches[closed_col[0]].astype(str).str.lower().isin(["true", "1"])
            ]["poi_id"].unique())
            features["has_sf_closed_registration"] = features["poi_id"].isin(closed_ids).astype(int)

    # 7. license_expected_but_not_found
    features["license_expected"] = features["poi_category"].isin(LICENSE_REQUIRED_CATEGORIES).astype(int)
    features["license_not_found"] = (
        (features["license_expected"] == 1) & 
        (features["has_license_match"] == 0) &
        (features["poi_country"] == "US")
    ).astype(int)

    # --- Continuous features ---

    # Count of matched sources
    source_counts = matches.groupby("poi_id")["match_source"].nunique().reset_index()
    source_counts.columns = ["poi_id", "num_license_sources"]
    features = features.merge(source_counts, on="poi_id", how="left")
    features["num_license_sources"] = features["num_license_sources"].fillna(0).astype(int)

    # Best match ratio
    best_ratio = matches.groupby("poi_id")["match_ratio"].max().reset_index()
    best_ratio.columns = ["poi_id", "best_match_ratio"]
    features = features.merge(best_ratio, on="poi_id", how="left")
    features["best_match_ratio"] = features["best_match_ratio"].fillna(0)

    # --- Filter to US only ---
    us_features = features[features["poi_country"] == "US"].copy()

    # --- Correlation analysis ---
    print(f"\n{'='*60}")
    print(f"FEATURE ANALYSIS (US POIs only)")
    print(f"{'='*60}")
    print(f"  Total US POIs: {len(us_features):,}")

    feature_cols = [c for c in us_features.columns 
                    if c not in ("poi_id", "poi_category", "poi_label", "poi_country")]

    print(f"\n  Feature correlations with label (Open=1, Closed=0):")
    for col in sorted(feature_cols):
        if us_features[col].nunique() > 1:
            corr = us_features[col].corr(us_features["poi_label"])
            coverage = (us_features[col] > 0).mean() if us_features[col].dtype in [int, float, np.float64, np.int64] else 0
            print(f"    {col:40s} r={corr:+.4f}   coverage={100*coverage:.1f}%")

    # Save
    output_features = features[["poi_id"] + feature_cols]
    output_features.to_parquet(output_path, index=False)
    print(f"\n  ✅ Saved {len(feature_cols)} features to {output_path}")

    return output_features


def main():
    parser = argparse.ArgumentParser(description="Engineer features from public record matches")
    parser.add_argument("-i", "--input", type=str,
                        default="data/combined_truth_dataset_all.parquet")
    parser.add_argument("--matches", type=str,
                        default=os.path.join(OUTPUT_DIR, "poi_matches.parquet"))
    parser.add_argument("-o", "--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "license_features.parquet"))
    args = parser.parse_args()

    build_features(args.input, args.matches, args.output)


if __name__ == "__main__":
    main()
