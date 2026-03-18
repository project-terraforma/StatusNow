"""
Fuzzy matching engine — match StatusNow POIs to public records.

Matching hierarchy:
  1. Exact:  normalized_name + normalized_street + zip5  → HIGH confidence
  2. Fuzzy:  Levenshtein ratio ≥ 0.85 on name, exact street+zip → MEDIUM confidence
  3. Address-only: exact street+zip, category alignment → LOW confidence
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

OUTPUT_DIR = "data/public_records"


# ──────────────────────────────────────────────
# Normalization utils (shared across all scripts)
# ──────────────────────────────────────────────

def normalize_name(name):
    """Normalize a business name for matching."""
    if pd.isna(name) or name is None:
        return ""
    name = str(name).upper().strip()
    for suffix in [" LLC", " INC", " CORP", " LTD", " CO", " DBA"]:
        name = name.replace(suffix, "")
    for ch in ["&", "'", ".", ",", "-", "#", "/"]:
        name = name.replace(ch, " ")
    return " ".join(name.split())


def normalize_street(street):
    """Normalize a street address for matching."""
    if pd.isna(street) or street is None:
        return ""
    street = str(street).upper().strip()
    replacements = {
        "AVENUE": "AVE", "STREET": "ST", "BOULEVARD": "BLVD",
        "DRIVE": "DR", "ROAD": "RD", "PLACE": "PL",
        "LANE": "LN", "COURT": "CT", "TERRACE": "TER",
        "PARKWAY": "PKWY", "HIGHWAY": "HWY",
        "1ST": "1", "2ND": "2", "3RD": "3",
        "4TH": "4", "5TH": "5", "6TH": "6",
        "7TH": "7", "8TH": "8", "9TH": "9",
        "10TH": "10", "11TH": "11", "12TH": "12",
        "13TH": "13",
    }
    for old, new in replacements.items():
        street = street.replace(old, new)
    for ch in [".", ",", "#", "-"]:
        street = street.replace(ch, " ")
    return " ".join(street.split())


def fuzzy_ratio(a, b):
    """Quick fuzzy match ratio (0–1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ──────────────────────────────────────────────
# POI extraction from StatusNow dataset
# ──────────────────────────────────────────────

def extract_pois(dataset_path):
    """
    Load the StatusNow dataset and extract name, address, category, label
    for matching against public records.
    """
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} POIs from {dataset_path}")

    records = []
    for idx, row in df.iterrows():
        # Parse name
        try:
            names = json.loads(row["names"]) if isinstance(row["names"], str) else row["names"]
            primary_name = names.get("primary", "") if isinstance(names, dict) else ""
        except (json.JSONDecodeError, AttributeError):
            primary_name = ""

        # Parse address
        try:
            addrs = json.loads(row["addresses"]) if isinstance(row["addresses"], str) else row["addresses"]
            if isinstance(addrs, list) and len(addrs) > 0:
                addr = addrs[0]
            elif isinstance(addrs, dict):
                addr = addrs
            else:
                addr = {}
        except (json.JSONDecodeError, AttributeError):
            addr = {}

        # Parse category
        try:
            cats = json.loads(row["categories"]) if isinstance(row["categories"], str) else row["categories"]
            primary_cat = cats.get("primary", "") if isinstance(cats, dict) else ""
        except (json.JSONDecodeError, AttributeError):
            primary_cat = ""

        country = addr.get("country", "")
        if country != "US":
            continue  # skip non-US

        records.append({
            "poi_id": row["id"],
            "poi_name": primary_name,
            "poi_name_norm": normalize_name(primary_name),
            "poi_street": addr.get("freeform", ""),
            "poi_street_norm": normalize_street(addr.get("freeform", "")),
            "poi_locality": addr.get("locality", ""),
            "poi_region": addr.get("region", ""),
            "poi_zip": str(addr.get("postcode", ""))[:5],
            "poi_category": primary_cat,
            "poi_label": row.get("label", None),
            "source_dataset": row.get("source_dataset", ""),
        })

    pois = pd.DataFrame(records)
    print(f"  US POIs extracted: {len(pois):,}")
    print(f"  States: {pois['poi_region'].value_counts().head(5).to_dict()}")
    return pois


# ──────────────────────────────────────────────
# Matching logic
# ──────────────────────────────────────────────

def build_index(records_df, name_col="name_normalized", street_col="street_normalized", zip_col="zip5"):
    """Build a lookup index: (zip5, street_norm) → list of record rows."""
    index = defaultdict(list)
    for idx, row in records_df.iterrows():
        key = (str(row.get(zip_col, ""))[:5], str(row.get(street_col, "")))
        index[key].append(row)
    return index


def match_pois_to_records(pois, records_df, source_name,
                          name_col="name_normalized",
                          street_col="street_normalized",
                          zip_col="zip5"):
    """
    Match POIs to public records using the 3-tier hierarchy.
    Returns a DataFrame with match results.
    """
    print(f"\nMatching {len(pois):,} POIs against {len(records_df):,} {source_name} records...")

    # Build address index
    addr_index = build_index(records_df, name_col, street_col, zip_col)

    matches = []
    for _, poi in pois.iterrows():
        poi_zip = poi["poi_zip"]
        poi_street = poi["poi_street_norm"]
        poi_name = poi["poi_name_norm"]

        if not poi_zip or not poi_street:
            continue

        key = (poi_zip, poi_street)
        candidates = addr_index.get(key, [])

        best_match = None
        best_confidence = None
        best_ratio = 0

        for rec in candidates:
            rec_name = str(rec.get(name_col, ""))
            ratio = fuzzy_ratio(poi_name, rec_name)

            if poi_name == rec_name and poi_name:
                # Tier 1: exact name + exact address
                if best_confidence != "HIGH" or ratio > best_ratio:
                    best_match = rec
                    best_confidence = "HIGH"
                    best_ratio = ratio
            elif ratio >= 0.85:
                # Tier 2: fuzzy name + exact address
                if best_confidence not in ("HIGH",) or ratio > best_ratio:
                    best_match = rec
                    best_confidence = "MEDIUM"
                    best_ratio = ratio
            elif ratio >= 0.5:
                # Tier 3: partial name + exact address
                if best_confidence not in ("HIGH", "MEDIUM"):
                    best_match = rec
                    best_confidence = "LOW"
                    best_ratio = ratio

        if best_match is not None:
            matches.append({
                "poi_id": poi["poi_id"],
                "match_source": source_name,
                "match_confidence": best_confidence,
                "match_ratio": best_ratio,
                "match_name": best_match.get(name_col, ""),
                **{f"rec_{k}": v for k, v in best_match.items()
                   if k not in (name_col, street_col, zip_col)},
            })

    match_df = pd.DataFrame(matches)
    n_high = (match_df["match_confidence"] == "HIGH").sum() if len(match_df) else 0
    n_med = (match_df["match_confidence"] == "MEDIUM").sum() if len(match_df) else 0
    n_low = (match_df["match_confidence"] == "LOW").sum() if len(match_df) else 0
    print(f"  Matches: {len(match_df):,} / {len(pois):,} ({100 * len(match_df) / max(len(pois), 1):.1f}%)")
    print(f"    HIGH: {n_high:,}  MEDIUM: {n_med:,}  LOW: {n_low:,}")

    return match_df


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

def run_all_matching(dataset_path, records_dir=OUTPUT_DIR):
    """Run matching against all available public record sources."""

    pois = extract_pois(dataset_path)

    # Split POIs by region for efficient matching
    nyc_pois = pois[pois["poi_region"].isin(["NY", "NJ", "CT"])].copy()
    sf_pois = pois[pois["poi_region"].isin(["CA"])].copy()

    all_matches = []

    # 1. NYC DCWP Business Licenses
    dcwp_path = os.path.join(records_dir, "nyc_dcwp_licenses.parquet")
    if os.path.exists(dcwp_path):
        dcwp = pd.read_parquet(dcwp_path)
        m = match_pois_to_records(nyc_pois, dcwp, "nyc_dcwp")
        all_matches.append(m)
    else:
        print(f"  ⏭️  Skipping NYC DCWP (not found: {dcwp_path})")

    # 2. NYC SLA Liquor Licenses
    sla_path = os.path.join(records_dir, "nyc_sla_licenses.parquet")
    if os.path.exists(sla_path):
        sla = pd.read_parquet(sla_path)
        m = match_pois_to_records(nyc_pois, sla, "nyc_sla")
        all_matches.append(m)
    else:
        print(f"  ⏭️  Skipping NYC SLA (not found: {sla_path})")

    # 3. NYC DOHMH Inspections
    dohmh_path = os.path.join(records_dir, "nyc_dohmh_inspections.parquet")
    if os.path.exists(dohmh_path):
        dohmh = pd.read_parquet(dohmh_path)
        m = match_pois_to_records(nyc_pois, dohmh, "nyc_dohmh")
        all_matches.append(m)
    else:
        print(f"  ⏭️  Skipping NYC DOHMH (not found: {dohmh_path})")

    # 4. SF Registered Businesses
    sf_path = os.path.join(records_dir, "sf_businesses.parquet")
    if os.path.exists(sf_path):
        sf_biz = pd.read_parquet(sf_path)
        m = match_pois_to_records(sf_pois, sf_biz, "sf_business")
        all_matches.append(m)
    else:
        print(f"  ⏭️  Skipping SF Businesses (not found: {sf_path})")

    # 5. CA ABC Liquor Licenses
    abc_path = os.path.join(records_dir, "ca_abc_licenses.parquet")
    if os.path.exists(abc_path):
        abc = pd.read_parquet(abc_path)
        m = match_pois_to_records(sf_pois, abc, "ca_abc")
        all_matches.append(m)
    else:
        print(f"  ⏭️  Skipping CA ABC (not found: {abc_path})")

    # Combine all matches
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"COMBINED MATCH RESULTS")
        print(f"{'='*60}")
        print(f"  Total matches: {len(combined):,}")
        print(f"  Unique POIs matched: {combined['poi_id'].nunique():,} / {len(pois):,}")
        print(f"\n  By source:")
        for src, cnt in combined["match_source"].value_counts().items():
            print(f"    {src:20s} {cnt:>6,}")
        print(f"\n  By confidence:")
        for conf, cnt in combined["match_confidence"].value_counts().items():
            print(f"    {conf:10s} {cnt:>6,}")

        # Save
        output_path = os.path.join(records_dir, "poi_matches.parquet")
        combined.to_parquet(output_path, index=False)
        print(f"\n  ✅ Saved to {output_path}")
        return combined
    else:
        print("  ❌ No match sources available!")
        return None


def main():
    parser = argparse.ArgumentParser(description="Match StatusNow POIs to public records")
    parser.add_argument("-i", "--input", type=str,
                        default="data/combined_truth_dataset_all.parquet",
                        help="Path to StatusNow dataset")
    parser.add_argument("--records-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    run_all_matching(args.input, args.records_dir)


if __name__ == "__main__":
    main()
