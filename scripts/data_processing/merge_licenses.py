"""
merge_licenses.py -- Left-join public license data onto the existing truth dataset.
Matches on normalized business name + zip code, with street-number tiebreaking.
Only enriches existing rows; never adds new ones.
"""

import os
import sys
import io
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TRUTH_FILE = "data/combined_truth_dataset_all.parquet"
SF_LICENSES = "data/licenses/sf_licenses.parquet"
NYC_LICENSES = "data/licenses/nyc_licenses.parquet"
OUTPUT_FILE = "data/combined_truth_dataset_all.parquet"

REFERENCE_DATE = datetime(2026, 2, 23)

# Suffixes to strip during name normalization
STRIP_SUFFIXES = re.compile(
    r"\b(inc|llc|corp|co|ltd|l\.?l\.?c|incorporated|corporation|company|limited|"
    r"the|dba|d/b/a|and|of)\b", re.IGNORECASE
)
STRIP_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalize_name(name):
    """Normalize a business name for matching."""
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    name = str(name).lower().strip()
    name = STRIP_PUNCT.sub(" ", name)
    name = STRIP_SUFFIXES.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_zip(z):
    """Extract 5-digit zip code."""
    if not z or (isinstance(z, float) and np.isnan(z)):
        return ""
    z = str(z).strip()
    # Handle "10001-1234" format
    match = re.match(r"(\d{5})", z)
    return match.group(1) if match else z


def extract_street_number(addr):
    """Pull leading street number from an address string."""
    if not addr or (isinstance(addr, float) and np.isnan(addr)):
        return ""
    match = re.match(r"(\d+)", str(addr).strip())
    return match.group(1) if match else ""


def parse_json_field(x):
    """Parse a JSON-encoded string field."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def extract_primary_name(names_field):
    """Extract the primary business name from Overture's names JSON."""
    data = parse_json_field(names_field)
    if isinstance(data, dict):
        return data.get("primary", "")
    return ""


def extract_address_parts(addresses_field):
    """Extract freeform address and postcode from Overture's addresses JSON."""
    data = parse_json_field(addresses_field)
    if isinstance(data, list) and len(data) > 0:
        addr = data[0] if isinstance(data[0], dict) else {}
        return addr.get("freeform", ""), addr.get("postcode", "")
    if isinstance(data, dict):
        return data.get("freeform", ""), data.get("postcode", "")
    return "", ""


def compute_license_features(row):
    """Compute derived license features from matched license record."""
    features = {}

    # license_active
    status = str(row.get("lic_status", "")).lower()
    features["license_active"] = 1 if status in ("active", "") else 0

    # days_to_license_expiry
    end_date = row.get("lic_end_date")
    if pd.notna(end_date):
        try:
            dt = pd.to_datetime(end_date)
            features["days_to_license_expiry"] = (dt - REFERENCE_DATE).days
        except Exception:
            features["days_to_license_expiry"] = np.nan
    else:
        features["days_to_license_expiry"] = np.nan

    # license_age_days
    start_date = row.get("lic_start_date")
    if pd.notna(start_date):
        try:
            dt = pd.to_datetime(start_date)
            features["license_age_days"] = (REFERENCE_DATE - dt).days
        except Exception:
            features["license_age_days"] = np.nan
    else:
        features["license_age_days"] = np.nan

    return features


def prepare_truth_data(df):
    """Extract normalized join keys from the truth dataset."""
    print("  Extracting names and addresses from truth dataset ...")

    names_addrs = df[["names", "addresses"]].apply(
        lambda row: pd.Series({
            "raw_name": extract_primary_name(row["names"]),
            "raw_addr": extract_address_parts(row["addresses"])[0],
            "raw_zip":  extract_address_parts(row["addresses"])[1],
        }),
        axis=1,
    )

    df["_norm_name"] = names_addrs["raw_name"].apply(normalize_name)
    df["_norm_zip"] = names_addrs["raw_zip"].apply(normalize_zip)
    df["_street_num"] = names_addrs["raw_addr"].apply(extract_street_number)

    return df


def prepare_license_data(sf_df, nyc_df):
    """Normalize and combine license datasets."""
    frames = []

    if sf_df is not None and len(sf_df) > 0:
        print(f"  Preparing SF licenses ({len(sf_df):,} rows) ...")
        sf = sf_df.copy()
        sf["_norm_name"] = sf["business_name"].apply(normalize_name)
        sf["_norm_zip"] = sf["zip"].apply(normalize_zip)
        sf["_street_num"] = sf["address"].apply(extract_street_number)

        # SF doesn't have explicit license status -- infer from end date
        sf["lic_status"] = sf["license_end_date"].apply(
            lambda x: "Active" if pd.notna(x) and pd.to_datetime(x, errors="coerce") and
                       pd.to_datetime(x, errors="coerce") >= REFERENCE_DATE else "Expired"
        )
        sf["lic_end_date"] = sf["license_end_date"]
        sf["lic_start_date"] = sf["license_start_date"]
        sf["lic_category"] = sf.get("license_category", pd.Series(dtype=str))
        frames.append(sf[["_norm_name", "_norm_zip", "_street_num",
                          "lic_status", "lic_end_date", "lic_start_date",
                          "lic_category", "city_source"]])

    if nyc_df is not None and len(nyc_df) > 0:
        print(f"  Preparing NYC licenses ({len(nyc_df):,} rows) ...")
        nyc = nyc_df.copy()
        nyc["_norm_name"] = nyc["business_name"].apply(normalize_name)
        nyc["_norm_zip"] = nyc["zip"].apply(normalize_zip)
        nyc["_street_num"] = nyc["address"].apply(extract_street_number)
        nyc["lic_status"] = nyc["license_status"]
        nyc["lic_end_date"] = nyc["license_end_date"]
        nyc["lic_start_date"] = nyc["license_start_date"]
        nyc["lic_category"] = nyc.get("license_category", pd.Series(dtype=str))
        frames.append(nyc[["_norm_name", "_norm_zip", "_street_num",
                           "lic_status", "lic_end_date", "lic_start_date",
                           "lic_category", "city_source"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Filter out empty names
    combined = combined[combined["_norm_name"].str.len() > 0].copy()

    return combined


def match_and_merge(truth_df, license_df):
    """Left-join license data onto truth dataset using name+zip with tiebreaking."""
    print("\n  Matching licenses to truth dataset ...")

    # Step 1: Group license records by (name, zip)
    # For each group, count records and pick the "best" one
    license_grouped = license_df.groupby(["_norm_name", "_norm_zip"])

    # Build a lookup dict: (name, zip) -> list of license records
    lookup = {}
    for (name, zipcode), group in license_grouped:
        if name and zipcode:
            lookup[(name, zipcode)] = group.to_dict("records")

    print(f"  License lookup: {len(lookup):,} unique (name, zip) keys")

    # Step 2: For each truth row, try to find a match
    results = []
    matched = 0
    total_eligible = 0

    for idx, row in truth_df.iterrows():
        src = row.get("source_dataset", "")
        if src not in ("overture_nyc", "overture_sf"):
            # Season2 rows -- skip matching
            results.append({
                "license_status": None,
                "license_active": np.nan,
                "days_to_license_expiry": np.nan,
                "license_category": None,
                "license_age_days": np.nan,
                "license_count": 0,
            })
            continue

        total_eligible += 1
        key = (row["_norm_name"], row["_norm_zip"])
        candidates = lookup.get(key, [])

        if not candidates:
            results.append({
                "license_status": None,
                "license_active": np.nan,
                "days_to_license_expiry": np.nan,
                "license_category": None,
                "license_age_days": np.nan,
                "license_count": 0,
            })
            continue

        # Tiebreak: prefer candidate whose street number matches
        best = candidates[0]
        truth_sn = row["_street_num"]
        if len(candidates) > 1 and truth_sn:
            for c in candidates:
                if c.get("_street_num") == truth_sn:
                    best = c
                    break
            else:
                # No street match -- pick most recent by start date
                try:
                    candidates_sorted = sorted(
                        candidates,
                        key=lambda c: pd.to_datetime(c.get("lic_start_date", "1900-01-01"),
                                                      errors="coerce") or pd.Timestamp.min,
                        reverse=True,
                    )
                    best = candidates_sorted[0]
                except Exception:
                    pass

        feats = compute_license_features(best)
        feats["license_status"] = best.get("lic_status")
        feats["license_category"] = best.get("lic_category")
        feats["license_count"] = len(candidates)
        results.append(feats)
        matched += 1

    print(f"  Matched: {matched:,} / {total_eligible:,} eligible "
          f"({matched/max(total_eligible,1):.1%})")

    result_df = pd.DataFrame(results, index=truth_df.index)
    return result_df


def main():
    print("=" * 60)
    print("  License Data Merge")
    print("=" * 60)

    # Load truth dataset
    print(f"\nLoading truth dataset: {TRUTH_FILE}")
    truth_df = pd.read_parquet(TRUTH_FILE)
    original_count = len(truth_df)
    print(f"  Rows: {original_count:,}")

    # Load license data
    sf_df = None
    nyc_df = None
    if os.path.exists(SF_LICENSES):
        sf_df = pd.read_parquet(SF_LICENSES)
        print(f"  SF licenses: {len(sf_df):,}")
    else:
        print(f"  [WARN] {SF_LICENSES} not found -- skipping SF")

    if os.path.exists(NYC_LICENSES):
        nyc_df = pd.read_parquet(NYC_LICENSES)
        print(f"  NYC licenses: {len(nyc_df):,}")
    else:
        print(f"  [WARN] {NYC_LICENSES} not found -- skipping NYC")

    # Prepare data
    print("\nPreparing data ...")
    truth_df = prepare_truth_data(truth_df)
    license_df = prepare_license_data(sf_df, nyc_df)

    if license_df.empty:
        print("  No license data available. Exiting.")
        return

    print(f"  Combined license records: {len(license_df):,}")

    # Match and merge
    enrichment = match_and_merge(truth_df, license_df)

    # Add new columns to truth dataset
    new_cols = ["license_status", "license_active", "days_to_license_expiry",
                "license_category", "license_age_days", "license_count"]
    for col in new_cols:
        truth_df[col] = enrichment[col]

    # Drop internal join columns
    truth_df = truth_df.drop(columns=["_norm_name", "_norm_zip", "_street_num"])

    # Verify row count unchanged
    assert len(truth_df) == original_count, \
        f"Row count changed! {original_count} -> {len(truth_df)}"

    # Stats
    print(f"\n{'=' * 60}")
    print("  MERGE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Output rows: {len(truth_df):,} (unchanged)")
    print(f"  New columns: {new_cols}")
    print()

    for src in ["overture_nyc", "overture_sf", "season2"]:
        sub = truth_df[truth_df["source_dataset"] == src]
        matched = (sub["license_count"] > 0).sum()
        print(f"  {src:>15s}: {matched:,} / {len(sub):,} matched "
              f"({matched/max(len(sub),1):.1%})")

    print()
    matched_total = (truth_df["license_count"] > 0).sum()
    active_total = (truth_df["license_active"] == 1).sum()
    print(f"  Total matched: {matched_total:,} / {len(truth_df):,} "
          f"({matched_total/len(truth_df):.1%})")
    print(f"  Active licenses: {active_total:,}")

    # Save
    print(f"\n  Saving -> {OUTPUT_FILE}")
    truth_df.to_parquet(OUTPUT_FILE, index=False)
    print("  Done!")


if __name__ == "__main__":
    main()
