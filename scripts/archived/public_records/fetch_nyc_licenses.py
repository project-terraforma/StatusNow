"""
Fetch NYC DCWP (Dept of Consumer & Worker Protection) issued business licenses.
Source: NYC Open Data — Socrata API
Dataset: "Legally Operating Businesses" (w7w3-xahh)
No auth required — public API with pagination.
"""

import requests
import pandas as pd
import os
import time
import argparse

API_URL = "https://data.cityofnewyork.us/resource/w7w3-xahh.json"
OUTPUT_DIR = "data/public_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nyc_dcwp_licenses.parquet")

# Socrata defaults to 1000 rows; max $limit = 50000
PAGE_SIZE = 50000


def fetch_all_licenses(limit=None):
    """
    Paginate through the full DCWP dataset.
    Returns a DataFrame with all records.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    offset = 0
    total_fetched = 0

    print("Fetching NYC DCWP business licenses...")
    print(f"  API: {API_URL}")
    print(f"  Page size: {PAGE_SIZE:,}")
    print()

    while True:
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": ":id",  # stable pagination order
        }

        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ❌ Request failed at offset {offset}: {e}")
            break

        batch = resp.json()
        if not batch:
            break

        all_records.extend(batch)
        total_fetched += len(batch)
        print(f"  Fetched {total_fetched:>8,} records (offset={offset})")

        if limit and total_fetched >= limit:
            print(f"  Hit limit ({limit}), stopping.")
            break

        if len(batch) < PAGE_SIZE:
            break  # last page

        offset += PAGE_SIZE
        time.sleep(0.5)  # be nice to the API

    print(f"\n  Total raw records: {len(all_records):,}")

    if not all_records:
        print("  ❌ No records fetched!")
        return None

    df = pd.DataFrame(all_records)
    return df


def clean_licenses(df):
    """Normalize and clean the raw license data for matching."""

    # Keep only relevant columns
    keep_cols = [
        "business_name", "business_name_2",
        "address_building", "address_street_name", "address_city",
        "address_state", "address_zip",
        "license_type", "license_category",
        "lic_expir_dd",  # expiration date
        "license_status",
        "industry",
        "detail",  # sub-type detail
        "longitude", "latitude",
    ]

    existing_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[existing_cols].copy()

    # Normalize business name for matching
    def normalize_name(name):
        if pd.isna(name):
            return ""
        name = str(name).upper().strip()
        # Remove common suffixes
        for suffix in [" LLC", " INC", " CORP", " LTD", " CO", " DBA"]:
            name = name.replace(suffix, "")
        # Remove punctuation
        for ch in ["&", "'", ".", ",", "-", "#", "/"]:
            name = name.replace(ch, " ")
        # Collapse whitespace
        return " ".join(name.split())

    def normalize_street(street):
        if pd.isna(street):
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
        }
        for old, new in replacements.items():
            street = street.replace(old, new)
        for ch in [".", ",", "#", "-"]:
            street = street.replace(ch, " ")
        return " ".join(street.split())

    df_clean["name_normalized"] = df_clean["business_name"].apply(normalize_name)

    # Build full normalized address
    df_clean["street_normalized"] = (
        df_clean["address_building"].fillna("").astype(str) + " " +
        df_clean["address_street_name"].fillna("").astype(str)
    ).apply(normalize_street)

    df_clean["zip5"] = df_clean["address_zip"].astype(str).str[:5]

    # Parse expiration date
    df_clean["expiration_date"] = pd.to_datetime(
        df_clean.get("lic_expir_dd"), errors="coerce"
    )

    # Classify status
    df_clean["is_active"] = df_clean.get("license_status", pd.Series(dtype=str)).str.upper().isin(["ACTIVE", ""])
    df_clean["is_expired"] = df_clean["expiration_date"] < pd.Timestamp.now()

    print(f"\n  Cleaned records: {len(df_clean):,}")
    print(f"  Active: {df_clean['is_active'].sum():,}")
    print(f"  Expired: {df_clean['is_expired'].sum():,}")
    print(f"  Unique businesses (by name): {df_clean['name_normalized'].nunique():,}")
    print(f"  Unique ZIPs: {df_clean['zip5'].nunique():,}")

    # License type distribution
    if "license_type" in df_clean.columns:
        print(f"\n  Top license types:")
        for lt, cnt in df_clean["license_type"].value_counts().head(10).items():
            print(f"    {lt:30s} {cnt:>6,}")

    if "industry" in df_clean.columns:
        print(f"\n  Top industries:")
        for ind, cnt in df_clean["industry"].value_counts().head(15).items():
            print(f"    {ind:40s} {cnt:>6,}")

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Fetch NYC DCWP business licenses")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total records fetched (for testing)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetch; just re-clean from cached JSON")
    args = parser.parse_args()

    df_raw = fetch_all_licenses(limit=args.limit)

    if df_raw is None:
        return

    df_clean = clean_licenses(df_raw)

    # Save
    df_clean.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
