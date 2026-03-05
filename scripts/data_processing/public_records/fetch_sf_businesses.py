"""
Fetch SF registered business locations.
Source: DataSF — Socrata API
Dataset: "Registered Business Locations - San Francisco" (g8m3-pdis)
"""

import requests
import pandas as pd
import os
import time
import argparse

API_URL = "https://data.sfgov.org/resource/g8m3-pdis.json"
OUTPUT_DIR = "data/public_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sf_businesses.parquet")
PAGE_SIZE = 50000


def fetch_sf_businesses(limit=None):
    """Paginate through SF registered business locations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records = []
    offset = 0

    print("Fetching SF registered business locations...")
    print(f"  API: {API_URL}")
    print()

    while True:
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": ":id",
        }

        try:
            resp = requests.get(API_URL, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ❌ Request failed at offset {offset}: {e}")
            break

        batch = resp.json()
        if not batch:
            break

        all_records.extend(batch)
        print(f"  Fetched {len(all_records):>8,} records")

        if limit and len(all_records) >= limit:
            break

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)

    print(f"\n  Total raw records: {len(all_records):,}")

    if not all_records:
        print("  ❌ No records fetched!")
        return None

    return pd.DataFrame(all_records)


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    for suffix in [" LLC", " INC", " CORP", " LTD", " CO", " DBA"]:
        name = name.replace(suffix, "")
    for ch in ["&", "'", ".", ",", "-", "#", "/"]:
        name = name.replace(ch, " ")
    return " ".join(name.split())


def normalize_street(street):
    if pd.isna(street):
        return ""
    street = str(street).upper().strip()
    replacements = {
        "AVENUE": "AVE", "STREET": "ST", "BOULEVARD": "BLVD",
        "DRIVE": "DR", "ROAD": "RD", "PLACE": "PL",
        "LANE": "LN", "COURT": "CT",
    }
    for old, new in replacements.items():
        street = street.replace(old, new)
    for ch in [".", ",", "#", "-"]:
        street = street.replace(ch, " ")
    return " ".join(street.split())


def clean_sf_businesses(df):
    """Normalize SF business data for matching."""

    # Use DBA name (doing-business-as) for matching — this is the public-facing name
    name_col = "dba_name" if "dba_name" in df.columns else "ownership_name"
    df["name_normalized"] = df.get(name_col, pd.Series(dtype=str)).apply(normalize_name)

    # Address
    addr_col = "full_business_address" if "full_business_address" in df.columns else "street_address"
    df["street_normalized"] = df.get(addr_col, pd.Series(dtype=str)).apply(normalize_street)

    # ZIP
    mail_zip = "business_zip" if "business_zip" in df.columns else "mailing_address_zip_code"
    if mail_zip in df.columns:
        df["zip5"] = df[mail_zip].astype(str).str[:5]
    else:
        df["zip5"] = ""

    # Parse dates
    for date_col in ["business_start_date", "business_end_date",
                      "location_start_date", "location_end_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Status flags
    if "location_end_date" in df.columns:
        df["is_closed"] = df["location_end_date"].notna()
    else:
        df["is_closed"] = False

    if "business_end_date" in df.columns:
        df["business_ended"] = df["business_end_date"].notna()
    else:
        df["business_ended"] = False

    print(f"\n  Cleaned records: {len(df):,}")
    print(f"  With location_end_date (closed): {df['is_closed'].sum():,}")
    print(f"  With business_end_date: {df['business_ended'].sum():,}")
    print(f"  Unique business names: {df['name_normalized'].nunique():,}")

    # NAICS distribution
    if "naic_code" in df.columns:
        print(f"\n  Top NAICS codes:")
        for code, cnt in df["naic_code"].value_counts().head(10).items():
            print(f"    {str(code):10s} {cnt:>6,}")

    if "naic_code_description" in df.columns:
        print(f"\n  Top NAICS descriptions:")
        for desc, cnt in df["naic_code_description"].value_counts().head(10).items():
            print(f"    {str(desc):50s} {cnt:>6,}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch SF registered businesses")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df_raw = fetch_sf_businesses(limit=args.limit)
    if df_raw is None:
        return

    df_clean = clean_sf_businesses(df_raw)

    df_clean.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
