"""
Fetch NYC DOHMH restaurant inspection results.
Source: NYC Open Data — Socrata API
Dataset: "DOHMH New York City Restaurant Inspection Results" (43nn-pn8j)
"""

import requests
import pandas as pd
import os
import time
import argparse

API_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
OUTPUT_DIR = "data/public_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nyc_dohmh_inspections.parquet")
PAGE_SIZE = 50000


def fetch_inspections(limit=None):
    """Paginate through restaurant inspection results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    offset = 0

    print("Fetching NYC DOHMH restaurant inspections...")
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

    df = pd.DataFrame(all_records)
    return df


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    for suffix in [" LLC", " INC", " CORP", " LTD", " CO", " DBA"]:
        name = name.replace(suffix, "")
    for ch in ["&", "'", ".", ",", "-", "#", "/"]:
        name = name.replace(ch, " ")
    return " ".join(name.split())


def normalize_street(building, street):
    parts = []
    if pd.notna(building):
        parts.append(str(building).strip())
    if pd.notna(street):
        s = str(street).upper().strip()
        replacements = {
            "AVENUE": "AVE", "STREET": "ST", "BOULEVARD": "BLVD",
            "DRIVE": "DR", "ROAD": "RD", "PLACE": "PL",
            "LANE": "LN", "COURT": "CT",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        for ch in [".", ",", "#", "-"]:
            s = s.replace(ch, " ")
        parts.append(s)
    return " ".join(" ".join(parts).split())


def clean_inspections(df):
    """Aggregate to one row per restaurant with latest inspection info."""

    # Normalize for matching
    df["name_normalized"] = df.get("dba", pd.Series(dtype=str)).apply(normalize_name)
    df["street_normalized"] = df.apply(
        lambda r: normalize_street(r.get("building"), r.get("street")), axis=1
    )
    df["zip5"] = df.get("zipcode", pd.Series(dtype=str)).astype(str).str[:5]

    # Parse dates
    df["inspection_date"] = pd.to_datetime(df.get("inspection_date"), errors="coerce")
    df["grade_date"] = pd.to_datetime(df.get("grade_date"), errors="coerce")

    # Parse score
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")

    # Aggregate: keep latest inspection per restaurant (by camis)
    camis_col = "camis" if "camis" in df.columns else None
    if camis_col:
        df_sorted = df.sort_values("inspection_date", ascending=False)
        latest = df_sorted.groupby(camis_col).first().reset_index()
    else:
        latest = df

    print(f"\n  Unique restaurants: {len(latest):,}")
    print(f"  With grades: {latest['grade'].notna().sum():,}" if "grade" in latest.columns else "")

    # Grade distribution
    if "grade" in latest.columns:
        print(f"\n  Grade distribution:")
        for grade, cnt in latest["grade"].value_counts().head(10).items():
            print(f"    {grade:5s} {cnt:>6,}")

    # Cuisine distribution
    if "cuisine_description" in latest.columns:
        print(f"\n  Top cuisines:")
        for cuisine, cnt in latest["cuisine_description"].value_counts().head(10).items():
            print(f"    {str(cuisine):30s} {cnt:>6,}")

    return latest


def main():
    parser = argparse.ArgumentParser(description="Fetch NYC DOHMH restaurant inspections")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df_raw = fetch_inspections(limit=args.limit)
    if df_raw is None:
        return

    df_clean = clean_inspections(df_raw)

    # Save
    df_clean.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
