"""
Fetch NY State Liquor Authority (SLA) license data — active + inactive.
Source: NY Open Data — Socrata API
Active: https://data.ny.gov/resource/hrvs-fxs2.json
Inactive: https://data.ny.gov/resource/jg5h-i3cs.json
"""

import requests
import pandas as pd
import os
import time
import argparse

ACTIVE_URL = "https://data.ny.gov/resource/9s3h-dpkz.json"
INACTIVE_URL = "https://data.ny.gov/resource/9s3h-dpkz.json"  # Same dataset, filter by status
OUTPUT_DIR = "data/public_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nyc_sla_licenses.parquet")
PAGE_SIZE = 10000  # NY Open Data may throttle at 50k

# NYC-area counties to filter
NYC_COUNTIES = [
    "New York", "Kings", "Queens", "Bronx", "Richmond",  # 5 boroughs
    "Nassau", "Suffolk", "Westchester",  # nearby
]


def fetch_sla_dataset(url, label, nyc_only=True, limit=None):
    """Paginate through an SLA dataset."""
    all_records = []
    offset = 0

    print(f"\nFetching SLA {label} licenses...")
    print(f"  API: {url}")

    while True:
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": ":id",
        }

        # Filter to NYC counties if requested
        if nyc_only:
            county_filter = " OR ".join([f"premisescounty='{c}'" for c in NYC_COUNTIES])
            params["$where"] = county_filter

        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ❌ Request failed at offset {offset}: {e}")
            break

        batch = resp.json()
        if not batch:
            break

        all_records.extend(batch)
        print(f"  Fetched {len(all_records):>8,} {label} records")

        if limit and len(all_records) >= limit:
            break

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)

    return all_records


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


def main():
    parser = argparse.ArgumentParser(description="Fetch NY SLA liquor licenses")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-ny", action="store_true",
                        help="Fetch all NY state, not just NYC area")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nyc_only = not args.all_ny

    active_records = fetch_sla_dataset(ACTIVE_URL, "active", nyc_only, args.limit)
    inactive_records = fetch_sla_dataset(INACTIVE_URL, "inactive", nyc_only, args.limit)

    # Combine
    for r in active_records:
        r["sla_status"] = "active"
    for r in inactive_records:
        r["sla_status"] = "inactive"

    all_records = active_records + inactive_records
    print(f"\n  Total SLA records: {len(all_records):,}")
    print(f"    Active: {len(active_records):,}")
    print(f"    Inactive: {len(inactive_records):,}")

    if not all_records:
        print("  ❌ No records fetched!")
        return

    df = pd.DataFrame(all_records)

    # Normalize for matching — use actual API column names
    name_col = "legalname" if "legalname" in df.columns else "premises_name"
    df["name_normalized"] = df[name_col].apply(normalize_name)

    addr_col = "actualaddressofpremises" if "actualaddressofpremises" in df.columns else "premises_address"
    df["street_normalized"] = df.get(addr_col, pd.Series(dtype=str)).apply(normalize_street)

    zip_col = [c for c in df.columns if "zip" in c.lower()]
    if zip_col:
        df["zip5"] = df[zip_col[0]].astype(str).str[:5]
    else:
        df["zip5"] = ""

    # Parse dates
    for date_col in ["effectivedate", "expirationdate", "originalissuedate", "lastissuedate"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # License type breakdown
    type_col = "description" if "description" in df.columns else "license_type_name"
    if type_col in df.columns:
        print(f"\n  Top license types:")
        for lt, cnt in df[type_col].value_counts().head(10).items():
            print(f"    {str(lt):40s} {cnt:>6,}")

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
