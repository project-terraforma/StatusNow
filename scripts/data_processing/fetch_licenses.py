"""
fetch_licenses.py -- Download public business license data from SF and NYC open data portals.
Uses the Socrata Open Data API (SODA). No authentication required.
Saves to data/licenses/sf_licenses.parquet and data/licenses/nyc_licenses.parquet
"""

import os
import sys
import io
import time
import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT_DIR = "data/licenses"
os.makedirs(OUT_DIR, exist_ok=True)

PAGE_SIZE = 50000  # SODA max per request


def fetch_soda(base_url, dataset_id, select=None, where=None, order=None):
    """Paginate through a SODA API endpoint and return all records as a DataFrame."""
    endpoint = f"{base_url}/resource/{dataset_id}.json"
    all_rows = []
    offset = 0

    params_base = {"$limit": PAGE_SIZE}
    if select:
        params_base["$select"] = select
    if where:
        params_base["$where"] = where
    if order:
        params_base["$order"] = order

    while True:
        params = {**params_base, "$offset": offset}
        print(f"    Fetching offset {offset:,} ...", end=" ", flush=True)
        resp = requests.get(endpoint, params=params, timeout=120)
        resp.raise_for_status()
        rows = resp.json()
        print(f"got {len(rows):,} rows")

        if not rows:
            break
        all_rows.extend(rows)
        offset += PAGE_SIZE

        if len(rows) < PAGE_SIZE:
            break

        time.sleep(0.5)  # be polite

    return pd.DataFrame(all_rows)


# ============================================================================
# SF: Registered Businesses (g8m3-pdis)
# ============================================================================
def fetch_sf():
    print("\n[SF] Fetching registered businesses from DataSF ...")
    select = ",".join([
        "dba_name",
        "ownership_name",
        "full_business_address",
        "city",
        "state",
        "business_zip",
        "naic_code",
        "naic_code_description",
        "dba_start_date",
        "dba_end_date",
        "location_start_date",
        "location_end_date",
        "parking_tax",
        "transient_occupancy_tax",
        "neighborhoods_analysis_boundaries",
        "uniqueid",
    ])

    # Only SF city records with a DBA name
    where = "city='San Francisco' AND dba_name IS NOT NULL"
    order = "dba_name"

    df = fetch_soda("https://data.sfgov.org", "g8m3-pdis",
                    select=select, where=where, order=order)

    print(f"  Total SF records: {len(df):,}")

    # Normalize column names to unified schema
    df = df.rename(columns={
        "dba_name": "business_name",
        "full_business_address": "address",
        "business_zip": "zip",
        "naic_code_description": "license_category",
        "dba_start_date": "license_start_date",
        "dba_end_date": "license_end_date",
        "neighborhoods_analysis_boundaries": "neighborhood",
        "uniqueid": "license_id",
    })
    df["city_source"] = "sf"

    out = os.path.join(OUT_DIR, "sf_licenses.parquet")
    df.to_parquet(out, index=False)
    print(f"  Saved -> {out}")
    return df


# ============================================================================
# NYC: Issued Licenses (w7w3-xahh)
# ============================================================================
def fetch_nyc():
    print("\n[NYC] Fetching issued licenses from NYC OpenData ...")
    select = ",".join([
        "license_nbr",
        "business_name",
        "business_category",
        "license_type",
        "license_status",
        "license_creation_date",
        "lic_expir_dd",
        "contact_phone",
        "address_building",
        "address_street_name",
        "address_city",
        "address_state",
        "address_zip",
        "address_borough",
        "latitude",
        "longitude",
    ])

    where = "business_name IS NOT NULL"
    order = "business_name"

    df = fetch_soda("https://data.cityofnewyork.us", "w7w3-xahh",
                    select=select, where=where, order=order)

    print(f"  Total NYC records: {len(df):,}")

    # Build combined address field
    df["address"] = (
        df["address_building"].fillna("") + " " +
        df["address_street_name"].fillna("")
    ).str.strip()

    df = df.rename(columns={
        "license_nbr": "license_id",
        "business_category": "license_category",
        "license_creation_date": "license_start_date",
        "lic_expir_dd": "license_end_date",
        "address_zip": "zip",
    })
    df["city_source"] = "nyc"

    out = os.path.join(OUT_DIR, "nyc_licenses.parquet")
    df.to_parquet(out, index=False)
    print(f"  Saved -> {out}")
    return df


# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  License Data Fetcher")
    print("=" * 60)

    sf_df = fetch_sf()
    nyc_df = fetch_nyc()

    print(f"\n{'=' * 60}")
    print(f"  Done! SF: {len(sf_df):,} rows | NYC: {len(nyc_df):,} rows")
    print(f"{'=' * 60}")
