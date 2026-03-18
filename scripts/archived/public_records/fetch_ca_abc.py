"""
Fetch California ABC (Alcoholic Beverage Control) license data.
Source: CA ABC Licensing Reports — CSV bulk download
URL: https://www.abc.ca.gov/licensing/licensing-reports/
"""

import requests
import pandas as pd
import os
import io
import argparse

# CA ABC provides daily data exports as tab-delimited files
# These URLs may change — check https://www.abc.ca.gov/licensing/licensing-reports/
ABC_ACTIVE_URL = "https://www.abc.ca.gov/wp-content/uploads/Licensing_Reports/active_licenses.csv"
OUTPUT_DIR = "data/public_records"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ca_abc_licenses.parquet")

# SF-area ZIP code prefixes (941xx)
SF_ZIP_PREFIXES = ["941"]
# Include broader Bay Area
BAY_AREA_ZIP_PREFIXES = ["940", "941", "943", "944", "945", "946", "947", "948", "949", "950", "951"]


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


def fetch_abc_data(sf_only=True):
    """Download and parse CA ABC active license data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching California ABC active licenses...")
    print(f"  URL: {ABC_ACTIVE_URL}")

    try:
        resp = requests.get(ABC_ACTIVE_URL, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        print("  Trying alternative approach — direct CSV download...")

        # Try alternative: the raw data link
        alt_url = "https://www.abc.ca.gov/wp-content/uploads/Licensing_Reports/LICENSEE_DATA.csv"
        try:
            resp = requests.get(alt_url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e2:
            print(f"  ❌ Alternative also failed: {e2}")
            print("  You may need to manually download from https://www.abc.ca.gov/licensing/licensing-reports/")
            return None

    # Try different delimiters (ABC uses tab-delimited)
    try:
        df = pd.read_csv(io.StringIO(resp.text), sep="\t", dtype=str, on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(resp.text), sep=",", dtype=str, on_bad_lines="skip")
        except Exception as e:
            print(f"  ❌ Parse failed: {e}")
            # Save raw for debugging
            raw_path = os.path.join(OUTPUT_DIR, "ca_abc_raw.txt")
            with open(raw_path, "w") as f:
                f.write(resp.text[:10000])
            print(f"  Saved first 10K chars to {raw_path} for debugging")
            return None

    print(f"  Raw records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Filter to SF area if requested
    if sf_only:
        zip_col = [c for c in df.columns if "zip" in c.lower() or "postal" in c.lower()]
        if zip_col:
            zip_col = zip_col[0]
            sf_mask = df[zip_col].astype(str).str[:3].isin(SF_ZIP_PREFIXES)
            bay_mask = df[zip_col].astype(str).str[:3].isin(BAY_AREA_ZIP_PREFIXES)
            print(f"  SF area ({SF_ZIP_PREFIXES}): {sf_mask.sum():,}")
            print(f"  Bay Area: {bay_mask.sum():,}")
            df = df[bay_mask].copy()
        else:
            # Try filtering by city
            city_col = [c for c in df.columns if "city" in c.lower()]
            if city_col:
                sf_cities = ["SAN FRANCISCO", "DALY CITY", "SOUTH SAN FRANCISCO", "BRISBANE"]
                df = df[df[city_col[0]].str.upper().isin(sf_cities)].copy()

    return df


def clean_abc(df):
    """Normalize ABC data for matching."""

    # Identify columns by inspection
    name_cols = [c for c in df.columns if any(k in c.lower() for k in ["licensee", "business", "dba", "name"])]
    addr_cols = [c for c in df.columns if any(k in c.lower() for k in ["addr", "premises", "street"])]
    zip_cols = [c for c in df.columns if "zip" in c.lower() or "postal" in c.lower()]
    status_cols = [c for c in df.columns if "status" in c.lower()]

    print(f"\n  Detected columns:")
    print(f"    Name: {name_cols}")
    print(f"    Address: {addr_cols}")
    print(f"    ZIP: {zip_cols}")
    print(f"    Status: {status_cols}")

    if name_cols:
        df["name_normalized"] = df[name_cols[0]].apply(normalize_name)
    else:
        df["name_normalized"] = ""

    if addr_cols:
        df["street_normalized"] = df[addr_cols[0]].apply(normalize_street)
    else:
        df["street_normalized"] = ""

    if zip_cols:
        df["zip5"] = df[zip_cols[0]].astype(str).str[:5]
    else:
        df["zip5"] = ""

    if status_cols:
        print(f"\n  Status distribution:")
        for status, cnt in df[status_cols[0]].value_counts().head(10).items():
            print(f"    {str(status):20s} {cnt:>6,}")

    print(f"\n  Cleaned records: {len(df):,}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch CA ABC liquor licenses")
    parser.add_argument("--all-ca", action="store_true",
                        help="Fetch all California, not just SF area")
    args = parser.parse_args()

    df_raw = fetch_abc_data(sf_only=not args.all_ca)
    if df_raw is None:
        return

    df_clean = clean_abc(df_raw)

    df_clean.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  ✅ Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
