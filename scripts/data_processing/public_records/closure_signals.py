"""
Closure Signal Enrichment — Advanced features from NYC & SF Open Data.

Fetches and matches 4 orthogonal closure signals:
  1. Vacancy Registries (NYC LL157 + SF Commercial Vacancy Tax)
  2. 311 Blight Complaints (graffiti, trash, blight - last 180 days)
  3. Permit Churn (change-of-use / tenant improvement - last 12 months)
  4. Stale Health Scores (DOHMH restaurant inspections > 18 months old)

Output one-hot features:
  - is_on_vacancy_registry
  - has_recent_blight_complaint
  - has_new_tenant_permit
  - stale_health_score

Uses rapidfuzz for address matching and sodapy for batched Socrata queries.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import time
import argparse
from datetime import datetime, timedelta
from sodapy import Socrata
from rapidfuzz import fuzz, process

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

OUTPUT_DIR = "data/public_records"
BATCH_SIZE = 5000  # Socrata query batch size
BLIGHT_WINDOW_DAYS = 180
PERMIT_WINDOW_MONTHS = 12
STALE_INSPECTION_MONTHS = 18
FUZZ_THRESHOLD = 82  # rapidfuzz score threshold (0-100)

# API endpoints (domain, resource_id)
APIS = {
    "nyc_vacancy":       ("data.cityofnewyork.us", "92iy-9c3n"),
    "sf_vacancy":        ("data.sfgov.org",        "qqbk-j3a9"),  # Taxable Commercial Spaces
    "nyc_311":           ("data.cityofnewyork.us", "erm2-nwe9"),
    "sf_311":            ("data.sfgov.org",         "vw6y-z8j6"),
    "nyc_dob_permits":   ("data.cityofnewyork.us", "rbx6-tga4"),
    "sf_permits":        ("data.sfgov.org",         "i98e-djp9"),
    "nyc_inspections":   ("data.cityofnewyork.us", "43nn-pn8j"),
}

# 311 complaint types indicating blight / vacancy
BLIGHT_TYPES_NYC = [
    "Graffiti", "Dirty Conditions", "Unsanitary Condition",
    "Derelict Vehicles", "Vacant Lot",
]
BLIGHT_TYPES_SF = [
    "Graffiti", "Street and Sidewalk Cleaning",
    "General Cleaning", "Blight",
]

# Permit keywords indicating tenant change
TENANT_PERMIT_KEYWORDS = [
    "TENANT IMPROVEMENT", "CHANGE OF USE", "CHANGE OF OCCUPANCY",
    "NEW TENANT", "TENANT IMPR", "ALTERATION",
]

# Restaurant categories (for stale health score)
RESTAURANT_CATEGORIES = {
    "restaurant", "pizza_restaurant", "fast_food_restaurant", "coffee_shop",
    "bakery", "deli", "food_stand", "food_truck", "bar", "nightclub",
    "brewery", "winery", "wine_bar", "pub", "beer_garden", "cafe",
    "ice_cream_shop", "juice_bar", "tea_room",
}


# ─────────────────────────────────────────────
# Address Normalization (rapidfuzz-ready)
# ─────────────────────────────────────────────

_STREET_ABBRV = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bBOULEVARD\b": "BLVD",
    r"\bDRIVE\b": "DR", r"\bROAD\b": "RD", r"\bPLACE\b": "PL",
    r"\bLANE\b": "LN", r"\bCOURT\b": "CT", r"\bTERRACE\b": "TER",
    r"\bPARKWAY\b": "PKWY", r"\bHIGHWAY\b": "HWY",
    r"\bCIRCLE\b": "CIR", r"\bSQUARE\b": "SQ",
    r"\b1ST\b": "1", r"\b2ND\b": "2", r"\b3RD\b": "3",
    r"\b(\d+)TH\b": r"\1", r"\b(\d+)RD\b": r"\1", r"\b(\d+)ND\b": r"\1",
    r"\bSUITE\b": "STE", r"\bAPARTMENT\b": "APT",
    r"\bFLOOR\b": "FL", r"\bUNIT\b": "UNIT",
}


def normalize_address(addr):
    """Normalize an address string for fuzzy matching."""
    if pd.isna(addr) or not addr:
        return ""
    addr = str(addr).upper().strip()
    # Remove suite/apt/unit numbers for matching
    addr = re.sub(r"\b(STE|SUITE|APT|APARTMENT|UNIT|FL|FLOOR|RM|ROOM)\s*#?\s*\w*", "", addr)
    # Apply abbreviations
    for pattern, replacement in _STREET_ABBRV.items():
        addr = re.sub(pattern, replacement, addr)
    # Remove punctuation
    addr = re.sub(r"[&'.,#\-/]", " ", addr)
    # Collapse whitespace
    return " ".join(addr.split())


def normalize_name(name):
    """Normalize a business name for matching."""
    if pd.isna(name) or not name:
        return ""
    name = str(name).upper().strip()
    for suffix in [" LLC", " INC", " CORP", " LTD", " CO", " DBA", " L.L.C."]:
        name = name.replace(suffix, "")
    name = re.sub(r"[&'.,#\-/]", " ", name)
    return " ".join(name.split())


def extract_zip5(z):
    """Extract 5-digit ZIP from various formats."""
    if pd.isna(z):
        return ""
    z = str(z).strip()
    m = re.match(r"(\d{5})", z)
    return m.group(1) if m else ""


# ─────────────────────────────────────────────
# POI Extraction
# ─────────────────────────────────────────────

def load_pois(dataset_path):
    """Load and parse POIs for matching."""
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} POIs")

    records = []
    for _, row in df.iterrows():
        try:
            names = json.loads(row["names"]) if isinstance(row["names"], str) else row["names"]
            name = names.get("primary", "") if isinstance(names, dict) else ""
        except:
            name = ""

        try:
            addrs = json.loads(row["addresses"]) if isinstance(row["addresses"], str) else row["addresses"]
            addr = addrs[0] if isinstance(addrs, list) and addrs else {}
        except:
            addr = {}

        try:
            cats = json.loads(row["categories"]) if isinstance(row["categories"], str) else row["categories"]
            cat = cats.get("primary", "") if isinstance(cats, dict) else ""
        except:
            cat = ""

        country = addr.get("country", "")
        if country != "US":
            continue

        records.append({
            "poi_id": row["id"],
            "poi_name": name,
            "poi_name_norm": normalize_name(name),
            "poi_addr": addr.get("freeform", ""),
            "poi_addr_norm": normalize_address(addr.get("freeform", "")),
            "poi_locality": addr.get("locality", ""),
            "poi_region": addr.get("region", ""),
            "poi_zip5": extract_zip5(addr.get("postcode", "")),
            "poi_category": cat,
            "poi_label": row.get("label", None),
        })

    pois = pd.DataFrame(records)
    print(f"  US POIs: {len(pois):,}")
    return pois


# ─────────────────────────────────────────────
# Socrata Client Helper
# ─────────────────────────────────────────────

def get_client(domain):
    """Create a Socrata client (no app token = anonymous, rate-limited)."""
    return Socrata(domain, None, timeout=60)


def fetch_batched(domain, resource_id, where_clause=None, select=None, limit=50000):
    """Fetch all records from a Socrata dataset in batches."""
    client = get_client(domain)
    all_records = []
    offset = 0

    while True:
        kwargs = {"limit": BATCH_SIZE, "offset": offset, "order": ":id"}
        if where_clause:
            kwargs["where"] = where_clause
        if select:
            kwargs["select"] = select

        try:
            batch = client.get(resource_id, **kwargs)
        except Exception as e:
            print(f"    ⚠️  API error at offset {offset}: {e}")
            time.sleep(2)
            try:
                batch = client.get(resource_id, **kwargs)
            except Exception as e2:
                print(f"    ❌ Retry failed: {e2}")
                break

        if not batch:
            break

        all_records.extend(batch)
        if len(all_records) >= limit:
            all_records = all_records[:limit]
            break
        if len(batch) < BATCH_SIZE:
            break

        offset += BATCH_SIZE
        time.sleep(0.3)  # rate limit

    client.close()
    return all_records


# ─────────────────────────────────────────────
# Fuzzy Address Matching
# ─────────────────────────────────────────────

def fuzzy_match_addresses(poi_addrs, registry_addrs, threshold=FUZZ_THRESHOLD):
    """
    Match POI addresses against registry addresses using rapidfuzz.
    Returns dict: poi_index → best matching registry_index (if above threshold).
    """
    # Build lookup by ZIP for efficiency
    registry_by_zip = {}
    for idx, (addr, zip5) in enumerate(registry_addrs):
        registry_by_zip.setdefault(zip5, []).append((idx, addr))

    matches = {}
    for poi_idx, (poi_addr, poi_zip) in enumerate(poi_addrs):
        if not poi_addr or not poi_zip:
            continue

        candidates = registry_by_zip.get(poi_zip, [])
        if not candidates:
            continue

        # Rapidfuzz extractOne (token_sort_ratio handles word reordering)
        candidate_strs = [c[1] for c in candidates]
        result = process.extractOne(
            poi_addr, candidate_strs,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )

        if result:
            matched_str, score, matched_idx = result
            matches[poi_idx] = {
                "registry_idx": candidates[matched_idx][0],
                "score": score,
                "matched_addr": matched_str,
            }

    return matches


# ─────────────────────────────────────────────
# Signal 1: Vacancy Registries
# ─────────────────────────────────────────────

def fetch_vacancy_signal(pois):
    """
    is_on_vacancy_registry:
      NYC: LL157 Storefront Registry — vacant_on_12_31 = 'YES'
      SF:  Commercial Vacancy Tax — vacancy_reported = true
    """
    print("\n" + "="*60)
    print("SIGNAL 1: Vacancy Registries")
    print("="*60)

    result = pd.Series(0, index=pois.index, name="is_on_vacancy_registry")

    # ── NYC Storefront Vacancy ──
    nyc_pois = pois[pois["poi_region"].isin(["NY", "NJ"])].copy()
    if len(nyc_pois) > 0:
        print(f"\n  Fetching NYC LL157 Storefront Vacancy...")
        domain, resource = APIS["nyc_vacancy"]
        records = fetch_batched(
            domain, resource,
            where_clause="vacant_on_12_31='YES'",
            select="property_street_address_or,property_number,property_street,zip_code,borough,primary_business_activity",
        )
        print(f"    Fetched {len(records):,} vacant storefronts")

        if records:
            vac_df = pd.DataFrame(records)
            vac_df["addr_norm"] = vac_df["property_street_address_or"].apply(normalize_address)
            vac_df["zip5"] = vac_df["zip_code"].apply(extract_zip5)

            # Fuzzy match
            poi_addrs = list(zip(nyc_pois["poi_addr_norm"], nyc_pois["poi_zip5"]))
            reg_addrs = list(zip(vac_df["addr_norm"], vac_df["zip5"]))
            matches = fuzzy_match_addresses(poi_addrs, reg_addrs)

            matched_pois = [nyc_pois.index[i] for i in matches.keys()]
            result.loc[matched_pois] = 1
            print(f"    Matched {len(matches):,} / {len(nyc_pois):,} NYC POIs to vacancy registry")

    # ── SF Commercial Vacancy Tax ──
    sf_pois = pois[pois["poi_region"] == "CA"].copy()
    if len(sf_pois) > 0:
        print(f"\n  Fetching SF Commercial Vacancy Tax...")
        domain, resource = APIS["sf_vacancy"]
        try:
            records = fetch_batched(
                domain, resource,
                where_clause="vacancy_reported='true' OR vacancy_reported='Yes'",
                limit=50000,
            )
            print(f"    Fetched {len(records):,} vacant commercial spaces")

            if records:
                vac_df = pd.DataFrame(records)
                # SF vacancy uses situs_address
                addr_col = next((c for c in vac_df.columns
                                 if "address" in c.lower() or "situs" in c.lower()), None)
                if addr_col:
                    vac_df["addr_norm"] = vac_df[addr_col].apply(normalize_address)
                    zip_col = next((c for c in vac_df.columns if "zip" in c.lower()), None)
                    vac_df["zip5"] = vac_df[zip_col].apply(extract_zip5) if zip_col else "94"

                    poi_addrs = list(zip(sf_pois["poi_addr_norm"], sf_pois["poi_zip5"]))
                    reg_addrs = list(zip(vac_df["addr_norm"], vac_df["zip5"]))
                    matches = fuzzy_match_addresses(poi_addrs, reg_addrs)

                    matched_pois = [sf_pois.index[i] for i in matches.keys()]
                    result.loc[matched_pois] = 1
                    print(f"    Matched {len(matches):,} / {len(sf_pois):,} SF POIs to vacancy registry")
        except Exception as e:
            print(f"    ⚠️  SF vacancy API error: {e}")
            print(f"    Continuing without SF vacancy data...")

    total = result.sum()
    print(f"\n  Total: {total} POIs on vacancy registries")
    return result


# ─────────────────────────────────────────────
# Signal 2: 311 Blight Complaints
# ─────────────────────────────────────────────

def fetch_blight_signal(pois):
    """
    has_recent_blight_complaint:
      2+ 311 complaints for graffiti/trash/blight at this address
      within the last 180 days.
    """
    print("\n" + "="*60)
    print("SIGNAL 2: 311 Blight Complaints (last 180 days)")
    print("="*60)

    result = pd.Series(0, index=pois.index, name="has_recent_blight_complaint")
    cutoff = (datetime.now() - timedelta(days=BLIGHT_WINDOW_DAYS)).strftime("%Y-%m-%dT00:00:00")

    # ── NYC 311 ──
    nyc_pois = pois[pois["poi_region"].isin(["NY", "NJ"])].copy()
    if len(nyc_pois) > 0:
        print(f"\n  Fetching NYC 311 blight complaints (since {cutoff[:10]})...")
        domain, resource = APIS["nyc_311"]
        type_filter = " OR ".join([f"complaint_type='{t}'" for t in BLIGHT_TYPES_NYC])
        where = f"created_date > '{cutoff}' AND ({type_filter})"

        records = fetch_batched(
            domain, resource,
            where_clause=where,
            select="incident_address,incident_zip,street_name,complaint_type,created_date",
            limit=100000,
        )
        print(f"    Fetched {len(records):,} blight complaints")

        if records:
            complaints = pd.DataFrame(records)
            complaints["addr_norm"] = complaints["incident_address"].apply(normalize_address)
            complaints["zip5"] = complaints.get("incident_zip", pd.Series(dtype=str)).apply(extract_zip5)

            # Count complaints per normalized address+zip
            complaint_counts = complaints.groupby(["addr_norm", "zip5"]).size().reset_index(name="count")
            blight_addrs = complaint_counts[complaint_counts["count"] >= 2]

            if len(blight_addrs) > 0:
                poi_addrs = list(zip(nyc_pois["poi_addr_norm"], nyc_pois["poi_zip5"]))
                reg_addrs = list(zip(blight_addrs["addr_norm"], blight_addrs["zip5"]))
                matches = fuzzy_match_addresses(poi_addrs, reg_addrs)

                matched_pois = [nyc_pois.index[i] for i in matches.keys()]
                result.loc[matched_pois] = 1
                print(f"    Matched {len(matches):,} / {len(nyc_pois):,} NYC POIs to blight complaints")

    # ── SF 311 ──
    sf_pois = pois[pois["poi_region"] == "CA"].copy()
    if len(sf_pois) > 0:
        print(f"\n  Fetching SF 311 blight complaints...")
        domain, resource = APIS["sf_311"]
        type_filter = " OR ".join([f"service_name='{t}'" for t in BLIGHT_TYPES_SF])
        where = f"requested_datetime > '{cutoff}' AND ({type_filter})"

        records = fetch_batched(
            domain, resource,
            where_clause=where,
            select="address,street,service_name,requested_datetime",
            limit=100000,
        )
        print(f"    Fetched {len(records):,} blight complaints")

        if records:
            complaints = pd.DataFrame(records)
            complaints["addr_norm"] = complaints["address"].apply(normalize_address)
            # SF 311 doesn't always have ZIP — extract from full address
            complaints["zip5"] = complaints["address"].apply(
                lambda x: extract_zip5(re.search(r"(\d{5})", str(x)).group(1))
                if pd.notna(x) and re.search(r"(\d{5})", str(x)) else ""
            )

            complaint_counts = complaints.groupby("addr_norm").size().reset_index(name="count")
            blight_addrs = complaint_counts[complaint_counts["count"] >= 2]

            if len(blight_addrs) > 0:
                # For SF 311 without ZIP, match on address only
                poi_addrs = list(zip(sf_pois["poi_addr_norm"], sf_pois["poi_zip5"]))
                reg_addrs = list(zip(blight_addrs["addr_norm"], blight_addrs.get("zip5", pd.Series("", index=blight_addrs.index))))
                # Use lower threshold for address-only matching
                matches = fuzzy_match_addresses(poi_addrs, reg_addrs, threshold=75)

                matched_pois = [sf_pois.index[i] for i in matches.keys()]
                result.loc[matched_pois] = 1
                print(f"    Matched {len(matches):,} / {len(sf_pois):,} SF POIs to blight complaints")

    total = result.sum()
    print(f"\n  Total: {total} POIs with recent blight complaints")
    return result


# ─────────────────────────────────────────────
# Signal 3: Permit Churn (Tenant Change)
# ─────────────────────────────────────────────

def fetch_permit_signal(pois):
    """
    has_new_tenant_permit:
      A 'Change of Use' or 'Tenant Improvement' permit filed
      in the last 12 months at this address — but for a DIFFERENT
      business name than the POI.
    """
    print("\n" + "="*60)
    print("SIGNAL 3: Permit Churn (last 12 months)")
    print("="*60)

    result = pd.Series(0, index=pois.index, name="has_new_tenant_permit")
    cutoff = (datetime.now() - timedelta(days=PERMIT_WINDOW_MONTHS * 30)).strftime("%Y-%m-%dT00:00:00")

    # ── NYC DOB Permits ──
    nyc_pois = pois[pois["poi_region"].isin(["NY", "NJ"])].copy()
    if len(nyc_pois) > 0:
        print(f"\n  Fetching NYC DOB permits (since {cutoff[:10]})...")
        domain, resource = APIS["nyc_dob_permits"]
        # Filter for tenant-related work
        keyword_filter = " OR ".join([
            f"upper(job_description) like '%{kw}%'" for kw in TENANT_PERMIT_KEYWORDS
        ])
        where = f"({keyword_filter})"

        records = fetch_batched(
            domain, resource,
            where_clause=where,
            select="house_no,street_name,borough,job_description,owner_business_name,applicant_business_name,permit_status",
            limit=50000,
        )
        print(f"    Fetched {len(records):,} tenant-related permits")

        if records:
            permits = pd.DataFrame(records)
            permits["addr_norm"] = (
                permits.get("house_no", pd.Series("")) .astype(str) + " " +
                permits.get("street_name", pd.Series("")).astype(str)
            ).apply(normalize_address)

            # Match addresses
            poi_addrs = list(zip(nyc_pois["poi_addr_norm"], nyc_pois["poi_zip5"]))
            # NYC DOB permits don't have ZIP — match on address only within NYC
            permit_addrs = list(zip(permits["addr_norm"], pd.Series("", index=permits.index)))

            # Direct address matching (without ZIP constraint)
            for poi_idx, (poi_addr, poi_zip) in enumerate(poi_addrs):
                if not poi_addr:
                    continue
                for perm_idx, (perm_addr, _) in enumerate(permit_addrs):
                    if not perm_addr:
                        continue
                    score = fuzz.token_sort_ratio(poi_addr, perm_addr)
                    if score >= FUZZ_THRESHOLD:
                        # Check if business name is DIFFERENT
                        poi_name = nyc_pois.iloc[poi_idx]["poi_name_norm"]
                        perm_name = normalize_name(
                            permits.iloc[perm_idx].get("owner_business_name", "") or
                            permits.iloc[perm_idx].get("applicant_business_name", "")
                        )
                        if perm_name and fuzz.ratio(poi_name, perm_name) < 70:
                            result.iloc[nyc_pois.index[poi_idx]] = 1
                        break  # one match is enough

    # ── SF Building Permits ──
    sf_pois = pois[pois["poi_region"] == "CA"].copy()
    if len(sf_pois) > 0:
        print(f"\n  Fetching SF building permits...")
        domain, resource = APIS["sf_permits"]
        keyword_filter = " OR ".join([
            f"upper(description) like '%{kw}%'" for kw in TENANT_PERMIT_KEYWORDS
        ])
        where = f"filed_date > '{cutoff}' AND ({keyword_filter})"

        records = fetch_batched(
            domain, resource,
            where_clause=where,
            select="street_number,street_name,street_suffix,zipcode,description,permit_type_definition",
            limit=50000,
        )
        print(f"    Fetched {len(records):,} tenant-related permits")

        if records:
            permits = pd.DataFrame(records)
            permits["addr_norm"] = (
                permits.get("street_number", pd.Series("")).astype(str) + " " +
                permits.get("street_name", pd.Series("")).astype(str) + " " +
                permits.get("street_suffix", pd.Series("")).astype(str)
            ).apply(normalize_address)
            permits["zip5"] = permits.get("zipcode", pd.Series("")).apply(extract_zip5)

            poi_addrs = list(zip(sf_pois["poi_addr_norm"], sf_pois["poi_zip5"]))
            perm_addrs = list(zip(permits["addr_norm"], permits["zip5"]))
            matches = fuzzy_match_addresses(poi_addrs, perm_addrs, threshold=80)

            matched_pois = [sf_pois.index[i] for i in matches.keys()]
            result.loc[matched_pois] = 1
            print(f"    Matched {len(matches):,} / {len(sf_pois):,} SF POIs to tenant permits")

    total = result.sum()
    print(f"\n  Total: {total} POIs with new tenant permits")
    return result


# ─────────────────────────────────────────────
# Signal 4: Stale Health Score
# ─────────────────────────────────────────────

def compute_stale_health_signal(pois):
    """
    stale_health_score:
      For restaurant-category POIs, if the last DOHMH inspection
      was > 18 months ago → likely no longer operating.
      Uses the previously-fetched DOHMH data.
    """
    print("\n" + "="*60)
    print("SIGNAL 4: Stale Health Score (DOHMH restaurants)")
    print("="*60)

    result = pd.Series(0, index=pois.index, name="stale_health_score")

    # Only applies to NYC restaurant-category POIs
    nyc_restaurants = pois[
        (pois["poi_region"].isin(["NY", "NJ"])) &
        (pois["poi_category"].isin(RESTAURANT_CATEGORIES))
    ].copy()
    print(f"\n  NYC restaurant POIs: {len(nyc_restaurants):,}")

    dohmh_path = os.path.join(OUTPUT_DIR, "nyc_dohmh_inspections.parquet")
    if not os.path.exists(dohmh_path):
        print("  ⚠️  DOHMH data not found — fetching fresh...")
        domain, resource = APIS["nyc_inspections"]
        records = fetch_batched(
            domain, resource,
            select="dba,building,street,zipcode,inspection_date,grade,score",
            limit=200000,
        )
        if records:
            dohmh = pd.DataFrame(records)
            dohmh.to_parquet(dohmh_path, index=False)
            print(f"    Fetched {len(dohmh):,} inspection records")
    else:
        dohmh = pd.read_parquet(dohmh_path)
        print(f"  Loaded {len(dohmh):,} inspection records from cache")

    if len(dohmh) == 0 or len(nyc_restaurants) == 0:
        return result

    # Parse and normalize
    dohmh["inspection_date"] = pd.to_datetime(dohmh.get("inspection_date"), errors="coerce")
    dohmh["addr_norm"] = dohmh.apply(
        lambda r: normalize_address(
            f"{r.get('building', '')} {r.get('street', '')}"
        ), axis=1
    )
    dohmh["zip5"] = dohmh.get("zipcode", pd.Series("")).apply(extract_zip5)

    # Get latest inspection per address+zip
    latest = dohmh.sort_values("inspection_date", ascending=False).groupby(
        ["addr_norm", "zip5"]
    ).first().reset_index()

    stale_cutoff = datetime.now() - timedelta(days=STALE_INSPECTION_MONTHS * 30)
    stale_addrs = latest[latest["inspection_date"] < stale_cutoff]
    print(f"  Inspections older than {STALE_INSPECTION_MONTHS} months: {len(stale_addrs):,}")

    if len(stale_addrs) > 0:
        # Match restaurant POIs to stale inspections
        poi_addrs = list(zip(nyc_restaurants["poi_addr_norm"], nyc_restaurants["poi_zip5"]))
        reg_addrs = list(zip(stale_addrs["addr_norm"], stale_addrs["zip5"]))
        matches = fuzzy_match_addresses(poi_addrs, reg_addrs)

        matched_pois = [nyc_restaurants.index[i] for i in matches.keys()]
        result.loc[matched_pois] = 1
        print(f"  Matched {len(matches):,} / {len(nyc_restaurants):,} restaurants to stale inspections")

    # Also flag restaurant POIs with NO inspection at all
    all_inspected_addrs = set(zip(latest["addr_norm"], latest["zip5"]))
    no_inspection = []
    for poi_idx, (poi_addr, poi_zip) in enumerate(
        zip(nyc_restaurants["poi_addr_norm"], nyc_restaurants["poi_zip5"])
    ):
        if not poi_addr or not poi_zip:
            continue
        # Check if any existing inspection is close
        found = False
        for (reg_addr, reg_zip) in all_inspected_addrs:
            if reg_zip == poi_zip and fuzz.token_sort_ratio(poi_addr, reg_addr) >= FUZZ_THRESHOLD:
                found = True
                break
        if not found:
            no_inspection.append(nyc_restaurants.index[poi_idx])

    result.loc[no_inspection] = 1
    print(f"  NYC restaurants with no inspection found: {len(no_inspection):,}")

    total = result.sum()
    print(f"\n  Total: {total} POIs with stale health scores")
    return result


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

def run_enrichment(dataset_path, output_path=None):
    """Run all 4 closure signals and combine into a feature DataFrame."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "closure_signals.parquet")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pois = load_pois(dataset_path)

    # Compute all 4 signals
    vacancy = fetch_vacancy_signal(pois)
    blight = fetch_blight_signal(pois)
    permits = fetch_permit_signal(pois)
    stale_health = compute_stale_health_signal(pois)

    # Combine
    features = pd.DataFrame({
        "poi_id": pois["poi_id"],
        "poi_label": pois["poi_label"],
        "poi_category": pois["poi_category"],
        "is_on_vacancy_registry": vacancy.values,
        "has_recent_blight_complaint": blight.values,
        "has_new_tenant_permit": permits.values,
        "stale_health_score": stale_health.values,
    })

    # Derived: composite closure risk
    features["closure_signal_count"] = (
        features["is_on_vacancy_registry"] +
        features["has_recent_blight_complaint"] +
        features["has_new_tenant_permit"] +
        features["stale_health_score"]
    )

    # ── Correlation Analysis ──
    print(f"\n{'='*60}")
    print(f"CLOSURE SIGNAL ANALYSIS")
    print(f"{'='*60}")

    signal_cols = [
        "is_on_vacancy_registry", "has_recent_blight_complaint",
        "has_new_tenant_permit", "stale_health_score", "closure_signal_count",
    ]

    print(f"  Total US POIs: {len(features):,}")
    print(f"\n  Feature correlations with label (Open=1, Closed=0):")
    for col in signal_cols:
        positives = features[col].sum()
        if positives > 0 and features["poi_label"].notna().sum() > 0:
            corr = features[col].corr(features["poi_label"])
            print(f"    {col:35s} r={corr:+.4f}   count={int(positives):,}")
        else:
            print(f"    {col:35s} count=0")

    # Cross-tab: signal vs label
    print(f"\n  Label distribution by signal:")
    for col in signal_cols[:4]:
        if features[col].sum() > 0:
            ct = pd.crosstab(features[col], features["poi_label"], margins=True)
            print(f"\n    {col}:")
            for idx_val in [0, 1]:
                if idx_val in ct.index:
                    open_n = ct.get(1.0, pd.Series(0, index=ct.index)).get(idx_val, 0)
                    closed_n = ct.get(0.0, pd.Series(0, index=ct.index)).get(idx_val, 0)
                    total_n = open_n + closed_n
                    r = closed_n / total_n * 100 if total_n > 0 else 0
                    print(f"      {col}={idx_val}: Open={open_n}, Closed={closed_n}, ClosedRate={r:.1f}%")

    # Save
    features.to_parquet(output_path, index=False)
    print(f"\n  ✅ Saved to {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")

    return features


def main():
    parser = argparse.ArgumentParser(description="Closure Signal Enrichment")
    parser.add_argument("-i", "--input", type=str,
                        default="data/combined_truth_dataset_all.parquet")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    run_enrichment(args.input, args.output)


if __name__ == "__main__":
    main()
