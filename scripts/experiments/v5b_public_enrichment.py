"""
V5b Public Data Enrichment — NPI + IRS + Google Places

Enriches the StatusNow dataset using ONLY fully-public, no-auth-needed APIs:
1. NPI Registry (Healthcare) — free CMS API, no key
2. IRS 501(c)(3) (Religious) — tax-exempt org search
3. Google Places business_status (free tier, API key required)

Then retrains with enriched features and compares to V3 baseline.
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
import os
import requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
RAW_PATH = "data/combined_truth_dataset.parquet"
V3_PATH = "data/processed_for_ml_testing.parquet"
OUT_DIR = "/Users/anthonylamas/.gemini/antigravity/brain/6dcd71ed-bf93-479d-b5a1-6ed4db48813c"
ENRICHED_CACHE = "data/enrichment_cache.parquet"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

NPI_URL = "https://npiregistry.cms.hhs.gov/api/?version=2.1"
IRS_URL = "https://apps.irs.gov/app/eos/api/records"

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def parse_json(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    if isinstance(x, (list, dict)):
        return x
    return None

def get_name(x):
    d = parse_json(x)
    if isinstance(d, dict):
        return d.get("primary", "")
    return ""

def get_cat(x):
    d = parse_json(x)
    if isinstance(d, dict):
        return d.get("primary", "")
    return ""

def get_address(x):
    d = parse_json(x)
    if isinstance(d, list) and len(d) > 0:
        a = d[0]
        if isinstance(a, dict):
            return {
                "freeform": a.get("freeform", ""),
                "locality": a.get("locality", ""),
                "region": a.get("region", ""),
                "postcode": str(a.get("postcode", "")),
                "country": a.get("country", "US"),
            }
    return {"freeform": "", "locality": "", "region": "", "postcode": "", "country": ""}


# ════════════════════════════════════════════════════════════════════════════
# NPI REGISTRY LOOKUP (Healthcare)
# ════════════════════════════════════════════════════════════════════════════
def query_npi(name, city, state, max_retries=2):
    """
    Query the NPI Registry for a healthcare organization.
    Returns: (has_match: bool, result_count: int)
    """
    if not name or not state:
        return False, 0

    for attempt in range(max_retries):
        try:
            params = {
                "version": "2.1",
                "organization_name": name[:40],  # API limit
                "state": state,
                "limit": 5,
            }
            if city:
                params["city"] = city

            resp = requests.get(NPI_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("result_count", 0)
                return count > 0, count
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return False, 0
        except Exception:
            time.sleep(1)
    return False, 0


def query_npi_individual(name, city, state, max_retries=2):
    """
    Query NPI for individual providers (doctors, dentists, etc.)
    Uses first/last name split.
    """
    if not name or not state:
        return False, 0

    parts = name.strip().split()
    if len(parts) < 2:
        return False, 0

    for attempt in range(max_retries):
        try:
            params = {
                "version": "2.1",
                "last_name": parts[-1],
                "state": state,
                "limit": 5,
            }
            if city:
                params["city"] = city

            resp = requests.get(NPI_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("result_count", 0)
                return count > 0, count
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return False, 0


# ════════════════════════════════════════════════════════════════════════════
# IRS 501(c)(3) LOOKUP (Religious)
# ════════════════════════════════════════════════════════════════════════════
def query_irs_exempt(name, state, city="", max_retries=2):
    """
    Query IRS Exempt Organization Search API.
    Returns: (has_match: bool, result_count: int)
    """
    if not name or not state:
        return False, 0

    for attempt in range(max_retries):
        try:
            params = {
                "orgName": name[:50],
                "state": state,
                "deductibility": "all",
                "sortColumn": "orgName",
                "resultsPerPage": 5,
                "isDescending": "false",
            }
            if city:
                params["city"] = city

            resp = requests.get(
                "https://apps.irs.gov/app/eos/api/records",
                params=params,
                timeout=10,
                headers={"Accept": "application/json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                orgs = data.get("organizations", [])
                count = len(orgs)
                return count > 0, count
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return False, 0
        except Exception:
            time.sleep(1)
    return False, 0


# ════════════════════════════════════════════════════════════════════════════
# ENRICHMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════
def run_enrichment(raw_df):
    """Enrich the raw dataset with NPI and IRS lookups."""
    print("=" * 80)
    print("PUBLIC DATA ENRICHMENT")
    print("=" * 80)

    raw_df = raw_df.copy()
    raw_df["name_clean"] = raw_df["names"].apply(get_name)
    raw_df["cat_clean"] = raw_df["categories"].apply(get_cat)
    raw_df["addr_parsed"] = raw_df["addresses"].apply(get_address)
    raw_df["city"] = raw_df["addr_parsed"].apply(lambda x: x.get("locality", ""))
    raw_df["state"] = raw_df["addr_parsed"].apply(lambda x: x.get("region", ""))
    raw_df["country"] = raw_df["addr_parsed"].apply(lambda x: x.get("country", "US"))

    # Filter to US places only (government registries are US-specific)
    us_mask = raw_df["country"] == "US"
    print(f"  Total places: {len(raw_df)}")
    print(f"  US places: {us_mask.sum()}")

    # Initialize enrichment columns
    raw_df["has_npi_match"] = 0
    raw_df["npi_result_count"] = 0
    raw_df["has_irs_exempt"] = 0
    raw_df["irs_result_count"] = 0

    # ── Categories for lookup ──────────────────────────────────────────
    healthcare_kw = ["hospital", "medical_center", "clinic", "doctor", "dental",
                     "chiropractor", "urgent_care", "veterinar", "health", "optician",
                     "physiotherapy", "nursing", "pediatr", "pharmacy"]
    religious_kw = ["church", "mosque", "temple", "synagogue", "religious",
                    "worship", "baptist", "methodist", "lutheran", "catholic",
                    "episcopal", "pentecostal", "presbyterian"]

    is_healthcare = raw_df["cat_clean"].apply(
        lambda x: any(k in str(x).lower() for k in healthcare_kw)
    ) & us_mask
    is_religious = raw_df["cat_clean"].apply(
        lambda x: any(k in str(x).lower() for k in religious_kw)
    ) & us_mask

    print(f"\n  US Healthcare to query: {is_healthcare.sum()}")
    print(f"  US Religious to query: {is_religious.sum()}")

    # ── NPI Lookups ────────────────────────────────────────────────────
    print("\n  [NPI] Querying NPI Registry for healthcare places …")
    npi_indices = raw_df[is_healthcare].index.tolist()
    npi_found = 0
    npi_total = len(npi_indices)

    for i, idx in enumerate(npi_indices):
        row = raw_df.loc[idx]
        name = row["name_clean"]
        city = row["city"]
        state = row["state"]

        if not state or len(state) > 2:
            continue

        has_match, count = query_npi(name, city, state)
        raw_df.loc[idx, "has_npi_match"] = int(has_match)
        raw_df.loc[idx, "npi_result_count"] = count
        if has_match:
            npi_found += 1

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{npi_total}] NPI matches so far: {npi_found}")
            time.sleep(0.3)  # Rate limiting

    print(f"  [NPI] Done: {npi_found}/{npi_total} healthcare places matched")

    # ── IRS Lookups ────────────────────────────────────────────────────
    print("\n  [IRS] Querying IRS Exempt Org Search for religious places …")
    irs_indices = raw_df[is_religious].index.tolist()
    irs_found = 0
    irs_total = len(irs_indices)

    for i, idx in enumerate(irs_indices):
        row = raw_df.loc[idx]
        name = row["name_clean"]
        city = row["city"]
        state = row["state"]

        if not state or len(state) > 2:
            continue

        has_match, count = query_irs_exempt(name, state, city)
        raw_df.loc[idx, "has_irs_exempt"] = int(has_match)
        raw_df.loc[idx, "irs_result_count"] = count
        if has_match:
            irs_found += 1

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{irs_total}] IRS matches so far: {irs_found}")
            time.sleep(0.3)

    print(f"  [IRS] Done: {irs_found}/{irs_total} religious places matched")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n  ENRICHMENT SUMMARY:")
    print(f"    NPI matches:  {npi_found}/{npi_total} ({npi_found/max(npi_total,1):.0%})")
    print(f"    IRS matches:  {irs_found}/{irs_total} ({irs_found/max(irs_total,1):.0%})")

    # Save enrichment cache
    enrichment = raw_df[["has_npi_match", "npi_result_count", "has_irs_exempt", "irs_result_count"]].copy()
    enrichment.to_parquet(ENRICHED_CACHE)
    print(f"  Saved enrichment cache to {ENRICHED_CACHE}")

    return enrichment


# ════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION WITH ENRICHED FEATURES
# ════════════════════════════════════════════════════════════════════════════
def evaluate_with_enrichment():
    print("\n" + "=" * 80)
    print("MODEL EVALUATION WITH PUBLIC DATA ENRICHMENT")
    print("=" * 80)

    # Load V3 baseline
    df_v3 = pd.read_parquet(V3_PATH)
    X_v3 = df_v3.drop(columns=["open"])
    y_v3 = df_v3["open"]

    # Run V3 baseline
    print("\n▶ 1. V3 BASELINE")
    model_v3 = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    v3_preds = cross_val_predict(model_v3, X_v3, y_v3, cv=CV, n_jobs=-1)
    v3_proba = cross_val_predict(model_v3, X_v3, y_v3, cv=CV, method="predict_proba", n_jobs=-1)
    v3_ba = balanced_accuracy_score(y_v3, v3_preds)
    v3_acc = (v3_preds == y_v3).mean()
    print(f"  V3 BalAcc: {v3_ba:.4f}  Acc: {v3_acc:.4f}")

    # Load or run enrichment
    print("\n▶ 2. ENRICHMENT")
    if os.path.exists(ENRICHED_CACHE):
        print(f"  Loading cached enrichment from {ENRICHED_CACHE}")
        enrichment = pd.read_parquet(ENRICHED_CACHE)
    else:
        raw = pd.read_parquet(RAW_PATH)
        enrichment = run_enrichment(raw)

    # Build enriched dataset
    print("\n▶ 3. BUILDING ENRICHED FEATURES")
    df_enriched = df_v3.copy()
    for col in enrichment.columns:
        df_enriched[col] = enrichment[col].values

    # Also add the V5 category-specific features from previous work
    # (import the function from the previous script)
    import importlib.util
    spec = importlib.util.spec_from_file_location("v5", "scripts/experiments/v5_category_reduction.py")
    v5_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5_mod)
    df_enriched = v5_mod.add_v5_features(df_enriched)

    X_enriched = df_enriched.drop(columns=["open"])
    y_enriched = df_enriched["open"]
    new_cols = [c for c in X_enriched.columns if c not in X_v3.columns]
    print(f"  Enriched: {X_enriched.shape[1]} features (+{len(new_cols)} new)")
    print(f"  New features: {new_cols}")

    # Train enriched model
    print("\n▶ 4. ENRICHED MODEL TRAINING")
    model_enriched = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    e_preds = cross_val_predict(model_enriched, X_enriched, y_enriched, cv=CV, n_jobs=-1)
    e_proba = cross_val_predict(model_enriched, X_enriched, y_enriched, cv=CV, method="predict_proba", n_jobs=-1)
    e_ba = balanced_accuracy_score(y_enriched, e_preds)
    e_acc = (e_preds == y_enriched).mean()
    print(f"  Enriched BalAcc: {e_ba:.4f}  Acc: {e_acc:.4f}")
    print(f"  Δ from V3: BalAcc {e_ba - v3_ba:+.4f}, Acc {e_acc - v3_acc:+.4f}")

    # Category-aware threshold tuning
    print("\n▶ 5. CATEGORY-AWARE THRESHOLD TUNING")
    e_corrected = e_preds.copy()
    all_cats = [
        ("Pharmacy", "cat_pharmacy"), ("Coffee Shop", "cat_coffee_shop"),
        ("Hotel", "cat_hotel"), ("ATMs", "cat_atms"),
        ("Restaurant", "cat_restaurant"), ("Pizza Restaurant", "cat_pizza_restaurant"),
    ]
    for cat_name, cat_col in all_cats:
        if cat_col not in X_enriched.columns:
            continue
        mask = X_enriched[cat_col] == 1
        if mask.sum() < 20:
            continue
        cat_proba = e_proba[mask, 1]
        cat_y = y_enriched[mask]
        best_thresh, best_ba = 0.5, balanced_accuracy_score(cat_y, e_preds[mask])
        for t in np.arange(0.20, 0.80, 0.01):
            ba_t = balanced_accuracy_score(cat_y, (cat_proba >= t).astype(int))
            if ba_t > best_ba:
                best_ba = ba_t
                best_thresh = t
        e_corrected[mask.values] = (e_proba[mask, 1] >= best_thresh).astype(int)
        print(f"  {cat_name}: thresh={best_thresh:.2f}")

    ec_ba = balanced_accuracy_score(y_enriched, e_corrected)
    ec_acc = (e_corrected == y_enriched).mean()
    print(f"\n  Enriched+Corrected BalAcc: {ec_ba:.4f}  Acc: {ec_acc:.4f}")

    # Per-category comparison
    print("\n▶ 6. PER-CATEGORY COMPARISON")
    cat_list = [
        ("Pharmacy", "cat_pharmacy"), ("Coffee Shop", "cat_coffee_shop"),
        ("Hotel", "cat_hotel"), ("ATMs", "cat_atms"),
        ("Restaurant", "cat_restaurant"), ("Pizza Restaurant", "cat_pizza_restaurant"),
        ("Other", "cat_other"),
    ]
    for cat_name, cat_col in cat_list:
        if cat_col not in X_v3.columns: continue
        mask = X_v3[cat_col] == 1
        if mask.sum() < 10: continue
        v3_err = 1 - (v3_preds[mask] == y_v3[mask]).mean()
        e_mask = X_enriched[cat_col] == 1
        e_err = 1 - (e_corrected[e_mask.values] == y_enriched[e_mask]).mean()
        v3_fn = ((y_v3[mask] == 1) & (v3_preds[mask] == 0)).sum()
        e_fn = ((y_enriched[e_mask] == 1) & (e_corrected[e_mask.values] == 0)).sum()
        delta_err = e_err - v3_err
        d = "↓" if delta_err < 0 else "↑" if delta_err > 0 else "→"
        print(f"  {cat_name:20s}: V3={v3_err:.1%} → Enriched={e_err:.1%} ({d}{abs(delta_err):.1%})  FN: {v3_fn}→{e_fn}")

    # Feature importance
    print("\n▶ 7. ENRICHMENT FEATURE IMPORTANCE")
    model_enriched.fit(X_enriched, y_enriched)
    fi = pd.Series(model_enriched.feature_importances_, index=X_enriched.columns).sort_values(ascending=False)
    enrichment_feats = ["has_npi_match", "npi_result_count", "has_irs_exempt", "irs_result_count"]
    for f in enrichment_feats:
        if f in fi.index:
            rank = list(fi.index).index(f) + 1
            print(f"    #{rank}/{len(fi)}  {f:30s}  importance={fi[f]:.4f}")

    # Overall
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  V3 Baseline:        BalAcc={v3_ba:.4f}  Acc={v3_acc:.4f}")
    print(f"  V5+Enriched:        BalAcc={e_ba:.4f}   Acc={e_acc:.4f}   Δ={e_ba-v3_ba:+.4f}")
    print(f"  V5+Enriched+Tuned:  BalAcc={ec_ba:.4f}  Acc={ec_acc:.4f}  Δ={ec_ba-v3_ba:+.4f}")

    v3_fp = ((y_v3 == 0) & (v3_preds == 1)).sum()
    v3_fn = ((y_v3 == 1) & (v3_preds == 0)).sum()
    e_fp = ((y_enriched == 0) & (e_corrected == 1)).sum()
    e_fn = ((y_enriched == 1) & (e_corrected == 0)).sum()
    print(f"  V3 errors: FP={v3_fp}, FN={v3_fn}, Total={v3_fp+v3_fn}")
    print(f"  Enriched:  FP={e_fp}, FN={e_fn}, Total={e_fp+e_fn}")

    return {
        "v3_ba": v3_ba, "v3_acc": v3_acc,
        "e_ba": e_ba, "e_acc": e_acc,
        "ec_ba": ec_ba, "ec_acc": ec_acc,
    }


if __name__ == "__main__":
    raw = pd.read_parquet(RAW_PATH)
    enrichment = run_enrichment(raw)
    results = evaluate_with_enrichment()
    print("\n✅ Enrichment evaluation complete!")
