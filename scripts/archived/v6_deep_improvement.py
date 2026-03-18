"""
V6 Deep Category Improvement — Pharmacy, Coffee Shop, Hotel

Implements multiple strategies using ONLY public data:
1. Website liveness crawling (HTTP status, SSL, redirects, meta keywords)
2. URL domain pattern analysis (chain vs independent detection)
3. Category-specific sub-models (specialist models for hard categories)
4. Meta-ensemble (base model + specialists)
5. Adaptive threshold tuning per category

Targets: Pharmacy (22.2%), Coffee Shop (15.2%), Hotel (14.7%)
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
import os
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
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
OUT_DIR = "pipeline_output"
CRAWL_CACHE = "data/website_crawl_cache.json"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

import importlib.util
spec = importlib.util.spec_from_file_location("v5", "scripts/experiments/v5_category_reduction.py")
v5_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5_mod)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def parse_json(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    if isinstance(x, str):
        try: return json.loads(x)
        except: return None
    return x if isinstance(x, (list, dict)) else None

def get_urls(x):
    d = parse_json(x)
    return [str(u) for u in d if isinstance(u, str)] if isinstance(d, list) else []

def get_socials(x):
    d = parse_json(x)
    return [str(u) for u in d if isinstance(u, str)] if isinstance(d, list) else []

def get_cat(x):
    d = parse_json(x)
    return d.get("primary", "") if isinstance(d, dict) else ""

def get_name(x):
    d = parse_json(x)
    return d.get("primary", "") if isinstance(d, dict) else ""

def get_addr(x):
    d = parse_json(x)
    if isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
        return d[0]
    return {}


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: WEBSITE LIVENESS CRAWLING
# ════════════════════════════════════════════════════════════════════════════
CLOSURE_KEYWORDS = [
    "permanently closed", "closed permanently", "we have closed",
    "no longer in business", "out of business", "shuttered",
    "temporarily closed", "closed for renovation", "coming soon",
    "under construction", "this location is closed", "this store is closed",
    "has been closed", "now closed", "we are closed",
]
OPEN_KEYWORDS = [
    "order now", "book now", "reserve", "schedule appointment",
    "online ordering", "place order", "add to cart", "shop now",
    "make a reservation", "check availability", "our menu",
    "open today", "now open", "grand opening", "hours of operation",
    "open daily", "open monday", "we are open",
]

def crawl_url(url, timeout=8):
    """
    Crawl a URL and extract liveness signals.
    Returns dict of features.
    """
    result = {
        "url_reachable": 0,
        "url_status_code": 0,
        "url_has_ssl": 0,
        "url_redirect_count": 0,
        "url_is_parked": 0,
        "url_closure_keywords": 0,
        "url_open_keywords": 0,
        "url_has_menu": 0,
        "url_has_booking": 0,
        "url_has_hours": 0,
        "url_content_length": 0,
    }

    if not url or not url.startswith("http"):
        return result

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StatusNow-Research/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, timeout=timeout, headers=headers,
                           allow_redirects=True, verify=False)

        result["url_reachable"] = 1
        result["url_status_code"] = resp.status_code
        result["url_has_ssl"] = 1 if url.startswith("https") else 0
        result["url_redirect_count"] = len(resp.history)
        result["url_content_length"] = len(resp.text)

        text_lower = resp.text.lower()[:50000]  # First 50KB

        # Parked domain detection
        parked_indicators = ["domain for sale", "buy this domain", "parked free",
                            "godaddy", "this domain", "domain parking", "sedoparking",
                            "domainmarket", "hugedomains", "afternic"]
        result["url_is_parked"] = int(any(p in text_lower for p in parked_indicators))

        # Closure keywords
        result["url_closure_keywords"] = sum(1 for k in CLOSURE_KEYWORDS if k in text_lower)

        # Open/active keywords
        result["url_open_keywords"] = sum(1 for k in OPEN_KEYWORDS if k in text_lower)

        # Has menu page (restaurants, coffee shops)
        menu_indicators = ["menu", "/menu", "our menu", "food menu", "drink menu",
                          "breakfast", "lunch", "dinner", "appetizer", "entree"]
        result["url_has_menu"] = int(sum(1 for m in menu_indicators if m in text_lower) >= 2)

        # Has booking/reservation (hotels)
        booking_indicators = ["book a room", "check availability", "reservation",
                            "book now", "check-in", "check-out", "room rate",
                            "per night", "book direct"]
        result["url_has_booking"] = int(sum(1 for b in booking_indicators if b in text_lower) >= 2)

        # Has hours
        hours_indicators = ["hours", "monday", "tuesday", "wednesday", "thursday",
                          "friday", "saturday", "sunday", "am -", "pm -", "a.m.", "p.m."]
        result["url_has_hours"] = int(sum(1 for h in hours_indicators if h in text_lower) >= 3)

    except requests.exceptions.SSLError:
        result["url_reachable"] = 0
        result["url_has_ssl"] = 0
    except requests.exceptions.Timeout:
        result["url_reachable"] = 0
    except Exception:
        result["url_reachable"] = 0

    return result


def run_website_crawling(raw_df):
    """Crawl websites for all places that have URLs."""
    print("\n" + "=" * 80)
    print("STRATEGY 1: WEBSITE LIVENESS CRAWLING")
    print("=" * 80)

    # Check cache
    if os.path.exists(CRAWL_CACHE):
        print(f"  Loading cached crawl results from {CRAWL_CACHE}")
        with open(CRAWL_CACHE) as f:
            return json.load(f)

    urls_list = raw_df["websites"].apply(get_urls)
    first_urls = urls_list.apply(lambda x: x[0] if x else "")

    to_crawl = [(i, url) for i, url in enumerate(first_urls) if url]
    print(f"  Total places with URLs: {len(to_crawl)}")
    print(f"  Crawling …")

    results = {}
    for batch_start in range(0, len(to_crawl), 20):
        batch = to_crawl[batch_start:batch_start + 20]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(crawl_url, url): idx for idx, url in batch}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[str(idx)] = future.result()
                except:
                    results[str(idx)] = crawl_url("")  # empty result

        done = min(batch_start + 20, len(to_crawl))
        reachable = sum(1 for r in results.values() if r.get("url_reachable"))
        print(f"    [{done}/{len(to_crawl)}] reachable: {reachable}")
        time.sleep(0.5)

    # Save cache
    with open(CRAWL_CACHE, "w") as f:
        json.dump(results, f)
    print(f"  Saved crawl cache ({len(results)} results)")

    return results


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: URL DOMAIN PATTERN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
CHAIN_DOMAINS = [
    "starbucks.com", "walmart.com", "walgreens.com", "cvs.com",
    "marriott.com", "hilton.com", "ihg.com", "choicehotels.com",
    "bestwestern.com", "wyndhamhotels.com", "hyatt.com",
    "panerabread.com", "dunkindonuts.com", "mcdonalds.com",
    "rfrk.com", "gianteagle.com", "kroger.com", "target.com",
    "costco.com", "tchibo.de", "costa.co.uk",
]

BOOKING_DOMAINS = [
    "booking.com", "expedia.com", "hotels.com", "agoda.com",
    "trivago.com", "tripadvisor.com", "airbnb.com",
]

def extract_url_features(url):
    """Extract features from URL structure alone (no crawling needed)."""
    if not url:
        return {"url_is_chain": 0, "url_is_booking_platform": 0,
                "url_has_location_path": 0, "url_domain_len": 0}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.lower()
    except:
        return {"url_is_chain": 0, "url_is_booking_platform": 0,
                "url_has_location_path": 0, "url_domain_len": 0}

    return {
        "url_is_chain": int(any(c in domain for c in CHAIN_DOMAINS)),
        "url_is_booking_platform": int(any(b in domain for b in BOOKING_DOMAINS)),
        "url_has_location_path": int(any(p in path for p in [
            "/store/", "/location", "/hotel", "/pharmacy", "/cafe",
            "/restaurant", "/branch", "/store-locator"
        ])),
        "url_domain_len": len(domain),
    }


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: SOCIAL MEDIA SIGNAL EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
def extract_social_features(socials):
    """Extract features from social media URLs."""
    social_list = get_socials(socials)
    n = len(social_list)
    has_fb = int(any("facebook.com" in s for s in social_list))
    has_ig = int(any("instagram.com" in s for s in social_list))
    has_yelp = int(any("yelp.com" in s for s in social_list))
    has_tripadvisor = int(any("tripadvisor" in s for s in social_list))
    has_booking = int(any("booking.com" in s for s in social_list))

    return {
        "social_count": n,
        "has_tripadvisor": has_tripadvisor,
        "has_booking_social": has_booking,
        "social_diversity": len(set(
            urlparse(s).netloc.replace("www.", "")
            for s in social_list if s.startswith("http")
        )),
    }


# ════════════════════════════════════════════════════════════════════════════
# BUILD ALL V6 FEATURES
# ════════════════════════════════════════════════════════════════════════════
def build_v6_features(v3_df, raw_df, crawl_results):
    """Build V6 = V5 features + website crawl + URL patterns + sub-model predictions."""
    print("\n" + "=" * 80)
    print("BUILDING V6 FEATURES")
    print("=" * 80)

    # Start with V5
    df = v5_mod.add_v5_features(v3_df)

    # Add enrichment if available
    if os.path.exists("data/enrichment_cache.parquet"):
        enrich = pd.read_parquet("data/enrichment_cache.parquet")
        for col in enrich.columns:
            df[col] = enrich[col].values

    # ── URL domain features ───────────────────────────────────────────────
    print("  Adding URL domain features …")
    urls = raw_df["websites"].apply(get_urls)
    first_url = urls.apply(lambda x: x[0] if x else "")
    url_feats = first_url.apply(extract_url_features).apply(pd.Series)
    for col in url_feats.columns:
        df[col] = url_feats[col].values

    # ── Crawl features ────────────────────────────────────────────────────
    print("  Adding website crawl features …")
    crawl_cols = [
        "url_reachable", "url_status_code", "url_has_ssl",
        "url_redirect_count", "url_is_parked",
        "url_closure_keywords", "url_open_keywords",
        "url_has_menu", "url_has_booking", "url_has_hours",
        "url_content_length",
    ]
    for col in crawl_cols:
        df[col] = 0

    for idx_str, result in crawl_results.items():
        idx = int(idx_str)
        if idx < len(df):
            for col in crawl_cols:
                df.iloc[idx, df.columns.get_loc(col)] = result.get(col, 0)

    # Derived crawl features
    df["url_is_alive"] = ((df["url_reachable"] == 1) & (df["url_status_code"].isin([200, 301, 302]))).astype(int)
    df["url_is_dead"] = ((df["url_reachable"] == 0) | (df["url_status_code"].isin([404, 410, 500, 503]))).astype(int)
    df["url_net_sentiment"] = df["url_open_keywords"] - df["url_closure_keywords"]
    df["url_activity_score"] = df["url_has_menu"] + df["url_has_booking"] + df["url_has_hours"] + df["url_open_keywords"].clip(upper=3)

    # ── Social features ───────────────────────────────────────────────────
    print("  Adding social signal features …")
    social_feats = raw_df["socials"].apply(extract_social_features).apply(pd.Series)
    for col in social_feats.columns:
        df[col] = social_feats[col].values

    # ── Category-specific crawl interactions ──────────────────────────────────
    print("  Adding category × crawl interactions …")
    all_cat_interactions = [
        ("cat_pharmacy", "pharmacy"),
        ("cat_coffee_shop", "coffee"),
        ("cat_hotel", "hotel"),
        ("cat_atms", "atm"),
        ("cat_restaurant", "restaurant"),
        ("cat_pizza_restaurant", "pizza"),
        ("cat_dentist", "dentist"),
        ("cat_doctor", "doctor"),
        ("cat_hair_salon", "hairsalon"),
        ("cat_beauty_salon", "beautysalon"),
        ("cat_clothing_store", "clothing"),
        ("cat_jewelry_store", "jewelry"),
        ("cat_real_estate_agent", "realestate"),
        ("cat_community_services_non_profits", "community"),
        ("cat_landmark_and_historical_building", "landmark"),
    ]

    for cat_col, prefix in all_cat_interactions:
        is_cat = df.get(cat_col, pd.Series(0, index=df.index))
        if isinstance(is_cat, (int, float)) and is_cat == 0:
            continue
        df[f"{prefix}_url_alive"] = is_cat * df["url_is_alive"]
        df[f"{prefix}_url_dead"] = is_cat * df["url_is_dead"]
        df[f"{prefix}_activity"] = is_cat * df["url_activity_score"]

    # Specialty interactions
    is_hotel = df.get("cat_hotel", 0)
    is_coffee = df.get("cat_coffee_shop", 0)
    df["hotel_has_booking_widget"] = is_hotel * df["url_has_booking"]
    df["coffee_has_menu"] = is_coffee * df["url_has_menu"]
    df["hotel_url_sentiment"] = is_hotel * df["url_net_sentiment"]

    print(f"  V6 total features: {len(df.columns) - 1}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: CATEGORY-SPECIFIC SUB-MODELS + ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
def run_ensemble_evaluation(X, y):
    """
    Train base model + category specialists, combine with meta-learner.
    """
    print("\n" + "=" * 80)
    print("STRATEGY 4: CATEGORY-SPECIFIC SUB-MODELS + ENSEMBLE")
    print("=" * 80)

    base_cfg = dict(iterations=500, learning_rate=0.05, depth=6, verbose=0,
                    auto_class_weights="Balanced", random_state=42, allow_writing_files=False)

    # Base model OOF predictions
    print("  Base model OOF predictions …")
    base_oof_proba = cross_val_predict(
        CatBoostClassifier(**base_cfg), X, y, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]

    # Category specialist OOF predictions — ALL categories with 50+ samples
    specialist_cats = [
        ("cat_pharmacy", "Pharmacy"),
        ("cat_coffee_shop", "Coffee Shop"),
        ("cat_hotel", "Hotel"),
        ("cat_atms", "ATMs"),
        ("cat_restaurant", "Restaurant"),
        ("cat_pizza_restaurant", "Pizza Restaurant"),
        ("cat_dentist", "Dentist"),
        ("cat_doctor", "Doctor"),
        ("cat_hair_salon", "Hair Salon"),
        ("cat_beauty_salon", "Beauty Salon"),
        ("cat_clothing_store", "Clothing Store"),
        ("cat_jewelry_store", "Jewelry Store"),
        ("cat_church_cathedral", "Church Cathedral"),
        ("cat_community_services_non_profits", "Community Services"),
        ("cat_professional_services", "Professional Services"),
        ("cat_real_estate_agent", "Real Estate Agent"),
        ("cat_landmark_and_historical_building", "Landmark"),
        ("cat_art_gallery", "Art Gallery"),
        ("cat_corporate_office", "Corporate Office"),
    ]

    specialist_oof = np.full(len(y), np.nan)
    for cat_col, cat_name in specialist_cats:
        if cat_col not in X.columns:
            continue
        mask = X[cat_col] == 1
        if mask.sum() < 30:
            continue

        print(f"  Training specialist: {cat_name} (n={mask.sum()}) …")
        cat_X = X[mask]
        cat_y = y[mask]

        # Use deeper model for specialists
        spec_cfg = dict(iterations=800, learning_rate=0.03, depth=8, verbose=0,
                        auto_class_weights="Balanced", random_state=42,
                        allow_writing_files=False, l2_leaf_reg=3)

        if cat_y.nunique() < 2:
            continue

        spec_proba = cross_val_predict(
            CatBoostClassifier(**spec_cfg), cat_X, cat_y, cv=CV,
            method="predict_proba", n_jobs=-1
        )[:, 1]
        specialist_oof[mask.values] = spec_proba

    # Combine: where specialist exists, blend base + specialist
    alpha = 0.4  # specialist weight
    ensemble_proba = base_oof_proba.copy()
    has_specialist = ~np.isnan(specialist_oof)
    ensemble_proba[has_specialist] = (
        (1 - alpha) * base_oof_proba[has_specialist] +
        alpha * specialist_oof[has_specialist]
    )

    # Threshold tune the ensemble
    best_global_thresh = 0.5
    best_global_ba = 0
    for t in np.arange(0.3, 0.7, 0.01):
        ba = balanced_accuracy_score(y, (ensemble_proba >= t).astype(int))
        if ba > best_global_ba:
            best_global_ba = ba
            best_global_thresh = t

    ensemble_preds = (ensemble_proba >= best_global_thresh).astype(int)
    print(f"\n  Ensemble global thresh: {best_global_thresh:.2f}")
    print(f"  Ensemble BalAcc: {best_global_ba:.4f}")

    # Per-category threshold tuning on ensemble — ALL categories
    all_threshold_cats = specialist_cats + [
        ("cat_package_locker", "Package Locker"),
    ]
    for cat_col, cat_name in all_threshold_cats:
        if cat_col not in X.columns: continue
        mask = X[cat_col] == 1
        if mask.sum() < 20: continue
        cat_proba = ensemble_proba[mask.values]
        cat_y_local = y[mask]
        best_t, best_ba = best_global_thresh, balanced_accuracy_score(cat_y_local, (cat_proba >= best_global_thresh).astype(int))
        for t in np.arange(0.15, 0.85, 0.01):
            ba = balanced_accuracy_score(cat_y_local, (cat_proba >= t).astype(int))
            if ba > best_ba:
                best_ba = ba
                best_t = t
        ensemble_preds[mask.values] = (ensemble_proba[mask.values] >= best_t).astype(int)
        print(f"  {cat_name}: thresh={best_t:.2f} → BalAcc={best_ba:.4f}")

    return ensemble_preds, ensemble_proba


# ════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
def full_evaluation():
    print("=" * 80)
    print("V6 DEEP CATEGORY IMPROVEMENT — FULL EVALUATION")
    print("=" * 80)

    # ── V3 baseline ───────────────────────────────────────────────────────
    print("\n▶ 1. V3 BASELINE")
    df_v3 = pd.read_parquet(V3_PATH)
    X_v3 = df_v3.drop(columns=["open"])
    y = df_v3["open"]
    base_cfg = dict(iterations=500, learning_rate=0.05, depth=6, verbose=0,
                    auto_class_weights="Balanced", random_state=42, allow_writing_files=False)
    v3_preds = cross_val_predict(CatBoostClassifier(**base_cfg), X_v3, y, cv=CV, n_jobs=-1)
    v3_proba = cross_val_predict(CatBoostClassifier(**base_cfg), X_v3, y, cv=CV, method="predict_proba", n_jobs=-1)
    v3_ba = balanced_accuracy_score(y, v3_preds)
    print(f"  V3 BalAcc: {v3_ba:.4f}")

    # ── Website crawling ──────────────────────────────────────────────────
    print("\n▶ 2. WEBSITE CRAWLING")
    raw = pd.read_parquet(RAW_PATH)
    crawl_results = run_website_crawling(raw)
    reachable = sum(1 for r in crawl_results.values() if r.get("url_reachable"))
    print(f"  Reachable: {reachable}/{len(crawl_results)}")

    # ── Build V6 features ─────────────────────────────────────────────────
    print("\n▶ 3. V6 FEATURES")
    df_v6 = build_v6_features(df_v3, raw, crawl_results)
    X_v6 = df_v6.drop(columns=["open"])
    y_v6 = df_v6["open"]

    new_feats = [c for c in X_v6.columns if c not in X_v3.columns]
    print(f"  V6: {X_v6.shape[1]} features (+{len(new_feats)} new vs V3)")

    # ── V6 base model ─────────────────────────────────────────────────────
    print("\n▶ 4. V6 BASE MODEL")
    v6_preds = cross_val_predict(CatBoostClassifier(**base_cfg), X_v6, y_v6, cv=CV, n_jobs=-1)
    v6_proba = cross_val_predict(CatBoostClassifier(**base_cfg), X_v6, y_v6, cv=CV, method="predict_proba", n_jobs=-1)
    v6_ba = balanced_accuracy_score(y_v6, v6_preds)
    print(f"  V6 Base BalAcc: {v6_ba:.4f} (Δ={v6_ba-v3_ba:+.4f})")

    # ── V6 ensemble with specialists ──────────────────────────────────────
    print("\n▶ 5. V6 ENSEMBLE WITH CATEGORY SPECIALISTS")
    v6e_preds, v6e_proba = run_ensemble_evaluation(X_v6, y_v6)
    v6e_ba = balanced_accuracy_score(y_v6, v6e_preds)
    v6e_acc = accuracy_score(y_v6, v6e_preds)
    print(f"\n  V6 Ensemble BalAcc: {v6e_ba:.4f} (Δ={v6e_ba-v3_ba:+.4f})")

    # ── Per-category comparison ───────────────────────────────────────────
    print("\n▶ 6. PER-CATEGORY COMPARISON")
    cats = [
        ("Pharmacy", "cat_pharmacy"), ("Coffee Shop", "cat_coffee_shop"),
        ("Hotel", "cat_hotel"), ("ATMs", "cat_atms"),
        ("Restaurant", "cat_restaurant"), ("Pizza Restaurant", "cat_pizza_restaurant"),
        ("Other", "cat_other"), ("Package Locker", "cat_package_locker"),
        ("Corporate Office", "cat_corporate_office"), ("Art Gallery", "cat_art_gallery"),
        ("Dentist", "cat_dentist"), ("Doctor", "cat_doctor"),
        ("Hair Salon", "cat_hair_salon"), ("Beauty Salon", "cat_beauty_salon"),
        ("Clothing Store", "cat_clothing_store"), ("Jewelry Store", "cat_jewelry_store"),
        ("Church Cathedral", "cat_church_cathedral"),
        ("Community Services", "cat_community_services_non_profits"),
        ("Professional Services", "cat_professional_services"),
        ("Real Estate Agent", "cat_real_estate_agent"),
        ("Landmark", "cat_landmark_and_historical_building"),
    ]
    print(f"\n  {'Category':<20} {'N':>4}  {'V3 Err':>7} {'V6E Err':>7} {'Δ':>7}  {'V3 P':>5} {'V6 P':>5}  {'V3 R':>5} {'V6 R':>5}  {'V3 F1':>5} {'V6 F1':>5}  {'V3FN':>4} {'V6FN':>4}")
    print("  " + "-"*105)

    cat_results = []
    for cat_name, cat_col in cats:
        if cat_col not in X_v3.columns: continue
        m = X_v3[cat_col] == 1
        m6 = X_v6[cat_col] == 1
        if m.sum() < 10: continue
        n = m.sum()
        cy3 = y[m]; cy6 = y_v6[m6]
        cp3 = v3_preds[m]; cp6 = v6e_preds[m6.values]

        v3e = 1-accuracy_score(cy3, cp3)
        v6e_err = 1-accuracy_score(cy6, cp6)
        v3p = precision_score(cy3, cp3, pos_label=1, zero_division=0)
        v6p = precision_score(cy6, cp6, pos_label=1, zero_division=0)
        v3r = recall_score(cy3, cp3, pos_label=1, zero_division=0)
        v6r = recall_score(cy6, cp6, pos_label=1, zero_division=0)
        v3f = f1_score(cy3, cp3, pos_label=1, zero_division=0)
        v6f = f1_score(cy6, cp6, pos_label=1, zero_division=0)
        v3fn = ((cy3==1)&(cp3==0)).sum()
        v6fn = ((cy6==1)&(cp6==0)).sum()
        d = "↓" if v6e_err < v3e else "↑" if v6e_err > v3e else "→"

        cat_results.append({
            "Category": cat_name, "N": n,
            "V3 Err": v3e, "V6 Err": v6e_err,
            "V3 P": v3p, "V6 P": v6p,
            "V3 R": v3r, "V6 R": v6r,
            "V3 F1": v3f, "V6 F1": v6f,
            "V3 FN": v3fn, "V6 FN": v6fn,
        })
        print(f"  {cat_name:<20} {n:>4}  {v3e:>7.1%} {v6e_err:>7.1%} {d}{abs(v6e_err-v3e):>5.1%}  {v3p:>5.1%} {v6p:>5.1%}  {v3r:>5.1%} {v6r:>5.1%}  {v3f:>5.1%} {v6f:>5.1%}  {v3fn:>4} {v6fn:>4}")

    # ── Feature importance for new features ───────────────────────────────
    print("\n▶ 7. NEW FEATURE IMPORTANCE")
    m = CatBoostClassifier(**base_cfg)
    m.fit(X_v6, y_v6)
    fi = pd.Series(m.feature_importances_, index=X_v6.columns).sort_values(ascending=False)

    web_feats = [f for f in new_feats if f in fi.index]
    web_feats_sorted = sorted(web_feats, key=lambda f: fi[f], reverse=True)
    print("  New V6 features (by importance):")
    for f in web_feats_sorted[:20]:
        rank = list(fi.index).index(f) + 1
        print(f"    #{rank:3d}/{len(fi)}  {f:35s}  {fi[f]:.4f}")

    # ── Generate Charts ───────────────────────────────────────────────────
    print("\n▶ 8. GENERATING CHARTS")
    generate_v6_charts(cat_results, v3_ba, v6_ba, v6e_ba)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    v3_total = (v3_preds != y).sum()
    v6_total = (v6e_preds != y_v6).sum()
    print(f"  V3 Baseline:    BalAcc={v3_ba:.4f}  Errors={v3_total}")
    print(f"  V6 Base:        BalAcc={v6_ba:.4f}  Δ={v6_ba-v3_ba:+.4f}")
    print(f"  V6 Ensemble:    BalAcc={v6e_ba:.4f}  Δ={v6e_ba-v3_ba:+.4f}  Errors={v6_total}")

    return cat_results


def generate_v6_charts(cat_results, v3_ba, v6_ba, v6e_ba):
    """Generate V6 comparison charts."""
    cr = pd.DataFrame(cat_results)

    # Chart 1: Per-category P/R/F1 comparison
    target_cats = cr[cr["Category"].isin(["Pharmacy", "Coffee Shop", "Hotel"])]
    if len(target_cats) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for idx, metric in enumerate(["P", "R", "F1"]):
            ax = axes[idx]
            cats = target_cats["Category"]
            y_pos = np.arange(len(cats))
            bh = 0.3
            ax.barh(y_pos+bh/2, target_cats[f"V3 {metric}"], bh, color="#94a3b8", label="V3", alpha=0.9)
            ax.barh(y_pos-bh/2, target_cats[f"V6 {metric}"], bh, color="#22c55e", label="V6 Ensemble", alpha=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cats, fontsize=11)
            full_name = {"P": "Precision", "R": "Recall", "F1": "F1 Score"}[metric]
            ax.set_title(full_name, fontsize=13, fontweight="bold")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.legend(fontsize=9)
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig.suptitle("V3 vs V6: Precision, Recall & F1 for Target Categories",
                     fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "v6_target_prf.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: v6_target_prf.png")
        plt.close()

    # Chart 2: Error rate + FN reduction for all cats
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(cr))
    bh = 0.3
    ax.barh(y_pos+bh/2, cr["V3 Err"], bh, color="#ef4444", label="V3 Error Rate", alpha=0.85)
    ax.barh(y_pos-bh/2, cr["V6 Err"], bh, color="#22c55e", label="V6 Ensemble Error Rate", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Category']} (n={r['N']})" for _, r in cr.iterrows()], fontsize=10)
    ax.set_xlabel("Error Rate")
    ax.set_title("Error Rate: V3 Baseline vs V6 Ensemble", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for i, (_, r) in enumerate(cr.iterrows()):
        delta = r["V6 Err"] - r["V3 Err"]
        color = "#22c55e" if delta < 0 else "#ef4444"
        ax.text(max(r["V3 Err"], r["V6 Err"])+0.005, i, f"{delta:+.1%}", va="center", fontsize=8, fontweight="600", color=color)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v6_error_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: v6_error_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = full_evaluation()
    print("\n✅ V6 evaluation complete!")
