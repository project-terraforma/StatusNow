"""
V6 Enrichment Experiment — V5 + Website Crawling + Public Records
==================================================================

Builds on the V5 best model by adding two enrichment layers:

  1. Website signals (no auth required):
       - Live crawl: reachability, status code, SSL, redirect count,
         closure/open keywords, menu/booking/hours detection
       - URL domain patterns: chain domain, booking platform, location path
       - Social URL analysis: platform diversity, TripAdvisor/booking presence

  2. Public records (pre-fetched, optional):
       - Business license matches (NYC DCWP, NYC SLA, SF Businesses, CA ABC)
       - Health inspection history (NYC DOHMH)
       - Derived: license_expected_but_not_found, expired license flags
       - Load from: data/public_records/license_features.parquet
         (generate with: scripts/archived/public_records/match_records.py
                          + scripts/archived/public_records/engineer_license_features.py)

Evaluation
----------
Same geographic hold-out as V5: Chicago + Miami never seen in training.
Reports V5 baseline side-by-side with each enrichment tier so we can see
exactly what each layer adds.

Crawl caching
-------------
Website crawls are cached to data/website_crawl_cache.json — first run
on 123k rows will take a while (5 workers, ~8s timeout). Subsequent runs
load from cache instantly.

Usage
-----
  python scripts/experiments/v6_enrichment_experiment.py

  Options:
    --input   FILE   Truth dataset (default: data/combined_truth_dataset_expanded.parquet)
    --records FILE   License features parquet (default: data/public_records/license_features.parquet)
    --crawl-cache FILE  Crawl cache path (default: data/website_crawl_cache.json)
    --no-crawl       Skip website crawling (use only public records enrichment)
"""

import argparse
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from research.process_data_v5 import build_v5_features


# ── Configuration ─────────────────────────────────────────────────────────────

HOLDOUT_CITIES = ["overture_chicago", "overture_miami"]
RANDOM_SEED    = 42

BEST_CB_PARAMS = dict(
    iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
    auto_class_weights="Balanced", verbose=0,
    random_state=RANDOM_SEED, allow_writing_files=False,
)
BEST_LGBM_PARAMS = dict(
    n_estimators=1500, learning_rate=0.05, max_depth=8,
    num_leaves=80, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)
W_CB, W_LGBM, THRESH = 0.7, 0.3, 0.52


# ── Website crawling ──────────────────────────────────────────────────────────

CLOSURE_KEYWORDS = [
    "permanently closed", "closed permanently", "we have closed",
    "no longer in business", "out of business", "shuttered",
    "temporarily closed", "this location is closed", "this store is closed",
    "has been closed", "now closed",
]
OPEN_KEYWORDS = [
    "order now", "book now", "reserve", "schedule appointment",
    "online ordering", "place order", "our menu", "shop now",
    "make a reservation", "open today", "now open", "hours of operation",
    "open daily", "we are open",
]


def crawl_url(url, timeout=8):
    result = {
        "url_reachable": 0, "url_status_code": 0, "url_has_ssl": 0,
        "url_redirect_count": 0, "url_is_parked": 0,
        "url_closure_keywords": 0, "url_open_keywords": 0,
        "url_has_menu": 0, "url_has_booking": 0, "url_has_hours": 0,
        "url_content_length": 0,
    }
    if not url or not url.startswith("http"):
        return result
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StatusNow-Research/1.0)",
                   "Accept": "text/html,application/xhtml+xml"}
        resp = requests.get(url, timeout=timeout, headers=headers,
                            allow_redirects=True, verify=False)
        result["url_reachable"]       = 1
        result["url_status_code"]     = resp.status_code
        result["url_has_ssl"]         = int(url.startswith("https"))
        result["url_redirect_count"]  = len(resp.history)
        result["url_content_length"]  = len(resp.text)
        t = resp.text.lower()[:50000]
        parked = ["domain for sale", "buy this domain", "parked free",
                  "domain parking", "sedoparking", "hugedomains"]
        result["url_is_parked"]           = int(any(p in t for p in parked))
        result["url_closure_keywords"]    = sum(1 for k in CLOSURE_KEYWORDS if k in t)
        result["url_open_keywords"]       = sum(1 for k in OPEN_KEYWORDS if k in t)
        result["url_has_menu"]    = int(sum(1 for m in ["menu", "our menu", "food menu",
                                    "breakfast", "lunch", "dinner"] if m in t) >= 2)
        result["url_has_booking"] = int(sum(1 for b in ["book a room", "check availability",
                                    "reservation", "check-in", "per night"] if b in t) >= 2)
        result["url_has_hours"]   = int(sum(1 for h in ["hours", "monday", "tuesday",
                                    "wednesday", "am -", "pm -"] if h in t) >= 3)
    except Exception:
        pass
    return result


CHAIN_DOMAINS = [
    "starbucks.com", "walmart.com", "walgreens.com", "cvs.com", "marriott.com",
    "hilton.com", "ihg.com", "bestwestern.com", "wyndhamhotels.com", "hyatt.com",
    "panerabread.com", "dunkindonuts.com", "mcdonalds.com", "target.com", "kroger.com",
]
BOOKING_DOMAINS = ["booking.com", "expedia.com", "hotels.com", "agoda.com",
                   "tripadvisor.com", "airbnb.com"]


def extract_url_features(url):
    if not url:
        return {"url_is_chain": 0, "url_is_booking_platform": 0,
                "url_has_location_path": 0, "url_domain_len": 0}
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path   = parsed.path.lower()
    except Exception:
        return {"url_is_chain": 0, "url_is_booking_platform": 0,
                "url_has_location_path": 0, "url_domain_len": 0}
    return {
        "url_is_chain":            int(any(c in domain for c in CHAIN_DOMAINS)),
        "url_is_booking_platform": int(any(b in domain for b in BOOKING_DOMAINS)),
        "url_has_location_path":   int(any(p in path for p in [
                                       "/store/", "/location", "/hotel", "/pharmacy",
                                       "/cafe", "/restaurant", "/branch"])),
        "url_domain_len":          len(domain),
    }


def extract_social_features(social_json):
    if not social_json or (isinstance(social_json, float) and pd.isna(social_json)):
        return {"social_diversity": 0, "has_tripadvisor": 0, "has_booking_social": 0}
    try:
        socials = json.loads(social_json) if isinstance(social_json, str) else social_json
        if not isinstance(socials, list):
            socials = []
    except Exception:
        socials = []
    return {
        "social_diversity":    len(set(urlparse(s).netloc.replace("www.", "")
                                   for s in socials if isinstance(s, str) and s.startswith("http"))),
        "has_tripadvisor":     int(any("tripadvisor" in str(s) for s in socials)),
        "has_booking_social":  int(any("booking.com" in str(s) for s in socials)),
    }


def get_first_url(website_json):
    if not website_json or (isinstance(website_json, float) and pd.isna(website_json)):
        return ""
    try:
        urls = json.loads(website_json) if isinstance(website_json, str) else website_json
        return urls[0] if isinstance(urls, list) and urls else ""
    except Exception:
        return ""


def run_crawl(raw_df, cache_path):
    """Crawl all URLs in the dataset, using cache to skip already-done rows."""
    if os.path.exists(cache_path):
        print(f"  Loading crawl cache: {cache_path}")
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    first_urls = raw_df["websites"].apply(get_first_url)
    to_crawl = [(str(i), url) for i, url in enumerate(first_urls)
                if url and str(i) not in cache]

    if to_crawl:
        print(f"  Crawling {len(to_crawl):,} new URLs (cached: {len(cache):,}) …")
        for batch_start in range(0, len(to_crawl), 20):
            batch = to_crawl[batch_start:batch_start + 20]
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(crawl_url, url): idx for idx, url in batch}
                for future in as_completed(futures):
                    cache[futures[future]] = future.result()
            done = min(batch_start + 20, len(to_crawl))
            reachable = sum(1 for r in cache.values() if r.get("url_reachable"))
            print(f"    [{done}/{len(to_crawl)}]  reachable so far: {reachable:,}")
            time.sleep(0.3)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"  Cache saved ({len(cache):,} entries)")
    else:
        print(f"  All {len(cache):,} entries already cached — skipping crawl")

    return cache


# ── Feature building ──────────────────────────────────────────────────────────

CRAWL_COLS = [
    "url_reachable", "url_status_code", "url_has_ssl", "url_redirect_count",
    "url_is_parked", "url_closure_keywords", "url_open_keywords",
    "url_has_menu", "url_has_booking", "url_has_hours", "url_content_length",
]


def add_website_features(df, raw_df, crawl_cache):
    """Merge website crawl + URL domain + social features into df (aligned by index)."""
    first_urls = raw_df["websites"].apply(get_first_url)

    # URL domain (no crawl needed)
    url_feats = first_urls.apply(extract_url_features).apply(pd.Series)
    for col in url_feats.columns:
        df[col] = url_feats[col].values

    # Crawl features
    for col in CRAWL_COLS:
        df[col] = 0
    for idx_str, result in crawl_cache.items():
        idx = int(idx_str)
        if idx < len(df):
            for col in CRAWL_COLS:
                df.iloc[idx, df.columns.get_loc(col)] = result.get(col, 0)

    # Derived
    df["url_is_alive"]       = ((df["url_reachable"] == 1) &
                                (df["url_status_code"].isin([200, 301, 302]))).astype(int)
    df["url_is_dead"]        = ((df["url_reachable"] == 0) |
                                (df["url_status_code"].isin([404, 410, 500, 503]))).astype(int)
    df["url_net_sentiment"]  = df["url_open_keywords"] - df["url_closure_keywords"]
    df["url_activity_score"] = (df["url_has_menu"] + df["url_has_booking"] +
                                df["url_has_hours"] + df["url_open_keywords"].clip(upper=3))

    # Social features
    social_col = "socials" if "socials" in raw_df.columns else None
    if social_col:
        soc_feats = raw_df[social_col].apply(extract_social_features).apply(pd.Series)
        for col in soc_feats.columns:
            df[col] = soc_feats[col].values

    return df


def add_public_records_features(df, records_path):
    """Merge license/inspection features by poi_id. Returns (df, n_features_added)."""
    if not os.path.exists(records_path):
        return df, 0
    records = pd.read_parquet(records_path)
    feat_cols = [c for c in records.columns if c != "poi_id"]
    if "id" not in df.columns:
        print(f"  ⚠️  'id' column not found in dataset — skipping public records merge")
        return df, 0
    merged = df.merge(records[["poi_id"] + feat_cols], left_on="id", right_on="poi_id", how="left")
    for col in feat_cols:
        merged[col] = merged[col].fillna(0)
    if "poi_id" in merged.columns:
        merged = merged.drop(columns=["poi_id"])
    return merged, len(feat_cols)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def eval_ensemble(cb_prob, lgbm_prob, y_test, label):
    avg_prob = W_CB * cb_prob + W_LGBM * lgbm_prob
    pred     = (avg_prob >= THRESH).astype(int)
    ba   = balanced_accuracy_score(y_test, pred)
    auc  = roc_auc_score(y_test, avg_prob)
    cm   = confusion_matrix(y_test, pred)
    cl_r = cm[0, 0] / cm[0].sum()
    op_r = cm[1, 1] / cm[1].sum()
    print(f"\n  {label}")
    print(f"    BalAcc={ba:.4f}  AUC={auc:.4f}  "
          f"Closed Recall={cl_r:.3f}  Open Recall={op_r:.3f}")
    return ba, auc, cl_r, op_r


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path, records_path, crawl_cache_path, no_crawl):
    print("=" * 70)
    print("V6 ENRICHMENT EXPERIMENT")
    print("Website crawling + public records on top of V5 best model")
    print("Hold-out: Chicago + Miami")
    print("=" * 70)

    # ── Load raw data + V5 features ───────────────────────────────────────────
    print(f"\nLoading raw data: {input_path}")
    raw_df = pd.read_parquet(input_path).reset_index(drop=True)

    print("Building V5 features …")
    df, numeric_feats = build_v5_features(input_path)
    df = df.reset_index(drop=True)

    # Carry over id column from raw for public records merge
    if "id" in raw_df.columns:
        df["id"] = raw_df["id"].values

    holdout_mask = df["source_dataset"].isin(HOLDOUT_CITIES)
    train_idx    = df[~holdout_mask].index
    test_idx     = df[holdout_mask].index

    print(f"  Train: {len(train_idx):,}  |  Test (hold-out): {len(test_idx):,}")

    # ── V5 baseline ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — V5 BASELINE")
    print("=" * 70)

    all_feat_cols = numeric_feats + ["category_primary"]
    df["category_primary"] = df["category_primary"].astype(str)
    cat_idx = [all_feat_cols.index("category_primary")]

    X_tr_v5  = df.loc[train_idx, all_feat_cols]
    y_tr     = df.loc[train_idx, "open"]
    X_te_v5  = df.loc[test_idx, all_feat_cols]
    y_te     = df.loc[test_idx, "open"]
    X_tr_num = df.loc[train_idx, numeric_feats]
    X_te_num = df.loc[test_idx, numeric_feats]

    print("  Training CatBoost-C … ", end="", flush=True)
    cb_v5 = CatBoostClassifier(**BEST_CB_PARAMS, cat_features=cat_idx)
    cb_v5.fit(X_tr_v5, y_tr)
    cb_prob_v5 = cb_v5.predict_proba(X_te_v5)[:, 1]
    print("done")

    print("  Training LightGBM-A … ", end="", flush=True)
    lgbm_v5 = LGBMClassifier(**BEST_LGBM_PARAMS)
    lgbm_v5.fit(X_tr_num, y_tr)
    lgbm_prob_v5 = lgbm_v5.predict_proba(X_te_num)[:, 1]
    print("done")

    ba_v5, *_ = eval_ensemble(cb_prob_v5, lgbm_prob_v5, y_te, "V5 baseline (CatBoost-C + LightGBM-A)")

    # ── Website enrichment ────────────────────────────────────────────────────
    if not no_crawl:
        print("\n" + "=" * 70)
        print("STEP 2 — WEBSITE ENRICHMENT")
        print("=" * 70)
        crawl_cache = run_crawl(raw_df, crawl_cache_path)
        df_web = df.copy()
        df_web = add_website_features(df_web, raw_df, crawl_cache)
        web_new_cols = [c for c in df_web.columns if c not in df.columns]
        print(f"  Added {len(web_new_cols)} website features: {web_new_cols}")

        web_numeric = numeric_feats + web_new_cols
        all_web_cols = web_numeric + ["category_primary"]
        cat_idx_web  = [all_web_cols.index("category_primary")]

        X_tr_web  = df_web.loc[train_idx, all_web_cols]
        X_te_web  = df_web.loc[test_idx,  all_web_cols]
        X_tr_wnum = df_web.loc[train_idx, web_numeric]
        X_te_wnum = df_web.loc[test_idx,  web_numeric]

        print("\n  Training V5+Web CatBoost-C … ", end="", flush=True)
        cb_web = CatBoostClassifier(**BEST_CB_PARAMS, cat_features=cat_idx_web)
        cb_web.fit(X_tr_web, y_tr)
        cb_prob_web = cb_web.predict_proba(X_te_web)[:, 1]
        print("done")

        print("  Training V5+Web LightGBM-A … ", end="", flush=True)
        lgbm_web = LGBMClassifier(**BEST_LGBM_PARAMS)
        lgbm_web.fit(X_tr_wnum, y_tr)
        lgbm_prob_web = lgbm_web.predict_proba(X_te_wnum)[:, 1]
        print("done")

        ba_web, *_ = eval_ensemble(cb_prob_web, lgbm_prob_web, y_te,
                                   "V5 + Website enrichment")
    else:
        print("\nSkipping website crawling (--no-crawl)")
        df_web = df.copy()
        web_numeric = numeric_feats
        ba_web = None

    # ── Public records enrichment ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — PUBLIC RECORDS ENRICHMENT")
    print("=" * 70)

    df_full, n_rec_feats = add_public_records_features(df_web.copy(), records_path)
    if n_rec_feats == 0:
        print(f"  No public records found at: {records_path}")
        print(f"  To generate: run scripts/archived/public_records/match_records.py")
        print(f"               then scripts/archived/public_records/engineer_license_features.py")
        ba_full = None
    else:
        print(f"  Merged {n_rec_feats} public record features")
        rec_new_cols = [c for c in df_full.columns
                        if c not in df_web.columns and c not in ("id",)]
        full_numeric  = web_numeric + rec_new_cols
        all_full_cols = full_numeric + ["category_primary"]
        cat_idx_full  = [all_full_cols.index("category_primary")]

        X_tr_full  = df_full.loc[train_idx, all_full_cols]
        X_te_full  = df_full.loc[test_idx,  all_full_cols]
        X_tr_fnum  = df_full.loc[train_idx, full_numeric]
        X_te_fnum  = df_full.loc[test_idx,  full_numeric]

        print("\n  Training V5+Web+Records CatBoost-C … ", end="", flush=True)
        cb_full = CatBoostClassifier(**BEST_CB_PARAMS, cat_features=cat_idx_full)
        cb_full.fit(X_tr_full, y_tr)
        cb_prob_full = cb_full.predict_proba(X_te_full)[:, 1]
        print("done")

        print("  Training V5+Web+Records LightGBM-A … ", end="", flush=True)
        lgbm_full = LGBMClassifier(**BEST_LGBM_PARAMS)
        lgbm_full.fit(X_tr_fnum, y_tr)
        lgbm_prob_full = lgbm_full.predict_proba(X_te_fnum)[:, 1]
        print("done")

        ba_full, *_ = eval_ensemble(cb_prob_full, lgbm_prob_full, y_te,
                                    "V5 + Website + Public Records")

        # Feature importance for new features
        m_fi = CatBoostClassifier(**BEST_CB_PARAMS, cat_features=cat_idx_full)
        m_fi.fit(X_tr_full, y_tr)
        fi = pd.Series(m_fi.feature_importances_, index=all_full_cols).sort_values(ascending=False)
        new_feats_sorted = [f for f in fi.index if f in rec_new_cols + (web_numeric[len(numeric_feats):]
                                                                          if not no_crawl else [])]
        if new_feats_sorted:
            print(f"\n  Top new feature importances:")
            for fname in new_feats_sorted[:15]:
                rank = list(fi.index).index(fname) + 1
                print(f"    #{rank:3d}/{len(fi)}  {fname:40s}  {fi[fname]:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    rows = [("V5 baseline", ba_v5)]
    if ba_web is not None:
        rows.append(("V5 + Website crawling", ba_web))
    if ba_full is not None:
        rows.append(("V5 + Website + Public records", ba_full))
    print(f"\n  {'Model':<35}  {'Hold-out BalAcc':>15}  {'Δ vs V5':>8}")
    print(f"  {'─'*35}  {'─'*15}  {'─'*8}")
    for name, ba in rows:
        delta = f"{ba - ba_v5:+.4f}" if ba != ba_v5 else "—"
        flag  = "  ← best" if ba == max(b for _, b in rows) and ba > ba_v5 else ""
        print(f"  {name:<35}  {ba:>15.4f}  {delta:>8}{flag}")

    print(f"\n  Note: public records coverage is NYC + SF only.")
    print(f"        Hold-out cities (Chicago + Miami) will have 0 for license features.")
    print(f"        True coverage test requires hold-out cities with public records.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V6 enrichment experiment: add website crawling and public records\n"
            "on top of the V5 best model, evaluated on Chicago + Miami hold-out.\n\n"
            "Requires: data/combined_truth_dataset_expanded.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i",
                        default="data/combined_truth_dataset_expanded.parquet")
    parser.add_argument("--records",
                        default="data/public_records/license_features.parquet",
                        help="Pre-fetched public records features parquet")
    parser.add_argument("--crawl-cache",
                        default="data/website_crawl_cache.json",
                        help="Path to website crawl cache (created on first run)")
    parser.add_argument("--no-crawl", action="store_true",
                        help="Skip website crawling, only use public records enrichment")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌  Dataset not found: {args.input}")
        print(f"    Run from the project root (StatusNow/).")
        sys.exit(1)

    main(args.input, args.records, args.crawl_cache, args.no_crawl)
