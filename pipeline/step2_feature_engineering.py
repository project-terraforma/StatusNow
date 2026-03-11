"""
Step 2 — Feature Engineering (Multi-Release Aware, Leak-Free)
=============================================================

What this script does
---------------------
Reads the raw labelled training data produced by step1 and engineers features
for the CB+LGBM ensemble model.

Feature sets
------------

All feature sets from V5 are preserved (leak-free):
  • Confidence       — base_confidence ONLY (no delta_confidence leakage)
  • Digital presence — websites, socials, phones, emails counts & flags
  • Brand
  • Delta features   — change between base and current snapshot
  • Identity changes — name, category, website domain, address changes
  • Recency          — staleness flags, log_days, recency_bucket, PCA
  • Interactions     — zombie_score, decay_velocity, brand×stale, etc.

Multi-release trajectory features (automatically activated when 3+ releases):
  • releases_seen        — how many releases this place appeared in (before closure)
  • consecutive_present  — longest run of consecutive releases with the place present
  • pre_closure_loss     — had ANY digital loss in the PREVIOUS comparison window
  • social_trend         — slope of social-count across all windows (linear regression)
  • website_trend        — slope of website-count across all windows

Why these matter
----------------
In a 2-release dataset, churned (closed) places all have delta=0 by construction
(COALESCE makes current = previous for disappeared places). With 3+ releases,
we can see the place's behaviour *before* disappearance — these trajectory
features are the true leading closure indicators.

Output
------
  pipeline_output/02_features.parquet

Usage
-----
  python pipeline/step2_feature_engineering.py [options]

  Options:
    --input    FILE   Raw training data (default: pipeline_output/01_training_data_raw.parquet)
    --output   FILE   Feature dataset path (default: pipeline_output/02_features.parquet)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ─── parsing helpers (identical to process_data_v5.py) ───────────────────────

def _parse(x):
    """Parse JSON string / dict / list → Python object, or None on failure."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def _len(x):
    """Count elements in a JSON-encoded list."""
    v = _parse(x)
    return len(v) if isinstance(v, list) else 0


def _primary(x, key="primary"):
    """Extract the primary value from a JSON-encoded categories/names object."""
    v = _parse(x)
    if isinstance(v, dict):
        return v.get(key, "").lower().strip() if key == "primary" else v.get(key)
    return None


def _recency(x, reference_date: datetime):
    """
    Return (min_days, max_days, avg_days) since the most/least recent source update.
    Returns (9999, 9999, 9999) when no valid update_time is found.
    """
    data = _parse(x)
    if not isinstance(data, list) or not data:
        return (9999, 9999, 9999)
    days = []
    for item in data:
        if isinstance(item, dict):
            ut = item.get("update_time")
            if ut:
                try:
                    dt = datetime.strptime(ut.split("T")[0], "%Y-%m-%d")
                    days.append((reference_date - dt).days)
                except Exception:
                    pass
    if not days:
        return (9999, 9999, 9999)
    return (min(days), max(days), sum(days) / len(days))


def _source_names(x):
    data = _parse(x)
    if isinstance(data, list):
        return [str(i.get("dataset", "")).lower() for i in data if isinstance(i, dict)]
    return []


def _website_domain(x):
    data = _parse(x)
    if isinstance(data, list) and data:
        url = str(data[0]).replace("http://", "").replace("https://", "").replace("www.", "")
        return url.split("/")[0].split("?")[0].lower()
    return None


def _address_key(x):
    data = _parse(x)
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            return f"{item.get('locality','')}_{item.get('postcode','')}".lower()
    return None


def _email_count(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return int(x)
    return _len(x)


def _congruence(row):
    """1 if website domain appears in any social URL."""
    w = _parse(row["websites"])
    s = _parse(row["socials"])
    if not w or not s or not isinstance(w, list) or not isinstance(s, list):
        return 0
    domain = (
        str(w[0]).replace("http://", "").replace("https://", "").replace("www.", "")
        .split("/")[0].split(".")[0].lower()
    )
    return 1 if any(isinstance(x, str) and domain in x.lower() for x in s) else 0


def _has_platform(x, platform: str):
    data = _parse(x)
    if isinstance(data, list):
        return 1 if any(isinstance(s, str) and platform in s.lower() for s in data) else 0
    return 0


# ─── trajectory feature helpers ──────────────────────────────────────────────

def _compute_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-window trajectory features.

    These require at least 2 release pairs (3 releases) to be meaningful.
    For places with only 1 window of data the features default to 0 / NaN-filled 0.

    Features produced
    -----------------
    releases_seen         int   — number of releases this id appeared in
    consecutive_present   int   — longest consecutive run of appearances
    pre_closure_loss      int   — 1 if there was any digital loss in the window
                                  BEFORE the current (most recent) window
    social_trend          float — slope of social count over time (units: count / window)
    website_trend         float — slope of website count over time
    """
    print("  trajectory features (multi-release) …")

    # Sort to ensure chronological order within each place
    df = df.sort_values(["id", "release_index"]).copy()

    # Build a pivot of social/website counts per (id, release_index)
    # For open places, current counts come from the current snapshot.
    # For churned closed places, current snapshot equals base (COALESCE),
    # so we use base counts as the "pre-closure" reading.
    df["_n_social_curr"] = df["socials"].apply(_len)
    df["_n_web_curr"]    = df["websites"].apply(_len)
    df["_n_social_base"] = df["base_socials"].apply(_len)
    df["_n_web_base"]    = df["base_websites"].apply(_len)
    df["_has_any_loss"]  = (
        (df["socials"].apply(_len) < df["base_socials"].apply(_len)) |
        (df["websites"].apply(_len) < df["base_websites"].apply(_len)) |
        (df["phones"].apply(_len) < df["base_phones"].apply(_len))
    ).astype(int)

    # Number of windows each place appeared in
    releases_seen = df.groupby("id")["release_index"].nunique().rename("releases_seen")
    df = df.join(releases_seen, on="id")

    # Longest consecutive run of release indices for each place
    def _longest_run(indices):
        sorted_idx = sorted(set(indices))
        if not sorted_idx:
            return 0
        best = cur = 1
        for a, b in zip(sorted_idx, sorted_idx[1:]):
            if b - a == 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    consec = (
        df.groupby("id")["release_index"]
        .apply(lambda s: _longest_run(s.tolist()))
        .rename("consecutive_present")
    )
    df = df.join(consec, on="id")

    # pre_closure_loss: did this place have a digital loss in the window BEFORE
    # the current (most recent) window it appears in?
    # We look up the loss flag from the previous release_index for each place.
    loss_lookup = df.set_index(["id", "release_index"])["_has_any_loss"]

    def _pre_loss(row):
        prev_idx = row["release_index"] - 1
        if prev_idx < 0:
            return 0
        try:
            return int(loss_lookup.loc[(row["id"], prev_idx)])
        except KeyError:
            return 0

    df["pre_closure_loss"] = df.apply(_pre_loss, axis=1)

    # Trend features: slope of social / website count over windows, per place
    # Uses base counts (available for all places including churned ones)
    def _slope(values):
        """Linear regression slope of a list of values."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        x -= x.mean()
        denom = (x ** 2).sum()
        if denom == 0:
            return 0.0
        return float((x * y).sum() / denom)

    social_counts  = df.groupby("id")["_n_social_base"].apply(list)
    website_counts = df.groupby("id")["_n_web_base"].apply(list)

    social_trend_map  = social_counts.apply(_slope).rename("social_trend")
    website_trend_map = website_counts.apply(_slope).rename("website_trend")

    df = df.join(social_trend_map, on="id")
    df = df.join(website_trend_map, on="id")

    # Clean up temp columns
    df.drop(
        columns=["_n_social_curr", "_n_web_curr", "_n_social_base",
                 "_n_web_base", "_has_any_loss"],
        inplace=True,
        errors="ignore",
    )

    return df


# ─── main feature engineering function ───────────────────────────────────────

def build_features(
    input_path:  str,
    output_path: str,
    reference_date: datetime | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the full feature set from the raw training data.

    Parameters
    ----------
    input_path      : path to 01_training_data_raw.parquet
    output_path     : where to write 02_features.parquet
    reference_date  : the "current" date used for staleness calculations.
                      Defaults to the date embedded in the newest release tag,
                      or today if it cannot be inferred.

    Returns
    -------
    (df_features, numeric_feature_names)
    """
    print("\n" + "=" * 70)
    print("STEP 2 — FEATURE ENGINEERING (Multi-Release, Leak-Free)")
    print("=" * 70)

    df = pd.read_parquet(input_path)
    print(f"\n  Loaded {len(df):,} rows | {df.shape[1]} columns")
    print(f"  Release pairs present: {sorted(df['release_pair'].unique())}")

    # ── Infer reference date ─────────────────────────────────────────────────
    if reference_date is None:
        # Use the most recent release_date_current in the data
        try:
            latest_tag = df["release_date_current"].max()  # e.g. "2026-02-18.0"
            reference_date = datetime.strptime(latest_tag.split(".")[0], "%Y-%m-%d")
            print(f"  Reference date (inferred from releases): {reference_date.date()}")
        except Exception:
            reference_date = datetime.today()
            print(f"  Reference date (fallback to today): {reference_date.date()}")

    n_pairs = df["release_pair"].nunique()
    has_trajectory = n_pairs >= 2  # Need 3+ releases → 2+ pairs

    # ── Multi-release trajectory features ────────────────────────────────────
    if has_trajectory:
        print(f"\n  3+ releases detected ({n_pairs} pairs) — computing trajectory features …")
        df = _compute_trajectory_features(df)
    else:
        print(f"\n  2 releases detected (1 pair) — trajectory features will be 0")
        df["releases_seen"]       = 1
        df["consecutive_present"] = 1
        df["pre_closure_loss"]    = 0
        df["social_trend"]        = 0.0
        df["website_trend"]       = 0.0

    # ── 1. SOURCES / RECENCY ─────────────────────────────────────────────────
    print("\n  sources & recency …")
    df["num_sources"]      = df["sources"].apply(_len)
    df["base_num_sources"] = df["base_sources"].apply(_len)
    df["delta_sources"]    = df["num_sources"] - df["base_num_sources"]
    df["has_lost_sources"] = (df["delta_sources"] < 0).astype(int)
    df["source_list"]      = df["sources"].apply(_source_names)
    df["source_has_msft"]  = df["source_list"].apply(
        lambda x: 1 if any(s in x for s in ("microsoft", "msft")) else 0
    )
    df["is_cross_verified"] = (df["num_sources"] > 1).astype(int)
    df["log_num_sources"]   = np.log1p(df["num_sources"])

    rec = df["sources"].apply(lambda x: _recency(x, reference_date))
    df["days_latest"] = rec.apply(lambda x: x[0])
    df["days_oldest"] = rec.apply(lambda x: x[1])
    df["days_avg"]    = rec.apply(lambda x: x[2])

    # ── 2. DIGITAL PRESENCE ──────────────────────────────────────────────────
    print("  digital presence …")
    df["num_websites"] = df["websites"].apply(_len)
    df["num_socials"]  = df["socials"].apply(_len)
    df["num_phones"]   = df["phones"].apply(_len)
    df["has_website"]  = (df["num_websites"] > 0).astype(int)
    df["has_social"]   = (df["num_socials"]  > 0).astype(int)
    df["has_phone"]    = (df["num_phones"]   > 0).astype(int)

    df["has_facebook"]  = df["socials"].apply(lambda x: _has_platform(x, "facebook.com"))
    df["has_instagram"] = df["socials"].apply(lambda x: _has_platform(x, "instagram.com"))
    df["has_yelp"]      = df["socials"].apply(lambda x: _has_platform(x, "yelp.com"))

    df["num_emails"]   = df["emails"].apply(_email_count)
    df["contact_depth"] = df["num_websites"] + df["num_socials"] + df["num_emails"]
    df["total_digital"] = (
        df["has_website"] + df["has_social"] + df["has_phone"] +
        df["has_facebook"] + df["has_instagram"] + df["has_yelp"]
    )

    # ── 3. BRAND ─────────────────────────────────────────────────────────────
    df["is_brand"] = df["brand"].apply(
        lambda x: 0 if (
            x is None or
            (isinstance(x, float) and np.isnan(x)) or
            str(x).strip() in ("", "null", "[]")
        ) else 1
    )

    # ── 4. CONFIDENCE — base_confidence ONLY (V5 leak fix) ───────────────────
    print("  confidence (base only — leak-free) …")
    df["base_conf"]     = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    df["base_conf_sq"]  = df["base_conf"] ** 2
    # Will be overwritten after staleness is computed
    df["base_conf_x_stale"] = df["base_conf"]

    # ── 5. CATEGORIES ────────────────────────────────────────────────────────
    print("  categories …")
    df["category_primary"] = df["categories"].apply(_primary)
    df["category_primary"] = df["category_primary"].fillna("unknown")
    df["cat_is_unknown"]   = (df["category_primary"] == "unknown").astype(int)

    # ── 6. DELTA FEATURES ────────────────────────────────────────────────────
    print("  delta features …")
    # Note: for churned (closed) places in 2-release data, COALESCE makes
    # current == base, so all deltas are 0 by construction. trajectory features
    # (pre_closure_loss, social_trend) address this limitation for 3+ releases.
    for cur, base in [
        ("websites", "base_websites"),
        ("socials",  "base_socials"),
        ("phones",   "base_phones"),
    ]:
        df[f"delta_{cur}"] = df[cur].apply(_len) - df[base].apply(_len)

    df["has_lost_website"]  = (df["delta_websites"] < 0).astype(int)
    df["has_gained_website"] = (df["delta_websites"] > 0).astype(int)
    df["has_lost_social"]   = (df["delta_socials"]  < 0).astype(int)
    df["has_gained_social"] = (df["delta_socials"]  > 0).astype(int)
    df["has_lost_phone"]    = (df["delta_phones"]   < 0).astype(int)
    df["has_gained_phone"]  = (df["delta_phones"]   > 0).astype(int)
    df["delta_total"]       = df["delta_websites"] + df["delta_socials"] + df["delta_phones"]
    df["has_any_loss"]      = (
        (df["delta_websites"] < 0) | (df["delta_socials"] < 0) | (df["delta_phones"] < 0)
    ).astype(int)
    df["has_any_gain"]      = (
        (df["delta_websites"] > 0) | (df["delta_socials"] > 0) | (df["delta_phones"] > 0)
    ).astype(int)
    df["num_loss_types"]    = df["has_lost_website"] + df["has_lost_social"] + df["has_lost_phone"]
    df["num_gain_types"]    = df["has_gained_website"] + df["has_gained_social"] + df["has_gained_phone"]
    df["has_complete_loss"] = (
        (df["delta_websites"] < 0) & (df["delta_socials"] < 0)
    ).astype(int)
    df["contact_loss_severity"] = (
        df["has_lost_website"] * 2 +
        df["has_lost_social"]  * 1.5 +
        df["has_lost_phone"]   * 1
    )

    # ── 7. IDENTITY CHANGES ──────────────────────────────────────────────────
    print("  identity changes …")
    df["curr_name"] = df["names"].apply(_primary)
    df["base_name"] = df["base_names"].apply(_primary)
    df["name_changed"] = (
        (df["curr_name"] != df["base_name"]) &
        df["curr_name"].notna() & df["base_name"].notna()
    ).astype(int)
    df["name_length"]       = df["curr_name"].apply(lambda x: len(x) if x else 0)
    df["base_name_length"]  = df["base_name"].apply(lambda x: len(x) if x else 0)
    df["name_length_delta"] = df["name_length"] - df["base_name_length"]

    df["base_cat"] = df["base_categories"].apply(_primary)
    df["cat_changed"] = (
        (df["category_primary"] != df["base_cat"]) &
        df["category_primary"].notna() & df["base_cat"].notna() &
        (df["category_primary"] != "unknown") & (df["base_cat"] != "unknown")
    ).astype(int)

    df["curr_domain"] = df["websites"].apply(_website_domain)
    df["base_domain"] = df["base_websites"].apply(_website_domain)
    df["website_domain_changed"] = (
        (df["curr_domain"] != df["base_domain"]) &
        df["curr_domain"].notna() & df["base_domain"].notna()
    ).astype(int)

    df["curr_addr"] = df["addresses"].apply(_address_key)
    df["base_addr"] = df["base_addresses"].apply(_address_key)
    df["address_changed"] = (
        (df["curr_addr"] != df["base_addr"]) &
        df["curr_addr"].notna() & df["base_addr"].notna()
    ).astype(int)

    df["identity_change_score"] = (
        df["name_changed"] + df["cat_changed"] + df["address_changed"]
    )

    # ── 8. RECENCY FEATURES ──────────────────────────────────────────────────
    print("  recency / staleness …")
    days = df["days_latest"].clip(upper=9999)
    df["log_days"]       = np.log1p(days)
    df["is_stale_3mo"]   = (days > 90).astype(int)
    df["is_stale_6mo"]   = (days > 180).astype(int)
    df["is_stale_1yr"]   = (days > 365).astype(int)
    df["is_stale_2yr"]   = (days > 730).astype(int)
    df["recency_bucket"] = pd.cut(
        days, bins=[-1, 90, 365, 730, 99999], labels=[0, 1, 2, 3]
    ).astype(int)
    df["recency_spread"] = (df["days_oldest"] - df["days_latest"]).clip(lower=0)

    rec_mat = df[["days_latest", "days_avg"]].fillna(9999)
    pca = PCA(n_components=1)
    df["recency_pca"] = pca.fit_transform(rec_mat).flatten()

    # Overwrite now that staleness flags exist
    df["base_conf_x_stale"] = df["base_conf"] * df["is_stale_1yr"]

    # ── 9. INTERACTION FEATURES ──────────────────────────────────────────────
    print("  interaction features …")
    df["zombie_score"]          = df["num_sources"] / (df["days_avg"] + 1)
    df["decay_velocity"]        = df["delta_total"] / (df["days_avg"] + 1)
    df["recency_x_loss"]        = days * df["has_any_loss"]
    df["recency_x_social_loss"] = days * df["has_lost_social"]
    df["brand_x_stale"]         = df["is_brand"] * df["is_stale_1yr"]
    df["nonbrand_stale_risk"]   = (1 - df["is_brand"]) * df["is_stale_6mo"]
    df["brand_x_name_change"]   = df["is_brand"] * df["name_changed"]
    df["nonbrand_x_name_change"]= (1 - df["is_brand"]) * df["name_changed"]
    df["recency_x_name_change"] = days * df["name_changed"]
    df["recency_x_cat_change"]  = days * df["cat_changed"]
    df["source_loss_x_stale"]   = df["has_lost_sources"] * df["is_stale_6mo"]
    df["stale_x_loss_x_nonbrand"] = (
        df["is_stale_6mo"] * df["has_any_loss"] * (1 - df["is_brand"])
    )
    df["loss_x_low_conf"]  = df["has_any_loss"]  * (df["base_conf"] < 0.5).astype(int)
    df["stale_x_low_conf"] = df["is_stale_1yr"]  * (df["base_conf"] < 0.5).astype(int)
    df["multi_signal_risk"] = (
        df["has_any_loss"] + df["is_stale_1yr"] +
        df["name_changed"] + df["cat_changed"]
    )
    df["digital_congruence"] = df.apply(_congruence, axis=1)

    # Multi-release interaction: pre-closure loss amplified by staleness
    df["pre_loss_x_stale"]    = df["pre_closure_loss"] * df["is_stale_6mo"]
    df["trajectory_risk"]     = df["pre_closure_loss"] + df["has_any_loss"]
    df["seen_then_lost"]      = (df["releases_seen"] > 1).astype(int) * df["has_any_loss"]

    # ── 10. ASSEMBLE ─────────────────────────────────────────────────────────
    print("  assembling final feature matrix …")

    numeric_features = [
        # Confidence (base only — leak-free)
        "base_conf", "base_conf_sq", "base_conf_x_stale",
        # Brand & sources
        "is_brand", "num_sources", "log_num_sources",
        "source_has_msft", "is_cross_verified",
        # Digital presence
        "has_website", "has_social", "has_phone", "contact_depth",
        "has_facebook", "has_instagram", "has_yelp",
        "total_digital", "num_websites", "num_socials",
        # Category
        "cat_is_unknown",
        # Delta features
        "delta_websites", "delta_socials", "delta_phones",
        "delta_total", "delta_sources",
        "has_gained_social", "has_lost_social",
        "has_any_loss", "has_any_gain",
        "has_lost_website", "has_gained_website",
        "has_lost_phone", "has_gained_phone",
        "has_complete_loss", "num_loss_types", "num_gain_types",
        "contact_loss_severity",
        # Recency
        "recency_pca", "log_days", "is_stale_3mo", "is_stale_6mo",
        "is_stale_1yr", "is_stale_2yr", "recency_bucket", "recency_spread",
        # Interactions
        "zombie_score", "decay_velocity",
        "recency_x_loss", "recency_x_social_loss",
        "digital_congruence",
        "brand_x_stale", "nonbrand_stale_risk",
        "brand_x_name_change", "nonbrand_x_name_change",
        "recency_x_name_change", "recency_x_cat_change",
        "source_loss_x_stale", "stale_x_loss_x_nonbrand",
        "loss_x_low_conf", "stale_x_low_conf",
        # Identity changes
        "name_changed", "cat_changed",
        "website_domain_changed", "address_changed",
        "identity_change_score", "name_length", "name_length_delta",
        "has_lost_sources", "multi_signal_risk",
        # ── Multi-release trajectory features ──
        "releases_seen", "consecutive_present",
        "pre_closure_loss", "social_trend", "website_trend",
        "pre_loss_x_stale", "trajectory_risk", "seen_then_lost",
    ]

    # Only keep features that were actually computed
    numeric_features = [f for f in numeric_features if f in df.columns]

    # Passthrough columns: category_primary (CatBoost native cat encoding),
    # metadata for hold-out splitting and inspection
    keep_cols = (
        numeric_features +
        ["category_primary", "release_pair", "release_date_base",
         "release_date_current", "release_index", "label"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]

    final = df[keep_cols].copy()
    final = final.dropna(subset=["label"])
    final.rename(columns={"label": "open"}, inplace=True)
    final["open"] = final["open"].astype(int)

    # Fill remaining NaN in numeric features with 0
    for col in numeric_features:
        if col in final.columns and final[col].isna().any():
            final[col] = final[col].fillna(0)

    final["category_primary"] = final["category_primary"].astype(str)

    print(f"\n  Shape:             {final.shape}")
    print(f"  Numeric features:  {len(numeric_features)}")
    print(f"  Open rate:         {final['open'].mean():.1%}")
    print(f"  Trajectory feats:  {'activated ✓' if has_trajectory else 'minimal (need 3+ releases)'}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.to_parquet(output_path, index=False)
    print(f"\n  ✅  Saved → {output_path}")

    return final, numeric_features


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: Engineer features from the raw training data.\n"
            "Run after step1_build_training_data.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="pipeline_output/01_training_data_raw.parquet",
        help="Input raw training data (default: pipeline_output/01_training_data_raw.parquet)",
    )
    parser.add_argument(
        "--output", "-o",
        default="pipeline_output/02_features.parquet",
        help="Output feature dataset (default: pipeline_output/02_features.parquet)",
    )
    args = parser.parse_args()

    build_features(input_path=args.input, output_path=args.output)
