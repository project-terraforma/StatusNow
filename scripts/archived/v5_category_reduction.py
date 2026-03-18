"""
V5 Misclassification Reduction — Built ON TOP of V3 Processed Dataset

Instead of rebuilding features from raw data, this script:
1. Loads the existing V3 processed dataset (52 features, 92.2% BalAcc baseline)
2. ADDS category-specific interaction features and liveness signals
3. Retrains CatBoost with the enriched feature set
4. Applies per-category threshold tuning as a post-processing step
5. Produces a full V3 vs V5 comparison report with charts
"""

import pandas as pd
import numpy as np
import time
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
V3_PATH = "data/processed_for_ml_testing.parquet"
OUT_DIR = "/Users/anthonylamas/.gemini/antigravity/brain/6dcd71ed-bf93-479d-b5a1-6ed4db48813c"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

HIGH_ERR_CATS = {
    "Pharmacy": "cat_pharmacy",
    "Coffee Shop": "cat_coffee_shop",
    "Hotel": "cat_hotel",
    "ATMs": "cat_atms",
    "Restaurant": "cat_restaurant",
    "Pizza Restaurant": "cat_pizza_restaurant",
}

ALL_DISPLAY_CATS = [
    ("Pharmacy", "cat_pharmacy"),
    ("Coffee Shop", "cat_coffee_shop"),
    ("Hotel", "cat_hotel"),
    ("ATMs", "cat_atms"),
    ("Restaurant", "cat_restaurant"),
    ("Pizza Restaurant", "cat_pizza_restaurant"),
    ("Other", "cat_other"),
    ("Package Locker", "cat_package_locker"),
    ("Corporate Office", "cat_corporate_office"),
    ("Art Gallery", "cat_art_gallery"),
    ("Dentist", "cat_dentist"),
    ("Doctor", "cat_doctor"),
    ("Hair Salon", "cat_hair_salon"),
    ("Beauty Salon", "cat_beauty_salon"),
    ("Clothing Store", "cat_clothing_store"),
    ("Jewelry Store", "cat_jewelry_store"),
    ("Church Cathedral", "cat_church_cathedral"),
    ("Community Services", "cat_community_services_non_profits"),
    ("Professional Services", "cat_professional_services"),
    ("Real Estate Agent", "cat_real_estate_agent"),
    ("Landmark", "cat_landmark_and_historical_building"),
]


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY IMPLEMENTATION: Add category-specific features to V3
# ════════════════════════════════════════════════════════════════════════════
def add_v5_features(df):
    """
    Add category-specific features ON TOP of the V3 processed dataset.
    These implement the strategies from the misclassification analysis.
    """
    df = df.copy()
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c != "cat_is_unknown"]

    # ── Identify high-error category membership ──────────────────────────
    food_cats = ["cat_restaurant", "cat_coffee_shop", "cat_pizza_restaurant"]
    is_food = df[[c for c in food_cats if c in df.columns]].max(axis=1) if any(c in df.columns for c in food_cats) else pd.Series(0, index=df.index)
    is_pharmacy = df.get("cat_pharmacy", pd.Series(0, index=df.index))
    is_hotel = df.get("cat_hotel", pd.Series(0, index=df.index))
    is_atm = df.get("cat_atms", pd.Series(0, index=df.index))

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Category-Specific Liveness Scores
    # These proxy what external API calls (DoorDash, Booking.com, etc.) would return
    # using the digital signals already in the dataset.
    # ══════════════════════════════════════════════════════════════════════

    # Food service liveness (proxies: delivery platform, health dept, online menu)
    df["food_liveness"] = (
        df.get("has_social", 0) * 2 +       # Social = proxy for delivery platform
        df.get("has_website", 0) * 1.5 +     # Website = proxy for online menu
        df.get("has_phone", 0) * 1 +         # Contactable
        df.get("is_cross_verified", 0) * 1.5 +# Multiple sources
        (1 - df.get("is_stale_6mo", 0)) * 2  # Fresh data
    ) * is_food

    # Pharmacy liveness (proxies: state license, GoodRx, vaccine scheduling)
    df["pharmacy_liveness"] = (
        df.get("is_brand", 0) * 3 +           # Brand pharmacy (CVS, Walgreens) = almost certainly open
        df.get("has_website", 0) * 2 +         # Website = proxy for online prescription portal
        df.get("has_phone", 0) * 1.5 +         # Active phone line
        df.get("is_cross_verified", 0) * 2 +   # Multiple sources confirm
        (1 - df.get("is_stale_6mo", 0)) * 2
    ) * is_pharmacy

    # Hotel liveness (proxies: OTA booking, TripAdvisor, health inspection)
    df["hotel_liveness"] = (
        df.get("has_website", 0) * 3 +         # Website = proxy for booking/OTA link
        df.get("has_phone", 0) * 2 +           # Bookable by phone
        df.get("has_social", 0) * 1.5 +        # Marketing active
        df.get("is_cross_verified", 0) * 2 +   # Listed in multiple directories
        (1 - df.get("is_stale_6mo", 0)) * 2
    ) * is_hotel

    # ATM liveness (proxies: bank ATM locator, Google business_status)
    df["atm_liveness"] = (
        df.get("is_brand", 0) * 4 +            # Bank-operated = tracked by bank
        df.get("is_cross_verified", 0) * 3 +   # Multiple sources verify
        df.get("source_has_msft", 0) * 2 +     # Microsoft/Bing verification
        (1 - df.get("is_stale_1yr", 0)) * 2    # ATMs tolerate longer gaps
    ) * is_atm

    # Composite liveness
    is_high_err = is_food | is_pharmacy | is_hotel | is_atm
    df["total_liveness"] = df["food_liveness"] + df["pharmacy_liveness"] + df["hotel_liveness"] + df["atm_liveness"]
    df["generic_liveness"] = (
        df.get("has_website", 0) + df.get("has_social", 0) + df.get("has_phone", 0) +
        df.get("is_cross_verified", 0) + (1 - df.get("is_stale_6mo", 0))
    ) * (1 - is_high_err)
    df["total_liveness"] = df["total_liveness"] + df["generic_liveness"]

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 2: Category × Key Feature Interactions
    # Captures that the SAME signal means different things per category
    # ══════════════════════════════════════════════════════════════════════

    # Non-brand × high-error category (core problem: non-brand places get false-closed)
    df["nonbrand_food"] = (1 - df.get("is_brand", 0)) * is_food
    df["nonbrand_pharmacy"] = (1 - df.get("is_brand", 0)) * is_pharmacy
    df["nonbrand_hotel"] = (1 - df.get("is_brand", 0)) * is_hotel
    df["nonbrand_atm"] = (1 - df.get("is_brand", 0)) * is_atm

    # Staleness × category (different staleness tolerances per category)
    df["stale_food"] = df.get("is_stale_6mo", 0) * is_food
    df["stale_pharmacy"] = df.get("is_stale_6mo", 0) * is_pharmacy
    df["stale_hotel"] = df.get("is_stale_1yr", 0) * is_hotel      # Hotels tolerate longer gaps
    df["stale_atm"] = df.get("is_stale_1yr", 0) * is_atm          # ATMs are stable infrastructure

    # Contact depth × category (high contact = different meaning per category)
    df["contact_food"] = df.get("contact_depth", 0) * is_food
    df["contact_hotel"] = df.get("contact_depth", 0) * is_hotel
    df["contact_pharmacy"] = df.get("contact_depth", 0) * is_pharmacy

    # Decay velocity × category
    df["decay_food"] = df.get("decay_velocity", 0) * is_food
    df["decay_hotel"] = df.get("decay_velocity", 0) * is_hotel

    # Confidence × category (what confidence score means differs by type)
    df["conf_food"] = df.get("confidence", 0) * is_food
    df["conf_hotel"] = df.get("confidence", 0) * is_hotel
    df["conf_atm"] = df.get("confidence", 0) * is_atm

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 3: Verification Gap per Category
    # High-error categories need MORE cross-verification to be trusted
    # ══════════════════════════════════════════════════════════════════════
    df["verification_gap_food"] = (2 - df.get("num_sources", 0)).clip(lower=0) * is_food
    df["verification_gap_hotel"] = (2 - df.get("num_sources", 0)).clip(lower=0) * is_hotel
    df["verification_gap_pharmacy"] = (2 - df.get("num_sources", 0)).clip(lower=0) * is_pharmacy

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 4: "Transition Zone" Detector
    # Fresh data + rapid decay = the ambiguous zone where model fails most
    # ══════════════════════════════════════════════════════════════════════
    is_fresh = (df.get("log_days_since_update", 10) < np.log1p(180)).astype(int)
    has_rapid_decay = (df.get("decay_velocity", 0) < df.get("decay_velocity", pd.Series(0, index=df.index)).quantile(0.1)).astype(int)
    df["transition_zone"] = is_fresh * has_rapid_decay

    df["transition_food"] = df["transition_zone"] * is_food
    df["transition_hotel"] = df["transition_zone"] * is_hotel
    df["transition_pharmacy"] = df["transition_zone"] * is_pharmacy

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 5: Recency-adjusted category risk
    # Blend staleness risk with category-specific churn rate
    # ══════════════════════════════════════════════════════════════════════
    df["churn_x_fresh_food"] = df.get("category_churn_risk", 0) * (1 - df.get("is_stale_6mo", 0)) * is_food
    df["churn_x_fresh_hotel"] = df.get("category_churn_risk", 0) * (1 - df.get("is_stale_6mo", 0)) * is_hotel
    df["churn_x_fresh_pharmacy"] = df.get("category_churn_risk", 0) * (1 - df.get("is_stale_6mo", 0)) * is_pharmacy

    return df


# ════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def run_full_evaluation():
    print("=" * 80)
    print("V5 EVALUATION — Category-Specific Feature Enrichment")
    print("=" * 80)

    # ── 1. Load V3 baseline ───────────────────────────────────────────────
    print("\n▶ STEP 1: V3 BASELINE")
    df_v3 = pd.read_parquet(V3_PATH)
    X_v3 = df_v3.drop(columns=["open"])
    y_v3 = df_v3["open"]
    print(f"  V3: {X_v3.shape[0]} samples, {X_v3.shape[1]} features")

    model_v3 = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    print("  Training V3 CatBoost (5-fold CV) …")
    t0 = time.time()
    v3_preds = cross_val_predict(model_v3, X_v3, y_v3, cv=CV, n_jobs=-1)
    v3_proba = cross_val_predict(model_v3, X_v3, y_v3, cv=CV, method="predict_proba", n_jobs=-1)
    v3_time = time.time() - t0
    v3_ba = balanced_accuracy_score(y_v3, v3_preds)
    v3_acc = (v3_preds == y_v3).mean()
    print(f"  V3 BalAcc: {v3_ba:.4f} | Accuracy: {v3_acc:.4f} | Time: {v3_time:.1f}s")

    # ── 2. Build V5 = V3 + category-specific features ────────────────────
    print("\n▶ STEP 2: ADDING V5 CATEGORY-SPECIFIC FEATURES")
    df_v5 = add_v5_features(df_v3)
    X_v5 = df_v5.drop(columns=["open"])
    y_v5 = df_v5["open"]

    new_features = [c for c in X_v5.columns if c not in X_v3.columns]
    print(f"  V5: {X_v5.shape[0]} samples, {X_v5.shape[1]} features (+{len(new_features)} new)")
    print(f"  New features: {new_features}")

    # ── 3. Train V5 model (same CatBoost config) ─────────────────────────
    print("\n▶ STEP 3: V5 MODEL — SAME CONFIG (iterations=500, depth=6)")
    model_v5_same = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    print("  Training V5 (same config) …")
    t0 = time.time()
    v5_preds = cross_val_predict(model_v5_same, X_v5, y_v5, cv=CV, n_jobs=-1)
    v5_proba = cross_val_predict(model_v5_same, X_v5, y_v5, cv=CV, method="predict_proba", n_jobs=-1)
    v5_time = time.time() - t0
    v5_ba = balanced_accuracy_score(y_v5, v5_preds)
    v5_acc = (v5_preds == y_v5).mean()
    print(f"  V5 BalAcc: {v5_ba:.4f} | Accuracy: {v5_acc:.4f} | Time: {v5_time:.1f}s")
    print(f"  Δ from V3: BalAcc {v5_ba - v3_ba:+.4f}, Acc {v5_acc - v3_acc:+.4f}")

    # ── 4. Train V5 with deeper model ────────────────────────────────────
    print("\n▶ STEP 4: V5 MODEL — DEEPER (iterations=800, depth=7)")
    model_v5_deep = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=7,
        verbose=0, auto_class_weights="Balanced",
        random_state=42, allow_writing_files=False,
    )
    print("  Training V5 (deeper) …")
    t0 = time.time()
    v5d_preds = cross_val_predict(model_v5_deep, X_v5, y_v5, cv=CV, n_jobs=-1)
    v5d_proba = cross_val_predict(model_v5_deep, X_v5, y_v5, cv=CV, method="predict_proba", n_jobs=-1)
    v5d_time = time.time() - t0
    v5d_ba = balanced_accuracy_score(y_v5, v5d_preds)
    v5d_acc = (v5d_preds == y_v5).mean()
    print(f"  V5 Deep BalAcc: {v5d_ba:.4f} | Accuracy: {v5d_acc:.4f} | Time: {v5d_time:.1f}s")
    print(f"  Δ from V3: BalAcc {v5d_ba - v3_ba:+.4f}, Acc {v5d_acc - v3_acc:+.4f}")

    # Pick the better V5 variant
    if v5d_ba >= v5_ba:
        best_v5_preds = v5d_preds
        best_v5_proba = v5d_proba
        best_v5_ba = v5d_ba
        best_v5_acc = v5d_acc
        best_v5_label = "V5 Deep (iter=800, depth=7)"
    else:
        best_v5_preds = v5_preds
        best_v5_proba = v5_proba
        best_v5_ba = v5_ba
        best_v5_acc = v5_acc
        best_v5_label = "V5 Same Config (iter=500, depth=6)"
    print(f"\n  Best V5: {best_v5_label}")

    # ── 5. Category-Aware Threshold Tuning ────────────────────────────────
    print("\n▶ STEP 5: CATEGORY-AWARE THRESHOLD TUNING")
    v5_corrected = best_v5_preds.copy()
    threshold_results = {}

    for cat_name, cat_col in HIGH_ERR_CATS.items():
        if cat_col not in X_v5.columns:
            continue
        mask = X_v5[cat_col] == 1
        if mask.sum() < 20:
            continue

        cat_proba = best_v5_proba[mask, 1]
        cat_y = y_v5[mask]
        best_thresh, best_ba_cat = 0.5, balanced_accuracy_score(cat_y, best_v5_preds[mask])
        orig_ba = best_ba_cat

        for t in np.arange(0.20, 0.80, 0.01):
            cat_preds_t = (cat_proba >= t).astype(int)
            ba_t = balanced_accuracy_score(cat_y, cat_preds_t)
            if ba_t > best_ba_cat:
                best_ba_cat = ba_t
                best_thresh = t

        tuned_preds = (best_v5_proba[mask, 1] >= best_thresh).astype(int)
        v5_corrected[mask.values] = tuned_preds

        threshold_results[cat_name] = {
            "n": int(mask.sum()),
            "threshold": best_thresh,
            "original_ba": orig_ba,
            "tuned_ba": best_ba_cat,
            "delta": best_ba_cat - orig_ba,
        }
        print(f"  {cat_name:20s}: thresh={best_thresh:.2f} | BalAcc {orig_ba:.4f} → {best_ba_cat:.4f} ({best_ba_cat-orig_ba:+.4f})")

    v5c_ba = balanced_accuracy_score(y_v5, v5_corrected)
    v5c_acc = (v5_corrected == y_v5).mean()
    print(f"\n  V5+Corrected BalAcc: {v5c_ba:.4f} | Accuracy: {v5c_acc:.4f}")
    print(f"  Δ from V3: BalAcc {v5c_ba - v3_ba:+.4f}, Acc {v5c_acc - v3_acc:+.4f}")

    # ── 6. Global Threshold Tuning on V3 (for fair comparison) ────────────
    print("\n▶ STEP 6: GLOBAL THRESHOLD TUNING ON V3 (fair comparison)")
    v3_corrected = v3_preds.copy()
    v3_thresh_results = {}
    for cat_name, cat_col in HIGH_ERR_CATS.items():
        if cat_col not in X_v3.columns:
            continue
        mask = X_v3[cat_col] == 1
        if mask.sum() < 20:
            continue
        cat_proba = v3_proba[mask, 1]
        cat_y = y_v3[mask]
        best_thresh, best_ba = 0.5, balanced_accuracy_score(cat_y, v3_preds[mask])
        orig_ba = best_ba
        for t in np.arange(0.20, 0.80, 0.01):
            ba_t = balanced_accuracy_score(cat_y, (cat_proba >= t).astype(int))
            if ba_t > best_ba:
                best_ba = ba_t
                best_thresh = t
        v3_corrected[mask.values] = (v3_proba[mask, 1] >= best_thresh).astype(int)
        v3_thresh_results[cat_name] = {"threshold": best_thresh, "original_ba": orig_ba, "tuned_ba": best_ba}
        print(f"  {cat_name:20s}: thresh={best_thresh:.2f} | BalAcc {orig_ba:.4f} → {best_ba:.4f} ({best_ba-orig_ba:+.4f})")

    v3c_ba = balanced_accuracy_score(y_v3, v3_corrected)
    v3c_acc = (v3_corrected == y_v3).mean()
    print(f"\n  V3+Corrected BalAcc: {v3c_ba:.4f} | Accuracy: {v3c_acc:.4f}")

    # ── 7. Per-Category Comparison ────────────────────────────────────────
    print("\n▶ STEP 7: PER-CATEGORY COMPARISON")
    cat_comparison = []
    for cat_name, cat_col in ALL_DISPLAY_CATS:
        if cat_col not in X_v3.columns or cat_col not in X_v5.columns:
            continue
        v3_mask = X_v3[cat_col] == 1
        v5_mask = X_v5[cat_col] == 1
        if v3_mask.sum() < 10:
            continue

        # V3 baseline
        v3_err = 1 - (v3_preds[v3_mask] == y_v3[v3_mask]).mean()
        v3_fp = ((y_v3[v3_mask] == 0) & (v3_preds[v3_mask] == 1)).sum()
        v3_fn = ((y_v3[v3_mask] == 1) & (v3_preds[v3_mask] == 0)).sum()

        # V5 + threshold
        v5_err = 1 - (v5_corrected[v5_mask.values] == y_v5[v5_mask]).mean()
        v5_fp = ((y_v5[v5_mask] == 0) & (v5_corrected[v5_mask.values] == 1)).sum()
        v5_fn = ((y_v5[v5_mask] == 1) & (v5_corrected[v5_mask.values] == 0)).sum()

        cat_comparison.append({
            "Category": cat_name,
            "N": int(v3_mask.sum()),
            "V3 Err%": v3_err,
            "V5 Err%": v5_err,
            "Δ Err": v5_err - v3_err,
            "V3 FP": int(v3_fp), "V3 FN": int(v3_fn),
            "V5 FP": int(v5_fp), "V5 FN": int(v5_fn),
            "ΔFP": int(v5_fp - v3_fp), "ΔFN": int(v5_fn - v3_fn),
        })

    comp_df = pd.DataFrame(cat_comparison).sort_values("V3 Err%", ascending=False)
    print("\n" + comp_df.to_string(index=False))

    # ── 8. Feature Importance for New Features ────────────────────────────
    print("\n▶ STEP 8: FEATURE IMPORTANCE FOR NEW V5 FEATURES")
    model_v5_deep.fit(X_v5, y_v5)
    fi = pd.Series(model_v5_deep.feature_importances_, index=X_v5.columns).sort_values(ascending=False)

    print("  Top 20 overall:")
    for i, (f, v) in enumerate(fi.head(20).items()):
        marker = " ★" if f in new_features else ""
        print(f"    #{i+1:2d}  {f:40s} {v:.4f}{marker}")

    print("\n  V5 new features ranked:")
    for f in new_features:
        rank = list(fi.index).index(f) + 1
        print(f"    #{rank:2d}/{len(fi)}  {f:40s} {fi[f]:.4f}")

    # ── 9. Overall Error Breakdown ────────────────────────────────────────
    print("\n▶ STEP 9: OVERALL ERROR BREAKDOWN")
    v3_fp_total = ((y_v3 == 0) & (v3_preds == 1)).sum()
    v3_fn_total = ((y_v3 == 1) & (v3_preds == 0)).sum()
    v5_fp_total = ((y_v5 == 0) & (v5_corrected == 1)).sum()
    v5_fn_total = ((y_v5 == 1) & (v5_corrected == 0)).sum()
    print(f"  V3 Baseline:  FP={v3_fp_total}, FN={v3_fn_total}, Total={v3_fp_total+v3_fn_total}")
    print(f"  V5 Enhanced:  FP={v5_fp_total}, FN={v5_fn_total}, Total={v5_fp_total+v5_fn_total}")
    print(f"  FP Change: {v5_fp_total - v3_fp_total:+d}  |  FN Change: {v5_fn_total - v3_fn_total:+d}")

    # ── 10. Generate Charts ───────────────────────────────────────────────
    print("\n▶ STEP 10: GENERATING CHARTS")
    generate_charts(comp_df, v3_ba, best_v5_ba, v5c_ba, v3_acc, best_v5_acc, v5c_acc,
                    v3c_ba, v3c_acc)

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  V3 Baseline:           BalAcc={v3_ba:.4f}  Acc={v3_acc:.4f}")
    print(f"  V3+Cat Thresholds:     BalAcc={v3c_ba:.4f}  Acc={v3c_acc:.4f}  Δ={v3c_ba-v3_ba:+.4f}")
    print(f"  V5 (new features):     BalAcc={best_v5_ba:.4f}  Acc={best_v5_acc:.4f}  Δ={best_v5_ba-v3_ba:+.4f}")
    print(f"  V5+Cat Thresholds:     BalAcc={v5c_ba:.4f}  Acc={v5c_acc:.4f}  Δ={v5c_ba-v3_ba:+.4f}")

    # High-error category summary
    print("\n  HIGH-ERROR CATEGORY CHANGES:")
    for _, row in comp_df.iterrows():
        if row["Category"] in HIGH_ERR_CATS:
            direction = "↓" if row["Δ Err"] < 0 else "↑" if row["Δ Err"] > 0 else "→"
            print(f"    {row['Category']:20s}: V3={row['V3 Err%']:.1%} → V5={row['V5 Err%']:.1%} ({direction} {abs(row['Δ Err']):.1%})  FN: {row['V3 FN']}→{row['V5 FN']} ({row['ΔFN']:+d})")

    return {
        "v3_ba": v3_ba, "v3_acc": v3_acc,
        "v3c_ba": v3c_ba, "v3c_acc": v3c_acc,
        "v5_ba": best_v5_ba, "v5_acc": best_v5_acc,
        "v5c_ba": v5c_ba, "v5c_acc": v5c_acc,
        "comp_df": comp_df,
        "threshold_results": threshold_results,
        "fi": fi,
        "new_features": new_features,
    }


def generate_charts(comp_df, v3_ba, v5_ba, v5c_ba, v3_acc, v5_acc, v5c_acc, v3c_ba, v3c_acc):
    """Generate all comparison charts."""

    # ── Chart 1: Overall Metrics ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = ["V3\nBaseline", "V3+Cat\nThreshold", "V5\n(+Features)", "V5+Cat\nThreshold"]
    colors_ba = ["#94a3b8", "#60a5fa", "#3b82f6", "#22c55e"]

    ax = axes[0]
    ba_vals = [v3_ba, v3c_ba, v5_ba, v5c_ba]
    bars = ax.bar(models, ba_vals, color=colors_ba, edgecolor="white", width=0.55)
    ax.bar_label(bars, [f"{v:.4f}" for v in ba_vals], padding=5, fontsize=10, fontweight="600")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy Comparison", fontweight="bold")
    ax.set_ylim(min(ba_vals) - 0.03, max(ba_vals) + 0.02)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    acc_vals = [v3_acc, v3c_acc, v5_acc, v5c_acc]
    bars = ax.bar(models, acc_vals, color=colors_ba, edgecolor="white", width=0.55)
    ax.bar_label(bars, [f"{v:.4f}" for v in acc_vals], padding=5, fontsize=10, fontweight="600")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy Comparison", fontweight="bold")
    ax.set_ylim(min(acc_vals) - 0.03, max(acc_vals) + 0.02)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("V3 vs V5 Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "v3_vs_v5_overall.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: v3_vs_v5_overall.png")
    plt.close()

    # ── Chart 2: Per-category error rate (high-error cats only) ───────────
    high_err = comp_df[comp_df["V3 Err%"] > 0.05].copy()
    if len(high_err) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(high_err) * 0.6)))
        y_pos = np.arange(len(high_err))
        bh = 0.35
        ax.barh(y_pos + bh/2, high_err["V3 Err%"], bh, color="#ef4444", label="V3 Baseline", alpha=0.85)
        ax.barh(y_pos - bh/2, high_err["V5 Err%"], bh, color="#22c55e", label="V5 Enhanced", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(high_err["Category"], fontsize=11)
        ax.set_xlabel("Error Rate")
        ax.set_title("Per-Category Error Rate: V3 vs V5 (High-Error Categories)", fontweight="bold", pad=15)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=10)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for i, (v3e, v5e) in enumerate(zip(high_err["V3 Err%"], high_err["V5 Err%"])):
            delta = v5e - v3e
            color = "#22c55e" if delta < 0 else "#ef4444"
            ax.text(max(v3e, v5e) + 0.008, i, f"{delta:+.1%}", va="center", fontsize=9, color=color, fontweight="600")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "v3_vs_v5_by_category.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: v3_vs_v5_by_category.png")
        plt.close()

    # ── Chart 3: FN reduction ─────────────────────────────────────────────
    fn_cats = comp_df[(comp_df["V3 FN"] > 0) & (comp_df["Category"].isin(HIGH_ERR_CATS.keys()))].copy()
    if len(fn_cats) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(fn_cats) * 0.7)))
        y_pos = np.arange(len(fn_cats))
        bh = 0.35
        ax.barh(y_pos + bh/2, fn_cats["V3 FN"], bh, color="#ef4444", label="V3 False Negatives", alpha=0.85)
        ax.barh(y_pos - bh/2, fn_cats["V5 FN"], bh, color="#22c55e", label="V5 False Negatives", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fn_cats["Category"], fontsize=11)
        ax.set_xlabel("False Negatives (Open places wrongly predicted Closed)")
        ax.set_title("False Negative Reduction: V3 → V5", fontweight="bold", pad=15)
        ax.legend(fontsize=10)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for i, (v3fn, v5fn) in enumerate(zip(fn_cats["V3 FN"], fn_cats["V5 FN"])):
            delta = v5fn - v3fn
            color = "#22c55e" if delta < 0 else "#ef4444"
            ax.text(max(v3fn, v5fn) + 0.5, i, f"{delta:+d}", va="center", fontsize=10, color=color, fontweight="600")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fn_reduction_by_category.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: fn_reduction_by_category.png")
        plt.close()


if __name__ == "__main__":
    results = run_full_evaluation()
    print("\n✅ V5 evaluation complete!")
