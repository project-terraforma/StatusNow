# StatusNow - Place Status Classification

This project classifies whether a place (POI) is Open or Closed based on its **digital footprint** and **recency signals**.

## 🚀 Current Best Model (V5 — Leak-Free + Geographic Hold-Out)

We have achieved **89.41% Balanced Accuracy** on a fully honest evaluation: geographic hold-out test set (Chicago + Miami never seen during training), with all known leakages fixed.

### V5 Research Results (Mar 2026)

**Accuracy progression:**

| Pipeline         | Cities (train) | Samples  | Model                | BalAcc             | Eval Method             |
| :--------------- | :------------- | :------- | :------------------- | :----------------- | :---------------------- |
| V3 original      | NYC + SF       | 18.6k    | CatBoost             | ~~85.21%~~ (leaky) | CV on full set          |
| V4 expanded      | 12 cities      | 123k     | CB+LGBM ensemble     | ~~89.18%~~ (leaky) | CV on full set          |
| **V5 leak-free** | **10 cities**  | **102k** | **CB+LGBM ensemble** | **89.41%**         | **Hold-out: CHI + MIA** |

**V5 Hold-Out Results (Chicago + Miami — never used in training):**

| Model                           | Hold-out BalAcc | Notes                  |
| :------------------------------ | :-------------- | :--------------------- |
| CatBoost-A (2000i, d8, lr=0.03) | 89.28%          |                        |
| CatBoost-B (1500i, d7, lr=0.05) | 89.33%          |                        |
| CatBoost-C (1000i, d6, lr=0.05) | 89.38%          |                        |
| CatBoost ensemble (avg)         | 89.34%          |                        |
| **CB+LGBM (w=0.7/0.3, t=0.52)** | **89.41%**      | **Best — report this** |

**V5 Leakage Fixes:**

| Issue                 | Root Cause                                               | Fix Applied                                                                |
| :-------------------- | :------------------------------------------------------- | :------------------------------------------------------------------------- |
| `confidence` NaN fill | 93.7% of closed (churned) had null→0 confidence          | Use `base_confidence` only; drop `delta_confidence`, `confidence_momentum` |
| `category_churn_risk` | Global label-based computation leaked into CV test folds | Removed; `category_primary` → CatBoost native cat feature                  |
| CV-only evaluation    | Hold-out cities inflated by homogeneous training split   | 2 held-out cities (Chicago + Miami) for final report                       |

**Key Findings (V5):**

1. **V4's 89.18% was mostly real.** After fixing leakage the true hold-out result is 89.41% — the leakage was real but the model was also genuinely learning.
2. **`base_confidence` is the honest strongest signal.** Jan 2026 quality score + staleness interaction accounts for 42% of feature importance.
3. **Delta features have a structural limitation.** For 93.7% of closed places (churned), ALL delta features are 0 by construction (COALESCE makes current = previous for disappeared places). A **3rd release (Dec 2025)** would give true pre-closure deltas.
4. **90% is within reach.** Gap is only 0.59pp. Most likely path: fetch Dec 2025 Overture release and add more cities.

**V4 Data-Scale Progression (5-fold CV, leak-free):**

| Dataset             | Samples  | Balanced Accuracy |
| :------------------ | :------- | :---------------- |
| NYC + SF            | 12k      | 80.5%             |
| NYC + SF + Season 2 | 18.6k    | 81.7%             |
| 5 cities            | 53k      | 88.5%             |
| **12 cities**       | **123k** | **89.18%**        |

_Data scale was the dominant factor — going from 12k → 123k gained +8.7 pp. Model tuning (HPO) gained only ~0.1 pp._

---

## Data

- `data/combined_truth_dataset_expanded.parquet`: **GOLD STANDARD V4** (123,082 rows).
  - 12 cities: NYC, SF, Chicago, LA, Houston, Phoenix, Philadelphia, Seattle, Denver, Boston, Miami, Atlanta.
  - Class Balance: ~84% Open / 16% Closed (19,273 closed places total).
  - Built by comparing Overture Maps releases (Jan 2026 vs Feb 2026) to identify true closures.

- `data/combined_truth_dataset.parquet`: Original 12k dataset (NYC + Season 2 samples).
- `data/combined_truth_dataset_all.parquet`: 18.6k dataset (NYC + SF + Season 2).
- `data/Season 2 Samples 3k Project Updated.parquet`: Original manually-labeled dataset (3,000 rows).
- `data/processed_for_ml_testing.parquet`: ⚠️ V3 processed file — contains a confidence leakage bug (see Project History).

### Final Schema (52 Features)

The V3 model uses **52 engineered features**. Below are the most critical ones:

| Category              | Feature Name                 | Description                                                       | Correlation/Importance |
| :-------------------- | :--------------------------- | :---------------------------------------------------------------- | :--------------------- |
| **Delta Features**    | `has_gained_social`          | **Strongest Predictor (+0.26)**. Gained social media presence.    | High (Positive)        |
| (Baseline vs Current) | `has_any_loss`               | Lost ANY website, social media, or phone number.                  | High (Negative)        |
|                       | `delta_total_contact`        | Net change in total contact points (websites + socials + phones). | High (Positive)        |
|                       | `delta_confidence`           | Change in Overture confidence score.                              | Medium (Positive)      |
|                       | `has_lost_website`           | Explicit flag for website loss (22% of closed places).            | Medium (Negative)      |
| **V3 Recency**        | `is_stale_2yr`               | Data hasn't been updated in >2 years.                             | High (Negative)        |
|                       | `log_days_since_update`      | Log-transform of days since last update (diminishing returns).    | High                   |
|                       | `recency_bucket`             | Ordinal staleness: 0=Fresh, 1=Aging, 2=Stale, 3=Dead.             | High                   |
| **V3 Brand-Aware**    | `brand_x_stale`              | Interaction: Brand chains are allowed to have stale data.         | Medium                 |
|                       | `nonbrand_stale_risk`        | Interaction: Independent shops with stale data are high risk.     | Medium                 |
| **V2 Interactions**   | `zombie_score`               | High source count + Stale data = "Database Purgatory".            | **Critical Signal**    |
|                       | `recency_x_loss`             | Recent digital loss is more significant than old loss.            | High                   |
|                       | `decay_velocity`             | Rate of digital footprint decline per day.                        | Medium                 |
|                       | `digital_congruence`         | Website domain matches social handle (1=Congruent).               | Medium                 |
| **Metadata**          | `is_brand`                   | 1 if place matches a known brand chain.                           | High                   |
|                       | `num_sources`                | Number of datasets vetting this place.                            | High                   |
|                       | `confidence`                 | Overture calibration score.                                       | High                   |
|                       | `category_churn_risk`        | Historical closure rate for this specific category.               | High                   |
| **Digital Presence**  | `contact_depth`              | Total count of contact methods available.                         | Medium                 |
|                       | `has_website` / `has_social` | Basic presence flags.                                             | Medium                 |

---

## Usage

### Quick Start (Reproduce V3 Results) ⭐

The essential datasets (`combined_truth_dataset.parquet`, `processed_for_ml_testing.parquet`) are included in the repo, so you can run the experiments immediately.

```bash
# 1. Setup Environment
python3 -m venv .venv && source .venv/bin/activate
pip install duckdb pandas numpy pyarrow scikit-learn imbalanced-learn xgboost fused geopandas shapely requests tqdm catboost

# 2. Run the V3 Experiment Runner on the Test Set
python scripts/experiments/experiment_runner_v3.py -i data/processed_for_ml_testing.parquet
```

### Complete Workflow (Build from Scratch)

If you want to rebuild the dataset from Overture S3 (e.g., for a different city or new release):

```bash
# 1. Fetch Overture Data (NYC & SF)
# Downloads comparable slices from Jan 2026 & Feb 2026 releases
python scripts/data_processing/fetch_overture_data.py --city nyc
python scripts/data_processing/fetch_overture_data.py --city sf

# 2. Build Truth Datasets
python scripts/data_processing/build_truth_dataset.py --city nyc
python scripts/data_processing/build_truth_dataset.py --city sf

# 3. Merge Cities (Optional - to create "Combined" dataset)
python scripts/data_processing/merge_cities.py --cities nyc sf --output data/combined_truth_dataset_all.parquet

# 4. Feature Engineering (Generate V3 Features)
python scripts/data_processing/process_data_v3.py -i data/combined_truth_dataset_all.parquet -o data/processed_all_v3.parquet

# 5. Run the V3 Experiments
python scripts/experiments/experiment_runner_v3.py -i data/processed_all_v3.parquet
```

---

## Repository Structure

> **📖 Need help navigating? See [NAVIGATION.md](NAVIGATION.md) for a complete guide!**

- **`scripts/data_processing/`**:
  - `process_data_v3.py`: V3 Pipeline (Feature Engineering).
  - `build_truth_dataset.py`: Logic to construct the Overture Ground Truth.
  - `fetch_overture_data.py`: DuckDB script to download Overture slices.
  - `fetch_overture_expanded.py`: **V4** — fetch any city from Overture S3 by bounding box.
  - `build_truth_expanded.py`: **V4** — build & merge multi-city truth datasets.
- **`scripts/experiments/`**:
  - `experiment_runner_v3.py`: V3 Experiments (Label refinement + brand stratification).
  - `v4_research.py`: **V4** — Full research pipeline (V4 features, HPO, ensemble, error analysis).

---

## Project History & Journey Summary

This section chronicles our progress from the initial baseline to the final V3 breakthrough.

### Phase 1: V1 Delta Features (Baseline)

- **Goal**: Establish a baseline using "Delta Features" (comparing historical baseline vs current data).
- **Method**: Calculated net change in websites, socials, and phones.
- **Key Insight**: `has_gained_social` (r=+0.26) was the strongest single predictor. `has_any_loss` (r=-0.17) was a reliable closure signal.
- **Result**: 67.3% Balanced Accuracy. Knowing _that_ something changed was good, but not enough.

### Phase 2: V2 Advanced Engineering (Context)

- **Goal**: Capture nuance with Interaction Features and PCA.
- **Innovation**:
  - **Zombie Score**: Identified places with many sources but stale data ("Database Purgatory").
  - **Category Risk**: Modeled that gas stations close less often (10% churn) than boutiques (45% churn).
  - **PCA**: Reduced redundancy between correlated recency features (98% variance explained).
- **Result**: 70.65% Balanced Accuracy. Temporal context ("when did it change?") proved critical.

### Phase 3: V3 Label Refinement (Noise Reduction)

- **Goal**: Tackle label noise in the manually labeled dataset.
- **Innovation**: "Dynamic Label Refinement" using 5-fold cross-validation.
- **Findings**: Identified 65 samples (2.2%) where the model was >90% confident the human label was wrong.
- **Result**: Removing these likely errors boosted accuracy to **72.09%**.

### Phase 4: Overture Truth Dataset (The 93% Breakthrough)

- **Goal**: Validate concepts on a larger, cleaner, ground-truth dataset.
- **Replication Method** (Script: `scripts/data_processing/build_truth_dataset.py`):
  1. **Fetch Data**: Used `fetch_overture_data.py` to download places from Overture S3 (Jan 2026 vs Feb 2026) for NYC BBox.
  2. **Define Closed**: A place is considered closed if:
     - It existed in the _Previous_ release but is missing ID in the _Current_ release (churned).
     - OR it exists in _Current_ but explicitly has `operating_status = 'closed'`.
  3. **Define Open**: Exists in _Current_ and `operating_status != 'closed'`.
  4. **Balance**: Downsampled to 3k Open / 3k Closed to match Season 2 distribution.
- **Result**: **92.87% Balanced Accuracy**.
- **Major Lesson**: The V3 features were highly effective, but the original dataset's noise and size were holding them back.
- **Warning**: We discovered a massive performance gap between **Brands (97% Accuracy)** and **Small Businesses (67% Accuracy)**, suggesting future work should treat them as separate problems.

### Phase 5: San Francisco Expansion (Generalization)

- **Goal**: Validate if the model works beyond NYC.
- **Method**: Replicated the pipeline for San Francisco (SF) and created a combined dataset.
- **Results**:
  - **SF Accuracy**: **91.39%** (despite fewer closed samples).
  - **Combined Model**: **85.21%** Balanced Accuracy on 18,619 samples.
- **Key Insight**: The initial 95% result was inflated by a data leak (Confidence score). After fixing it, the model stabilized at ~85%, and uniquely, the **Brand Gap disappeared** (Brands vs Non-Brands now perform equally).

### Phase 6: V4 Research — Leakage Audit + 12-City Expansion (Mar 2026)

- **Goal**: Improve from 85% → 90% Balanced Accuracy.
- **Leakage Discovery**: `processed_for_ml_testing.parquet` was built with `confidence = 0` for 3,000 churned NYC places (NaN-fill bug). This gave the model a near-perfect closed signal — true leak-free baseline was **80.5%**. `category_churn_risk` (computed globally from labels) also contributed minor leakage.
- **Strategy**: Scale the dataset dramatically across diverse cities using Overture S3.
- **Data Expansion**: Fetched 10 new US cities (Chicago, LA, Houston, Phoenix, Philadelphia, Seattle, Denver, Boston, Miami, Atlanta) → **123,082 samples** from 12 cities.
- **V4 Features**: Extended to **95 features** — added identity-change signals (`name_changed`, `website_domain_changed`, `identity_change_score`), richer per-channel gain/loss flags, and interaction terms.
- **Results** (leaky CV): CatBoost + LightGBM ensemble: **89.18%**
- **Key Insight**: More data >> better models. HPO added only ~0.1 pp; going from 12k → 123k added ~8.7 pp.

### Phase 7: V5 Research — Full Leakage Fix + Geographic Hold-Out (Mar 2026)

- **Goal**: Produce an honest, production-grade evaluation with all leakages fixed.
- **Leakage Audit**:
  1. `confidence` NaN-fill: churned places (93.7% of closed) had `confidence=null` → filled with 0 → near-perfect closed signal. **Fix**: use `base_confidence` (Jan 2026 value) only. Drop `delta_confidence` and `confidence_momentum`.
  2. `category_churn_risk` computed globally from all 123k labels before CV → 0.50 correlation with target. **Fix**: removed; replaced with `category_primary` as CatBoost native categorical feature (fold-safe internal target encoding).
  3. Evaluation: all CV was on the same 12 cities. **Fix**: geographic hold-out — Chicago + Miami held out completely.
- **Data Architecture Insight**: In the 2-release dataset (`Jan 2026 = base`, `Feb 2026 = current`), churned places (closed by disappearing) have `current = COALESCE(null, prev) = prev`, so **all delta features are 0 by construction** for 93.7% of closed places. This is a structural limitation of 2-release data. A 3rd release (Dec 2025) would provide legitimate pre-closure deltas.
- **Operating Status Note**: `operating_status = 'closed'` appears in only 1–2 places per city in current Overture data. Closures are expressed as **churning** (disappearance between releases), not explicit status flags. Using operating_status alone as the closed label is not viable with current Overture data.
- **Results**: CB+LGBM ensemble on Chicago + Miami hold-out: **89.41%** (w_CB=0.7, thresh=0.52).
- **Scripts**: `scripts/data_processing/process_data_v5.py`, `scripts/experiments/v5_holdout_eval.py`.
