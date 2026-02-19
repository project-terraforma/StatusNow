# StatusNow - Place Status Classification

This project classifies whether a place (POI) is Open or Closed based on its **digital footprint** and **recency signals**.

## 🚀 Current Best Model (V3 + Overture Truth)

We have achieved **92.87% Balanced Accuracy** using our V3 model on a ground-truth dataset constructed from Overture Maps.

### V3 Performance Breakthrough (Feb 2026)

| Model Version  | Features Description                            | Dataset                  | Balanced Accuracy | ROC AUC    |
| :------------- | :---------------------------------------------- | :----------------------- | :---------------- | :--------- |
| **V3 (Final)** | **Brand-aware + Recency Bins + Label Cleaning** | **Overture Truth (12k)** | **92.87%**        | **0.9874** |
| V3 (Interim)   | Label Refinement applied                        | Season 2 (3k)            | 72.09%            | 0.7912     |
| V2 Baseline    | Interactions + PCA + Category Risk              | Season 2 (3k)            | 70.65%            | 0.7842     |
| V1 Baseline    | Delta Features (Simple Change)                  | Season 2 (3k)            | 67.31%            | 0.7421     |

**Algorithm Comparison (V3 + Clean):**

| Algorithm           | Balanced Accuracy | ROC AUC    | Precision (Closed) | Recall (Closed) |
| :------------------ | :---------------- | :--------- | :----------------- | :-------------- |
| **CatBoost**        | **92.87%**        | 0.9874     | 87.3%              | **93.0%**       |
| XGBoost             | 92.17%            | **0.9876** | **92.9%**          | 88.0%           |
| Logistic Regression | 91.14%            | 0.9694     | 85.1%              | 90.8%           |

_CatBoost is preferred for its balanced accuracy and high recall on closed places._

**Key Findings:**

1.  **Massive Improvement**: Accuracy jumped from ~70% to **~93%** when validating on ground-truth data.
2.  **Brand Gap**: Brands (97% acc) are much easier to classify than Non-Brands (67% acc).
3.  **Validation**: Confirms that V3 features are highly effective at detecting closure signals like "zombie" states and digital decay.

---

## Data

- `data/combined_truth_dataset.parquet`: **GOLD STANDARD** (12,000 rows).
  - Combined dataset: 3,000 Season 2 samples + 9,000 Overture NYC samples.
  - Class Balance: ~65% Open / 35% Closed.
  - Built by comparing Overture Maps releases (Jan vs Feb 2026) to identify true closures.

- `data/Season 2 Samples 3k Project Updated.parquet`: Original primary dataset (3,000 rows).
- `data/processed_for_ml_testing.parquet`: The final processed feature file used for the V3 results above.

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
# 1. Build the Overture Truth Dataset (NYC BBox)
# This downloads comparable slices from Jan 2026 & Feb 2026 releases
python scripts/data_processing/fetch_overture_data.py
python scripts/data_processing/build_truth_dataset.py

# 2. Feature Engineering (Generate V3 Features)
python scripts/data_processing/process_data_v3.py

# 3. Run the V3 Experiments
python scripts/experiments/experiment_runner_v3.py
```

---

## Repository Structure

> **📖 Need help navigating? See [NAVIGATION.md](NAVIGATION.md) for a complete guide!**

- **`scripts/data_processing/`**:
  - `process_data_v3.py`: **V3 Pipeline** (Feature Engineering).
  - `build_truth_dataset.py`: Logic to construct the Overture Ground Truth.
  - `fetch_overture_data.py`: DuckDB script to download Overture slices.
- **`scripts/experiments/`**:
  - `experiment_runner_v3.py`: **V3 Experiments** (Label refinement + brand stratification).

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
