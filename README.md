# StatusNow - Place Status Classification

This project classifies whether a place (POI) is Open or Closed based on its **digital footprint** and **recency signals**.

## 🚀 Current Best Model (V3 + Combined Truth)

We have achieved **95.19% Balanced Accuracy** using our V3 model on a combined ground-truth dataset from NYC and San Francisco.

### V3 Performance Breakthrough (Feb 2026)

| Model Version     | Features Description                       | Dataset              | Balanced Accuracy | ROC AUC    |
| :---------------- | :----------------------------------------- | :------------------- | :---------------- | :--------- |
| **V3 (Combined)** | **Brand-aware + Recency + Label Cleaning** | **NYC + SF (18.6k)** | **95.19%**        | **0.9937** |
| V3 (NYC Only)     | Baseline V3 Model                          | Overture NYC (12k)   | 92.87%            | 0.9874     |
| V2 Baseline       | Interactions + PCA + Category Risk         | Season 2 (3k)        | 70.65%            | 0.7842     |

**Algorithm Comparison (Combined Dataset):**

| Algorithm           | Balanced Accuracy | ROC AUC    | Precision (Closed) | Recall (Closed) |
| :------------------ | :---------------- | :--------- | :----------------- | :-------------- |
| **CatBoost**        | **95.19%**        | 0.9937     | 84.8%              | **96.4%**       |
| XGBoost             | 93.43%            | **0.9938** | **93.6%**          | 89.0%           |
| Logistic Regression | 94.04%            | 0.9813     | 82.1%              | 95.3%           |

_CatBoost is preferred for its superior recall (96.4%) and balanced accuracy._

**Key Findings:**

1.  **Massive Improvement**: Accuracy jumped from ~70% to **~95%** by expanding to a larger, multi-city dataset.
2.  **Generalization**: The model performs robustly across both NYC (93% acc) and San Francisco (91% acc).
3.  **Brand Gap**: While improved, Brands (~90% acc) still perform differently than Non-Brands (~96% acc), suggesting a need for stratified models.

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

### Phase 5: San Francisco Expansion (Generalization)

- **Goal**: Validate if the model works beyond NYC.
- **Method**: Replicated the pipeline for San Francisco (SF) and created a combined dataset.
- **Results**:
  - **SF Accuracy**: **91.39%** (despite fewer closed samples).
  - **Combined Model**: **95.19%** Balanced Accuracy on 18,619 samples.
- **Key Insight**: The model generalizes well, but the brand/non-brand gap is wider in SF (-12.8%). Combining data significantly improves robustness.
