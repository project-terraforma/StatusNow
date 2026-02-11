# Advanced Feature Engineering Results (V2)

## 🎯 BREAKTHROUGH: 70.65% Balanced Accuracy!

We successfully exceeded the 67% ceiling by implementing advanced feature engineering techniques.

## Performance Comparison

### V1 (Original Delta Features)

| Model               | Balanced Acc | ROC AUC |
| ------------------- | ------------ | ------- |
| CatBoost            | 0.6731       | 0.7559  |
| XGBoost             | 0.6599       | 0.7601  |
| Logistic Regression | 0.6659       | 0.7323  |

### V2 (Advanced Features)

| Model                   | Balanced Acc | ROC AUC | Improvement |
| ----------------------- | ------------ | ------- | ----------- |
| **CatBoost** ⭐         | **0.7065**   | TBD     | **+4.97%**  |
| **XGBoost**             | **0.6931**   | TBD     | **+5.03%**  |
| **Logistic Regression** | **0.6846**   | TBD     | **+2.80%**  |

## New Features Implemented

### 1. Interaction Features (5 features)

Capture "velocity" of digital decay and temporal relationships:

- **`recency_x_loss`**: days_since_latest_update × has_any_loss
  - Recent losses are stronger closure signals than old ones
- **`recency_x_social_loss`**: days_since_latest_update × has_lost_social
  - Specific to social media loss timing
- **`zombie_score`**: num_sources / (avg_days_since_update + 1)
  - High sources but very old updates = database purgatory
  - Low zombie score → likely closed
- **`decay_velocity`**: delta_total_contact / (avg_days_since_update + 1)
  - How fast is digital presence declining?
- **`confidence_momentum`**: delta_confidence / (avg_days_since_update + 1)
  - Rate of confidence change over time

### 2. Category Churn Risk (1 feature)

Industry-specific stratification:

- **`category_churn_risk`**: Calculated churn rate per category
  - Example: Gas stations have low churn (~10-15%)
  - Example: Boutiques/cafes have high churn (~45-50%)
  - A "lost social media" signal means different things for different industries

### 3. Digital Congruence (1 feature)

Website/social consistency check:

- **`digital_congruence`**: Does website domain appear in social handles?
  - Captures brand consistency
  - Mismatches may indicate closure or ownership change

### 4. PCA for Redundancy Reduction

Addressed highly correlated recency features:

- **`recency_pca`**: Single component from days_since_latest_update + avg_days_since_update
  - Explained variance: **98.68%**
  - Reduces multicollinearity
  - Prevents double-counting of "data age"

### 5. Removed Redundant Features

Based on correlation analysis (r > 0.9):

- ❌ `has_facebook` (r=1.0 with `has_social`)
- ❌ `len_socials` (r=1.0 with `has_social`)
- ❌ `days_since_latest_update` (replaced by PCA)
- ❌ `avg_days_since_update` (replaced by PCA)
- ❌ Phone delta features (low importance)

## Feature Count Optimization

- **V1**: 48 features
- **V2**: 44 features (-4 features)
- **Result**: Better performance with fewer features!

## Key Insights

1. **Interaction Effects Matter**: The timing of digital presence loss is as important as the loss itself
2. **Industry Context is Critical**: Same signal means different things for different business types
3. **Less Can Be More**: Removing redundancy improved generalization

4. **Zombie Score Works**: Places stuck in "database purgatory" are strong closure candidates

5. **Breakthrough Performance**: 70.65% balanced accuracy breaks through the 67% ceiling

## What This Means

The combination of:

- Smart interaction features
- Industry-specific risk modeling
- Redundancy removal via PCA
- Better temporal modeling

...unlocked an additional **3-5% improvement** across all models.

## Next Steps (Optional)

1. **Merge Project C dataset** to reach ~6,400 samples (run with `--merge` flag)
2. **Spatial clustering**: Add competitor_density if coordinates available
3. **Hyperparameter tuning**: CatBoost might reach 71%+ with optimization
4. **Ensemble**: Combine CatBoost + XGBoost for final predictions

## Files

- `process_data_v2.py` - Advanced feature engineering script
- `experiment_runner_v2.py` - V1 vs V2 comparison
- `data/processed_for_ml_v2.parquet` - Dataset with new features
