# Model Comparison Results (V3 Update)

## Full Results Table

| Model                         | Dataset              | ROC AUC    | Precision (Closed) | Recall (Closed) | Balanced Acc |
| :---------------------------- | :------------------- | :--------- | :----------------- | :-------------- | :----------- |
| **V3 CatBoost (Combined)** 🏆 | **NYC + SF (18.6k)** | **0.9937** | 0.8476             | **0.9643**      | **0.9519**   |
| V3 CatBoost (NYC)             | Overture NYC (12k)   | 0.9874     | **0.8732**         | 0.9303          | 0.9287       |
| V3 CatBoost (SF)              | Overture SF (9.6k)   | 0.9755     | 0.6640             | 0.9395          | 0.9139       |
| V2 Baseline (CatBoost)        | Season 2 (3k)        | 0.7559     | 0.5726             | 0.6834          | 0.6731       |
| V1 Baseline (Logistic)        | Season 2 (3k)        | 0.7323     | 0.5903             | 0.7221          | 0.6659       |

_Note: V3 improvements are driven by the new Overture Ground Truth dataset and advanced feature engineering (Brand-Awareness, Recency Decay)._

## Model Rankings

### 🥇 Best Overall: **V3 CatBoost (Combined)**

- **Balanced Accuracy**: **95.19%** (New SOTA)
- **Recall (Closed)**: **96.43%** - Captures almost all closed places.
- **Robustness**: Trained on diverse data from two major cities.

### 🥈 Runner-up: **V3 CatBoost (NYC)**

- **Balanced Accuracy**: 92.87%
- **Precision**: 87.3% - Higher precision than Combined, likely due to cleaner initial labeling in NYC.

### 🥉 Baseline: **V2 CatBoost**

- **Balanced Accuracy**: 67.31%
- Served as the initial proof-of-concept but limited by small dataset size (3k) and noisy labels.

## Key Insights

### Logistic Regression Performance

✅ **Surprisingly competitive!**

- Achieved 0.6659 Balanced Accuracy (only 1% behind CatBoost)
- Fast training time (1.42s)
- **Best interpretability** - coefficients show feature importance
- Good recall (0.7221) - catches most closed places
- Moderate precision (0.5903) - some false positives

### Delta Features Impact

The delta features helped all models, especially the linear model:

- Simple linear relationships between delta features and target
- `has_gained_social` (r=+0.26) is highly predictive
- `has_any_loss` (r=-0.17) is a clear closure signal

### Model Comparison

**CatBoost (Recommended)**

- Best balance of precision (0.57) and recall (0.68)
- Handles categorical features automatically
- Strong overall performance

**XGBoost (Alternative)**

- Highest precision (0.67) if minimizing false positives is critical
- Lower recall (0.47) - misses more closed places
- Fastest training

**Logistic Regression (Baseline)**

- Great for understanding feature importance
- Solid performance with minimal complexity
- Easy to deploy and explain

## Feature Importance Analysis (Recommended Next Steps)

For **Logistic Regression**, you can extract coefficients to see which features matter most:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Get feature importance
coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', ascending=False)
```

This will show you exactly how much each delta feature contributes!

## Recommendations

1. **For submission**: Use **CatBoost** (best balanced accuracy)
2. **For interpretation**: Use **Logistic Regression** (transparent feature weights)
3. **For precision**: Use **XGBoost** (highest precision, fewer false alarms)

## Dataset Details

- Season 2 Samples 3k (10/90 split)
- 3,000 rows total
- 49 features (including 12 delta features)
- No mobility data (generalizable to all locations)
