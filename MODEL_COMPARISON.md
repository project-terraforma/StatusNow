# Model Comparison Results (with Delta Features)

## Full Results Table

| Model            | Time (s) | ROC AUC    | F1 Macro   | Precision (Closed) | Recall (Closed) | Balanced Acc |
| ---------------- | -------- | ---------- | ---------- | ------------------ | --------------- | ------------ |
| **CatBoost** ⭐  | 1.83     | 0.7559     | **0.6645** | 0.5726             | 0.6834          | **0.6731**   |
| **XGBoost**      | 1.02     | **0.7601** | 0.6629     | **0.6720**         | 0.4702          | 0.6599       |
| **Logistic Reg** | 1.42     | 0.7323     | 0.6502     | 0.5903             | **0.7221**      | 0.6659       |
| Balanced RF      | 1.92     | 0.7382     | 0.6366     | 0.5315             | 0.7380          | 0.6548       |
| EasyEnsemble     | 2.74     | 0.7181     | 0.6384     | 0.5423             | 0.6868          | 0.6502       |

## Model Rankings

### 🥇 Best Overall: **CatBoost**

- **Balanced Accuracy**: 0.6731 (1st)
- **F1 Macro**: 0.6645 (1st)
- **ROC AUC**: 0.7559 (2nd)
- Best all-around performance with good balance between precision and recall

### 🥈 Runner-up: **XGBoost**

- **ROC AUC**: 0.7601 (1st)
- **Precision (Closed)**: 0.6720 (1st)
- **Balanced Accuracy**: 0.6599 (3rd)
- Highest discrimination ability, but lower recall for closed places

### 🥉 Baseline: **Logistic Regression**

- **Balanced Accuracy**: 0.6659 (2nd)
- **Recall (Closed)**: 0.7221 (1st among non-RF models)
- Fast and interpretable with competitive performance!

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
