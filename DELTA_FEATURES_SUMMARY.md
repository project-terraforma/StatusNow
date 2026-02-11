# Dataset & Feature Engineering Summary

## Dataset Information

### Primary Dataset (Season 2 Samples 3k - 10/90 Split)

- **File**: `data/Season 2 Samples 3k Project Updated.parquet`
- **Rows**: 3,000
- **Class Distribution**: 60.3% Open (1,809), 39.7% Closed (1,191)
- **Purpose**: Correct dataset for this assignment
- **Total Columns**: 23

### Alternative Dataset (Project C - 60/40 Split)

- **File**: `data/Project C Samples.parquet`
- **Rows**: 3,425
- **Purpose**: From a different assignment, can be used if helpful
- **Row Difference**: +425 rows compared to Season 2
- **Total Columns**: 16

## Column Comparison

### Season 2 Unique Columns (baseline data):

- `label` - Target variable (renamed to 'open' in processing)
- **Baseline/Historical Fields** (12 columns):
  - `base_id`, `base_sources`, `base_names`, `base_categories`, `base_confidence`
  - `base_websites`, `base_socials`, `base_emails`, `base_phones`, `base_brand`, `base_addresses`

### Project C Unique Columns:

- `open` - Target variable
- `geometry`, `bbox`, `type`, `version` - Geospatial metadata

### Shared Core Columns (11):

- `id`, `sources`, `names`, `categories`, `confidence`
- `websites`, `socials`, `emails`, `phones`, `brand`, `addresses`

## Feature Engineering Strategy

### Removed Features

❌ **Mobility Features** - Removed for generalizability across all locations

- `mobility_score`
- `is_ghost_candidate`

### Added Delta Features ✅ (12 new features)

The delta features compare **baseline vs current** data to detect changes in digital presence - a powerful signal for place closure.

#### 1. Confidence Change

- `delta_confidence`: Change in confidence score (r=+0.1668)
  - Open places: -0.0403 average delta
  - Closed places: -0.1298 average delta

#### 2. Digital Presence Deltas (Raw Changes)

- `delta_num_websites`: Net change in website count (r=+0.0766)
- `delta_num_socials`: Net change in social media links (r=+0.1841)
- `delta_num_phones`: Net change in phone numbers (r=+0.0850)

#### 3. Binary Loss/Gain Indicators

- `has_lost_website`: Lost at least one website (r=-0.1227)
  - Open: 12.7%, Closed: 22.0%
- `has_gained_website`: Gained at least one website (r=-0.1354)
- `has_lost_social`: Lost social media presence (r=-0.1327)
  - Open: 45.2%, Closed: 58.8%
- `has_gained_social`: **Gained social media** (r=+0.2614) ⭐ **STRONGEST PREDICTOR**
  - Open: 28.5%, Closed: 7.1%
- `has_lost_phone`: Lost phone number (r=-0.1193)
  - Open: 4.8%, Closed: 11.2%
- `has_gained_phone`: Gained phone number (r=-0.0984)

#### 4. Composite Indicators

- `delta_total_contact`: Total change across all contact methods (r=+0.1895)
  - Open: -0.86 average, Closed: -1.47 average
- `has_any_loss`: **Lost ANY digital presence** (r=-0.1719) ⭐
  - Open: 46.6%, Closed: 64.1%

## Top 5 Most Predictive Delta Features

1. **has_gained_social** (r=+0.2614) - Places that gained social media are MORE likely open
2. **delta_total_contact** (r=+0.1895) - Positive change in total contacts → more likely open
3. **delta_num_socials** (r=+0.1841) - Increase in social media count → more likely open
4. **has_any_loss** (r=-0.1719) - Any loss in digital presence → MORE likely closed
5. **delta_confidence** (r=+0.1668) - Confidence increase → more likely open

## Model Performance (5-Fold CV)

### Current Results with Delta Features:

```
              Time (s)  ROC AUC  F1 Macro  Precision (Closed)  Recall (Closed)  Balanced Acc
Balanced RF     2.29     0.7382    0.6366         0.5315            0.7380          0.6548
XGBoost         1.06     0.7601    0.6629         0.6720            0.4702          0.6599
CatBoost        1.21     0.7559    0.6645         0.5726            0.6834          0.6731
EasyEnsemble    2.71     0.7181    0.6384         0.5423            0.6868          0.6502
```

**Best Model**: **CatBoost** with Balanced Accuracy of **0.6731**

## Final Feature Count

- **Total Features**: 49
- **Delta Features**: 12
- **Category One-Hot Features**: ~20
- **Digital Presence Features**: 9
- **Source/Recency Features**: 6
- **Other Features**: ~2

## Key Insights

1. ✅ **Delta features show strong predictive power** - Top feature has r=0.26
2. ✅ **Social media changes are the strongest signal** - Gaining social media strongly indicates the place is still open
3. ✅ **Digital presence loss is a clear closure indicator** - 64% of closed places lost some digital presence
4. ✅ **Model is now generalizable** - No mobility data dependency
5. ✅ **Using correct dataset** - Season 2 with 10/90 split (3,000 rows)

## Next Steps

Consider:

- Feature selection to remove redundant features
- Hyperparameter tuning for CatBoost
- Ensemble of top models
- Additional domain knowledge features
