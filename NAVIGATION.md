# StatusNow Project - Navigation Guide

Welcome to the StatusNow Place Classification Project! This guide will help you navigate the repository and understand the project structure.

## 📁 Repository Structure

```
StatusNow/
├── README.md                          # Main project documentation (START HERE!)
├── LICENSE                            # Apache 2.0 License
│
├── data/                              # Dataset files
│   ├── Season 2 Samples 3k Project Updated.parquet  # Primary dataset (3,000 rows)
│   ├── Project C Samples.parquet                    # Alternative dataset (3,425 rows)
│   ├── processed_for_ml.parquet                     # V1 features (48 features)
│   └── processed_for_ml_v2.parquet                  # V2 features (44 features) ⭐
│
├── scripts/                           # All Python scripts organized by function
│   │
│   ├── data_processing/              # Data loading and feature engineering
│   │   ├── read_data.py              # Quick data inspection tool
│   │   ├── inspect_data.py           # Deep data quality analysis
│   │   ├── process_data.py           # V1: Delta features pipeline
│   │   └── process_data_v2.py        # V2: Advanced features (RECOMMENDED) ⭐
│   │
│   ├── experiments/                   # Model training and evaluation
│   │   ├── experiment_runner.py      # V1: 5-fold CV on baseline features
│   │   └── experiment_runner_v2.py   # V2: Comparison + breakthrough results ⭐
│   │
│   ├── analysis/                      # Feature analysis and selection
│   │   ├── analyze_delta_features.py # Delta feature correlation analysis
│   │   └── feature_selection.py      # Redundancy detection & optimization
│   │
│   └── archived/                      # Deprecated scripts (kept for reference)
│       ├── enrich_with_fused.py      # Old: Mobility data enrichment
│       ├── fetch_geo_from_s3.py      # Old: S3 geocoding utilities
│       └── debug_fetch_geo.py        # Old: Debug tool
│
└── docs/                              # Documentation and analysis summaries
    ├── recommended_features.txt       # Optimal 30-feature subset
    └── summaries/                     # Detailed analysis documents
        ├── BREAKTHROUGH_V2_RESULTS.md # V2 advanced features summary ⭐
        ├── DELTA_FEATURES_SUMMARY.md  # V1 delta features analysis
        └── MODEL_COMPARISON.md        # Model performance comparison
```

## 🚀 Quick Start for Reviewers

### 1. **Understand the Project** (5 minutes)

- Read **[README.md](../README.md)** - Comprehensive overview with results
- Focus on:
  - Data section (what we're working with)
  - V2 Breakthrough results table (70.65% balanced accuracy!)
  - Journey summary (baseline → breakthrough)

### 2. **See the Breakthrough** (2 minutes)

- Check **[docs/summaries/BREAKTHROUGH_V2_RESULTS.md](summaries/BREAKTHROUGH_V2_RESULTS.md)**
- Shows V1 vs V2 comparison and new feature explanations

### 3. **Run the Code** (10 minutes)

```bash
# Setup (one-time)
python -m venv .venv
source .venv/bin/activate
pip install duckdb pandas numpy scikit-learn imbalanced-learn xgboost catboost pyarrow

# Generate V2 features (RECOMMENDED)
python scripts/data_processing/process_data_v2.py

# Run V2 experiments and see the breakthrough
python scripts/experiments/experiment_runner_v2.py
```

### 4. **Explore Advanced Features** (5 minutes)

- Open **[scripts/data_processing/process_data_v2.py](../scripts/data_processing/process_data_v2.py)**
- See sections:
  - Interaction Features (lines ~185-195)
  - Category Churn Risk (lines ~200-215)
  - Digital Congruence (lines ~220-240)
  - PCA Implementation (lines ~245-255)

## 📊 Key Results at a Glance

| Metric                | V1 (Delta Features) | V2 (Advanced Features) | Improvement       |
| --------------------- | ------------------- | ---------------------- | ----------------- |
| **Balanced Accuracy** | 67.31%              | **70.65%**             | **+4.97%**        |
| **Features**          | 48 features         | 44 features            | Fewer & better!   |
| **Best Model**        | CatBoost            | CatBoost               | Consistent winner |

## 🎯 What Makes V2 Special?

1. **Interaction Features**: Captures _when_ changes happened (recent loss >> old loss)
2. **Category Risk**: Gas stations ≠ Boutiques (industry-specific signals)
3. **PCA**: Removed redundancy (98.68% variance with 1 component vs 2 features)
4. **Digital Congruence**: Website/social consistency checks

## 📖 Documentation Index

### For Quick Understanding

- **[README.md](../README.md)** - Start here, complete overview
- **[docs/summaries/BREAKTHROUGH_V2_RESULTS.md](summaries/BREAKTHROUGH_V2_RESULTS.md)** - What changed in V2

### For Deep Dive

- **[docs/summaries/DELTA_FEATURES_SUMMARY.md](summaries/DELTA_FEATURES_SUMMARY.md)** - How delta features work
- **[docs/summaries/MODEL_COMPARISON.md](summaries/MODEL_COMPARISON.md)** - All 5 models compared
- **[docs/recommended_features.txt](recommended_features.txt)** - Optimal 30 features list

### For Implementation

- **[scripts/data_processing/process_data_v2.py](../scripts/data_processing/process_data_v2.py)** - V2 feature engineering
- **[scripts/experiments/experiment_runner_v2.py](../scripts/experiments/experiment_runner_v2.py)** - V2 training pipeline
- **[scripts/analysis/feature_selection.py](../scripts/analysis/feature_selection.py)** - Feature optimization analysis

## 🔄 Workflow

```
1. Raw Data (Season 2 Samples)
         ↓
2. Feature Engineering (process_data_v2.py)
         ↓
3. Feature Selection (feature_selection.py) [optional]
         ↓
4. Model Training (experiment_runner_v2.py)
         ↓
5. Breakthrough: 70.65% Balanced Accuracy! 🎯
```

## ⚡ Optional: Scale to 6,400 Samples

Want even better results? Merge both datasets:

```bash
python scripts/data_processing/process_data_v2.py --merge
python scripts/experiments/experiment_runner_v2.py
```

This combines Season 2 (3,000) + Project C (3,425) for ~6,400 total samples.

## 🏆 Recommended Review Order

1. ✅ **README.md** (10 min) - Big picture
2. ✅ **BREAKTHROUGH_V2_RESULTS.md** (5 min) - What's new
3. ✅ **Run experiment_runner_v2.py** (5 min) - See it work
4. ✅ **Read process_data_v2.py** (10 min) - Understand features
5. ✅ **Explore other docs** (optional) - Go deeper

---

**Questions?** Check the [README.md](../README.md) or review the inline comments in the scripts!
