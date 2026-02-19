# StatusNow Project - Navigation Guide

Welcome to the StatusNow Place Classification Project! This guide will help you navigate the repository and understand the project structure.

## 📁 Repository Structure

```
StatusNow/
├── README.md                          # Main project documentation (START HERE!)
├── LICENSE                            # Apache 2.0 License
│
├── data/                              # Dataset files
│   ├── combined_truth_dataset.parquet # V3 Gold Standard (12k rows) ⭐
│   ├── processed_for_ml_testing.parquet # Ready-to-use V3 features
│   └── Season 2 Samples...parquet     # Legacy Season 2 data
│
├── scripts/                           # All Python scripts organized by function
│   │
│   ├── data_processing/              # Data loading and feature engineering
│   │   ├── fetch_overture_data.py    # Step 1: Download Overture slices (Supports NYC, SF)
│   │   ├── build_truth_dataset.py    # Step 2: Construct Ground Truth (City-specific)
│   │   ├── merge_cities.py           # Step 3: Combine city datasets ⭐
│   │   └── process_data_v3.py        # Step 4: V3 Feature Engineering ⭐
│   │
│   ├── experiments/                   # Model training and evaluation
│   │   └── experiment_runner_v3.py   # Step 5: V3 Model Training & Eval ⭐
│   │
│   └── archived/                      # Deprecated scripts (V1/V2)
│       ├── process_data_v2.py
│       ├── experiment_runner_v2.py
│       └── ...
│
└── docs/                              # Documentation and analysis summaries
    ├── recommended_features.txt       # Optimal features list
    └── summaries/                     # Historical analysis documents
```

## 🚀 Quick Start for Reviewers

### 1. **Understand the Project** (5 minutes)

- Read **[README.md](../README.md)** - Comprehensive overview with results
- Focus on:
  - **V3 Breakthrough**: 92.87% Accuracy on Overture Truth Dataset.
  - **Process**: How we built the ground truth from Overture Maps.

### 2. **Run the Code** (10 minutes)

```bash
# Setup (one-time)
python -m venv .venv
source .venv/bin/activate
pip install duckdb pandas numpy scikit-learn imbalanced-learn xgboost catboost pyarrow fused geopandas shapely requests tqdm

# Run V3 experiments (using included processed data)
python scripts/experiments/experiment_runner_v3.py -i data/processed_for_ml_testing.parquet
```

### 3. **Explore Key Logic** (10 minutes)

- **Feature Engineering**: Open **[scripts/data_processing/process_data_v3.py](../scripts/data_processing/process_data_v3.py)**
  - See `process_data_v3` function for Recency Bins and Brand Interactions.
- **Ground Truth Logic**: Open **[scripts/data_processing/build_truth_dataset.py](../scripts/data_processing/build_truth_dataset.py)**
  - See how we define "Closed" using Overture's `operating_status` and ID churn.

## 📊 Key Results at a Glance

| Metric                | V2 (Legacy)   | V3 (Current Best) | Improvement      |
| --------------------- | ------------- | ----------------- | ---------------- |
| **Balanced Accuracy** | 70.65%        | **92.87%**        | **+22.22%** 🚀   |
| **Dataset**           | Season 2 (3k) | Overture (12k)    | Larger & Cleaner |

## 🎯 What Makes V3 Special?

1.  **Overture Truth Dataset**: Validated on 12,000 real-world samples.
2.  **Brand Awareness**: Explicitly models that Brands (97% acc) behave differently from Small Businesses.
3.  **Non-Linear Recency**: Uses "Staleness Bins" instead of just raw days.
4.  **Delta Features**: Captures the _change_ in digital footprint over time.

## 🔄 Workflow

```
1. Fetch Overture Data (fetch_overture_data.py)
         ↓
2. Build Truth Dataset (build_truth_dataset.py)
         ↓
3. Feature Engineering (process_data_v3.py)
         ↓
4. Model Training (experiment_runner_v3.py)
         ↓
5. Result: 93% Accuracy! 🎯
```
