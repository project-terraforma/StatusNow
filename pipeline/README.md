# StatusNow — Overture Multi-Release Training Pipeline

This pipeline trains an open/closed place classifier from Overture Maps historical release data.  
It is designed to be **simple to run** and **easy to understand** — the code is documented step-by-step.

---

## Quick Start

```bash
# 1. Drop your Overture release parquets into the releases folder
#    (see overture_releases/README.md for file naming convention)
ls overture_releases/
# → 2025-12-16.0_places.parquet
# → 2026-01-21.0_places.parquet
# → 2026-02-18.0_places.parquet

# 2. Run the full pipeline
python pipeline/run_pipeline.py

# 3. Your trained model is ready
ls pipeline_output/models/
# → catboost_a.cbm  catboost_b.cbm  catboost_c.cbm
# → lightgbm_a.pkl
# → ensemble_config.json
```

---

## Folder Structure

```
StatusNow/
├── overture_releases/       ← DROP PARQUET FILES HERE
│   └── README.md
│
├── pipeline/                ← Pipeline source code
│   ├── run_pipeline.py      ← Single entry-point (run this)
│   ├── step1_build_training_data.py
│   ├── step2_feature_engineering.py
│   ├── step3_train.py
│   └── README.md            ← You are here
│
└── pipeline_output/         ← Created automatically
    ├── 01_training_data_raw.parquet
    ├── 02_features.parquet
    └── models/
        ├── catboost_a.cbm
        ├── catboost_b.cbm
        ├── catboost_c.cbm
        ├── lightgbm_a.pkl
        └── ensemble_config.json
```

---

## How It Works

### The Core Idea

Overture Maps publishes place data monthly. When a place disappears between two consecutive releases, it has almost certainly closed. We use this signal — combined with shifts in digital footprint (websites, socials, phones) — to train a classifier.

```
Release R1 (Dec 2025) ─┐
                        ├─ Compare → labelled rows (Open / Closed)
Release R2 (Jan 2026) ─┘
     │
     └─ Also compared with R3 ↓

Release R2 (Jan 2026) ─┐
                        ├─ Compare → more labelled rows
Release R3 (Feb 2026) ─┘
```

With **N releases, you get N-1 comparison windows**. Each window contributes training rows with their own "before" and "after" snapshots.

### Labels

| Situation                                          | Label                     |
| -------------------------------------------------- | ------------------------- |
| Place in R*i, **missing** in R*{i+1}               | **0 — Closed** (churned)  |
| Place in R\_{i+1} with `operating_status='closed'` | **0 — Closed** (explicit) |
| Place in R\_{i+1}, not closed                      | **1 — Open**              |

### Why More Releases Matter

With only 2 releases, churned (closed) places have all delta features equal to 0 — because the JOIN fills the missing current data from the baseline. This is a structural limitation.

With **3+ releases**, the pipeline activates **trajectory features**:

| Feature               | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| `releases_seen`       | How many releases this place appeared in                                     |
| `consecutive_present` | Longest streak of consecutive appearances                                    |
| `pre_closure_loss`    | Did the place lose social/website/phone in the window BEFORE it disappeared? |
| `social_trend`        | Slope of social count over multiple windows                                  |
| `website_trend`       | Slope of website count over multiple windows                                 |

These features are the **true leading indicators** of closure and can significantly improve accuracy beyond the 89.41% V5 baseline.

---

## Pipeline Steps in Detail

### Step 1 — Build Training Data (`step1_build_training_data.py`)

- Discovers all `.parquet` files in `overture_releases/` sorted by date prefix
- For each consecutive pair (R*i → R*{i+1}), runs a full outer join on place `id`
- Labels each row as Open (1) or Closed (0)
- Optionally downsamples to balance class ratio within each pair
- Adds metadata: `release_pair`, `release_index`, `release_date_base`, `release_date_current`
- **Output**: `pipeline_output/01_training_data_raw.parquet`

```bash
python pipeline/step1_build_training_data.py \
    --releases-dir overture_releases/ \
    --output-dir pipeline_output/ \
    --max-open 9000 \
    --max-closed 3000
```

### Step 2 — Feature Engineering (`step2_feature_engineering.py`)

Builds the full feature set (63+ features, all V5 leak-free):

- **Confidence**: `base_confidence` only (no leaky delta confidence)
- **Digital presence**: website, social, phone counts and flags
- **Delta features**: changes between base and current snapshot
- **Identity changes**: name, category, website domain, address shifts
- **Recency/staleness**: `log_days`, staleness flags, recency bucket
- **Interaction features**: zombie_score, decay_velocity, brand×stale, etc.
- **Multi-release trajectory** (3+ releases only): `pre_closure_loss`, trends, etc.

**Output**: `pipeline_output/02_features.parquet`

```bash
python pipeline/step2_feature_engineering.py \
    --input pipeline_output/01_training_data_raw.parquet \
    --output pipeline_output/02_features.parquet
```

### Step 3 — Train (`step3_train.py`)

Trains the ensemble:

- **CatBoost-A** (2000 iterations, depth 8, lr=0.03)
- **CatBoost-B** (1500 iterations, depth 7, lr=0.05)
- **CatBoost-C** (1000 iterations, depth 6, lr=0.05)
- **LightGBM-A** (1500 estimators, depth 8)

Runs 5-fold stratified cross-validation for model selection, then searches for the best ensemble weights and decision threshold.

**Output**: `pipeline_output/models/` with all model files + `ensemble_config.json`

```bash
python pipeline/step3_train.py \
    --input pipeline_output/02_features.parquet \
    --output-dir pipeline_output/models/ \
    # optionally hold out a release for geographic evaluation:
    --holdout-cities 2026-02-18.0
```

---

## All Options

```
python pipeline/run_pipeline.py [options]

  --releases-dir   DIR       Input releases folder     (default: overture_releases/)
  --output-dir     DIR       Output folder             (default: pipeline_output/)
  --max-open       N         Max open rows per pair    (default: 9000)
  --max-closed     N         Max closed rows per pair  (default: 3000)
  --no-downsample            Use all rows
  --holdout-cities DATE ...  Hold-out release date(s) for evaluation
  --cv-folds       N         CV folds                  (default: 5)
  --seed           N         Random seed               (default: 42)
  --skip-step1               Skip data building step
  --skip-step2               Skip feature engineering step
  --skip-step3               Skip training step
```

---

## Using the Trained Model for Inference

The `ensemble_config.json` file contains everything needed to reproduce predictions:

```python
import json, pickle
import numpy as np
from catboost import CatBoostClassifier

with open("pipeline_output/models/ensemble_config.json") as f:
    cfg = json.load(f)

# Load models
cb = CatBoostClassifier()
cb.load_model("pipeline_output/models/catboost_c.cbm")

with open("pipeline_output/models/lightgbm_a.pkl", "rb") as f:
    lgbm = pickle.load(f)

# Predict on new data (X must have the same feature columns)
ens_cfg   = cfg["holdout_ensemble"]
w_cb      = ens_cfg["w_cb"]
w_lgbm    = ens_cfg["w_lgbm"]
threshold = ens_cfg["threshold"]

prob_cb   = cb.predict_proba(X_new)[:, 1]
prob_lgbm = lgbm.predict_proba(X_new_num)[:, 1]

prob_ensemble = w_cb * prob_cb + w_lgbm * prob_lgbm
predictions   = (prob_ensemble >= threshold).astype(int)  # 1=Open, 0=Closed
```

---

## Known Limitations

1. **2-release delta=0 problem**: When only 2 releases are provided, all churned closed places have delta features equal to 0 by construction (COALESCE fills from base). The model still reaches ~89% accuracy using confidence and recency signals, but the delta features carry no signal for 94% of closed places. **Solution: provide 3+ releases.**

2. **`operating_status` is sparse**: In current Overture data, only 1-2 places per city have `operating_status='closed'` explicitly. Closures are expressed as disappearance, not status flags. This is expected.

3. **Geographic coverage**: The model was originally validated on 12 US cities. Performance on non-US locations is unknown.
