# StatusNow — Place Status Classification

Classifies whether a POI is **Open** or **Closed** based on its digital footprint and recency signals from Overture Maps releases.

---

## Latest Model Results (V8 — Chicago + Miami Hold-out)

| Metric | Open | Closed |
|---|---|---|
| **Balanced Accuracy** | **89.29%** | |
| **AUC** | **95.30%** | |
| Precision | 99.2% | 75.9% |
| Recall | 79.5% | 99.0% |

_Evaluated on 46,907 hold-out rows (Chicago + Miami, never seen during training). Threshold = 0.50. Ensemble: CatBoost-A × 0.7 + LightGBM-A × 0.3._

**Top Features (CatBoost-A):**

| Rank | Feature | Importance | Description |
|---|---|---|---|
| 1 | `recency_spread` | 29.8% | Range between oldest and newest source update timestamps |
| 2 | `recency_pca` | 21.8% | PCA of recency metrics (fit on training rows only) |
| 3 | `zombie_score` | 19.7% | Source count / avg staleness — "database purgatory" signal |
| 4 | `identity_change_score` | 12.8% | Sum of name, category, and address changes |
| 5 | `is_brand` | 2.6% | Place matches a known brand chain |
| 6 | `total_digital` | 2.3% | Count of distinct digital presence types |
| 7 | `category_primary` | 2.2% | Business category (CatBoost native encoding) |
| 8 | `consecutive_present` | 1.6% | Longest consecutive run of release appearances |
| 9 | `has_phone` | 1.6% | Phone number present in base snapshot |
| 10 | `releases_seen` | 1.4% | Number of releases this place appeared in before closure |

---

## Current Pipeline (V8 — HQC Labels + Full Leak Audit)

### What's New

**High-Quality Closed (HQC) labels + 60/40 rebalancing**
- Closed label now requires a place to be present in **2 consecutive past releases** and absent in the next — confirmed churners with trajectory history.
- All 142,931 HQC places kept (no cap). Open downsampled globally to 60/40.
- Previous pipeline capped closed at 3,000/pair and discarded 97% of available signal.

**Leak fixes applied on top of V7:**

| Issue | Root Cause | Fix |
| :---- | :--------- | :-- |
| PCA fitted on full dataset | `recency_pca` was computed before train/test split; hold-out data influenced PCA direction | PCA now fit on training rows only in step3 (after split); `days_latest`/`days_avg` passed as passthrough columns |
| Hold-out used for optimisation | Ensemble weights + threshold searched over 918 combinations against `y_test`, then reported as accuracy | Weights and threshold now chosen via OOF (`cross_val_predict` on `y_train`); hold-out used only for final unbiased reporting |
| Single reference date across pairs | Staleness computed against the newest release date for all rows; pair-0 places appeared ~28 days older than pair-1 with identical update dates | Recency computed per `release_date_current` group so each pair uses its own prediction-window endpoint |
| Digital presence used post-event values | `has_website`, `num_socials`, etc. used COALESCED (R_{i+1}) data for open places but R_i data for churned places — asymmetric measurement window | All presence features now use `base_*` columns (R_i) for both classes |
| LightGBM missing category feature | LGBM received numeric features only, missing the 7.7%-importance `category_primary` | `LabelEncoder` fitted on training rows; LGBM receives `category_encoded` |

**V7 leak fixes (still applied):**

| Issue | Fix |
| :---- | :-- |
| Double-encoded JSON zeroed digital presence, sources, recency | `CAST(AS VARCHAR)` in step1 SQL |
| `releases_seen=2` was a proxy for `label=0` | Anchor both future churners AND equal-size stable-open sample in pair-0 |
| COALESCE-induced staleness asymmetry | Staleness from `base_sources` for all places |
| Overture `confidence` signal | Removed; 5 confidence-derived features dropped |

**V5 leak fixes (still applied):**

| Issue | Fix |
| :---- | :-- |
| `confidence` NaN-fill → perfect closed signal | Use `base_confidence` only; drop `delta_confidence`, `confidence_momentum` |
| `category_churn_risk` computed globally from all labels | Removed; `category_primary` passed as CatBoost native categorical |

---

## Contributor Pipeline

Drop Overture release parquets into `overture_releases/` (see `overture_releases/README.md` for naming convention) then run:

```bash
# Build release files from raw per-city parquets (one-time setup)
python scripts/data_processing/build_release_files.py

# Run the full pipeline: data → features → training
# Default holdout: Chicago + Miami; default balance: 60% open / 40% closed
python pipeline/run_pipeline.py

# Trained models → pipeline_output/models/
```

**Key options:**

```bash
python pipeline/run_pipeline.py \
  --holdout-cities chicago miami \
  --target-open-rate 0.6 \
  --cv-folds 5
```

With 3+ releases the pipeline activates trajectory features (`pre_closure_loss`, `social_trend`, `releases_seen`, `consecutive_present`) that capture pre-closure behaviour — directly addressing the 2-release limitation where all delta features are 0 for churned places by construction.

See [`pipeline/README.md`](pipeline/README.md) for the full guide.

---

## V6 Agent Layer

Sits on top of the classifier and researches low-confidence predictions (default threshold: 0.65) via targeted web search + LLM verdict.

| Mode | Script | Use Case |
|---|---|---|
| Sync (interactive) | `scripts/agent/main.py` | Approval-gated: review the research plan before execution |
| Async (high-throughput) | `scripts/agent/async_main.py` | 3 parallel research workers + live dashboard |

Requires `GROQ_API_KEY` and `TAVILY_API_KEY`. See [`docs/v6_agent_architecture.md`](docs/v6_agent_architecture.md).

---

## Repository Structure

```
StatusNow/
├── overture_releases/           ← Drop Overture parquet releases here
│   └── README.md
│
├── pipeline/                    ← Training pipeline (start here)
│   ├── run_pipeline.py          ← Single command to train a new model
│   ├── step1_build_training_data.py
│   ├── step2_feature_engineering.py
│   ├── step3_train.py
│   └── README.md
│
├── scripts/
│   ├── data_processing/
│   │   ├── build_release_files.py       ← Build overture_releases/ parquets
│   │   ├── fetch_overture_expanded.py   ← Fetch any city from Overture S3
│   │   ├── build_truth_expanded.py      ← Build + merge multi-city truth datasets
│   │   └── merge_cities.py
│   │
│   ├── experiments/
│   │   ├── v5_train_best.py             ← Train best model, export predictions
│   │   ├── v5_full_benchmark.py         ← Full CV + all models + ensemble search
│   │   ├── v6_enrichment_experiment.py
│   │   └── exp_predictive_labels.py     ← R2-oracle experiment (see below)
│   │
│   ├── agent/                           ← V6 AI agent layer
│   │   ├── main.py
│   │   ├── async_main.py
│   │   ├── config.py
│   │   ├── llm/interface.py
│   │   ├── ingest.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── schemas.py
│   │
│   ├── research/                        ← Research history (V3 → V5)
│   │   ├── README.md
│   │   ├── v5_holdout_eval.py
│   │   ├── process_data_v5.py
│   │   └── ...
│   │
│   └── archived/                        ← V1/V2 era scripts
│
└── data/
    └── combined_truth_dataset_expanded.parquet   ← V4 gold standard (123k rows, 12 cities)
```

---

## Project History & Journey Summary

This section chronicles our progress from the initial baseline to the current pipeline.

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
- **Data Architecture Insight**: In the 2-release dataset (`Jan 2026 = base`, `Feb 2026 = current`), churned places (closed by disappearing) have `current = COALESCE(null, prev) = prev`, so **all delta features are 0 by construction** for 93.7% of closed places. This is a structural limitation of 2-release data. A 3rd release would provide legitimate pre-closure deltas.
- **Operating Status Note**: `operating_status = 'closed'` appears in only 1–2 places per city in current Overture data. Closures are expressed as **churning** (disappearance between releases), not explicit status flags. Using operating_status alone as the closed label is not viable with current Overture data.
- **Results**: CB+LGBM ensemble on Chicago + Miami hold-out: **89.41%** (w_CB=0.7, thresh=0.52).
- **Scripts**: `scripts/research/process_data_v5.py`, `scripts/research/v5_holdout_eval.py`.

### Phase 8: V7 — 3rd Release, Trajectory Features, Full Leak Audit (Mar 2026)

- **Goal**: Break the 2-release structural ceiling (all delta features = 0 for churned places) and fix remaining data leaks.
- **3rd Release**: Added Overture `2026-03-18.0` for all 12 cities via `scripts/data_processing/build_release_files.py`. With 3 releases → 2 consecutive comparison pairs → trajectory features activated.
- **Leak Fixes**:
  1. **Double-encoded JSON** (`to_json()` on VARCHAR columns): all digital presence, sources, and recency features were silently zeroed out. Fix: `CAST(AS VARCHAR)` in step1 SQL.
  2. **Constructed `releases_seen` leak**: only future churners were force-included in pair 0's open set, making `releases_seen=2` a near-perfect proxy for `label=0`. Fix: also anchor a matching sample of future non-churners so `releases_seen=2` occurs for both classes.
  3. **COALESCE-induced staleness leak**: `log_days` was computed from the COALESCED `sources` column. Closed places (sources from prior release) appeared more stale than open places (sources from current release) by construction. Fix: compute staleness from `base_sources` for all places.
  4. **Overture `confidence` removed**: 5 confidence-derived features dropped (external quality signal with unclear provenance).
- **City column propagated**: `_city` from release parquets flows through step1 → step2 → step3, enabling city-name holdout (default: Chicago + Miami).
- **Results**: CatBoost-C: CV **97.15%**, hold-out **97.00%**. Top features: `recency_spread` (19.6%), `zombie_score` (16.7%), `recency_pca` (11.6%), `log_days` (9.3%).
- **Key Insight**: The 89.41% V5 result was partially suppressed by silently zeroed features (the JSON double-encoding bug was present from the start). The true signal in Overture recency metadata is much stronger than previously measured.

### Phase 10: R2-Oracle Experiment — Genuine Signal Validation (Mar 2026)

- **Question**: Is the model learning real closure signals, or just detecting "this place is absent from the latest release?"
- **Setup**: Features built from R0→R1 window only (Jan→Feb). R2 (Mar) used **exclusively as a label oracle** — label=1 if present in R2, label=0 (HQC) if in R0+R1 but not R2. R2 data never touches the feature matrix.
- **Result**:

| Metric | Current Pipeline (R2 in features) | Experiment (R2 labels only) | Delta |
|---|---|---|---|
| Balanced Accuracy | **89.29%** | 71.02% | −18.3 pp |
| AUC | **95.30%** | 80.28% | −15.0 pp |

- **Conclusion**: The significant drop confirms the model is **not** simply memorising R2 presence. Genuine predictive signals exist in the R0→R1 feature window (name changes, digital presence shifts, source volatility, recency). The additional ~18 pp in the current pipeline comes from the multi-release feature window giving the model more temporal evidence — not from target leakage. Script: `scripts/experiments/exp_predictive_labels.py`.

---

### Phase 9: V8 — HQC Labels + Remaining Leak Fixes (Mar 2026, Current)

- **Goal**: Tighten the closed label definition, fix remaining leaks found in a full audit, and improve dataset balance.
- **HQC Closed Labels**: Redefined closed as places present in **2 consecutive past releases and absent in the next** — confirmed churners with trajectory history. This yields 142,931 high-quality closed examples vs. the old 3,000/pair cap that discarded 97% of available signal. Dataset rebalanced globally to 60/40 (357k rows total).
- **Leak Fixes**:
  1. **PCA fitted on full dataset**: `recency_pca` was computed before the train/test split, so hold-out data influenced the PCA direction. Fix: PCA now fit on training rows only in step3 after the split; `days_latest`/`days_avg` passed as passthrough columns from step2.
  2. **Hold-out used for optimisation**: ensemble weights and threshold were searched over 918 combinations against `y_test`, then reported as the hold-out accuracy — a form of test-set overfitting. Fix: weights and threshold now chosen via OOF predictions (`cross_val_predict` on `y_train`); hold-out used only for final unbiased reporting.
  3. **Single reference date across pairs**: staleness was computed against the newest release date for all rows, making pair-0 places appear ~28 days older than pair-1 places with identical update dates. Fix: recency computed per `release_date_current` group.
  4. **Digital presence used post-event values**: `has_website`, `num_socials`, etc. used COALESCED (R_{i+1}) data for open places but R_i data for churned places — an asymmetric measurement window. Fix: all presence features now use `base_*` columns (R_i) for both classes.
  5. **LightGBM missing category feature**: LGBM received numeric features only, missing the 7.7%-importance `category_primary`. Fix: `LabelEncoder` fitted on training rows; LGBM receives `category_encoded`.
