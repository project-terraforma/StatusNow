# StatusNow — Place Status Classification

Classifies whether a POI is **Open** or **Closed** based on its digital footprint and recency signals from Overture Maps releases.

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
│   │   └── v6_enrichment_experiment.py
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

## Project History

### V1 — Delta Features (Baseline)
`has_gained_social` (r=+0.26) was the strongest single predictor. `has_any_loss` was a reliable closure signal. **67.3% Balanced Accuracy.**

### V2 — Interaction Features
Added zombie score (high sources + stale = database purgatory), category churn risk, and PCA over recency. **70.65%.**

### V3 — Label Refinement
Dynamic label refinement via 5-fold CV flagged 65 samples (2.2%) as likely human-labelling errors. Removing them: **72.09%.**

### V4 — Overture Truth Dataset + 12-City Scale
Replaced manually-labelled data with Overture churn labels (presence diff between Jan and Feb 2026 releases). Expanded from NYC to 12 cities (123k samples). Data scale dominated — 12k→123k gained +8.7 pp; HPO gained ~0.1 pp. **89.18% CV.**

### V5 — Full Leakage Audit + Geographic Hold-Out
- `confidence` NaN-fill removed (93.7% of closed had null→0, near-perfect proxy).
- `category_churn_risk` removed (globally computed from all labels, leaked into CV folds).
- Chicago + Miami held out for honest geographic evaluation.
- Result: **89.41% hold-out balanced accuracy** (CB+LGBM, w=0.7, t=0.52).

### V7 — 3rd Release + Trajectory Features
- Added March 2026 Overture release → 2 consecutive pairs → trajectory features activated (`releases_seen`, `consecutive_present`, `pre_closure_loss`, `social_trend`, `website_trend`).
- Fixed double-encoded JSON bug that was silently zeroing all digital presence and recency features.
- Fixed `releases_seen` construction leak.
- Fixed COALESCE-induced staleness asymmetry.
- Result: **97% hold-out balanced accuracy** (note: subsequent audit found remaining leaks in PCA and hold-out optimisation — see V8).

### V8 — HQC Labels + Remaining Leak Fixes (Current)
- Closed labels tightened to HQC definition (2-release presence, gone in third). 142,931 confirmed churners vs old 3k cap.
- Dataset rebalanced to 60/40 globally; 357k rows total.
- PCA fitted on training rows only.
- Ensemble search moved to OOF predictions; hold-out is now a clean evaluation.
- Per-pair reference date for staleness; digital presence from base snapshot for all places.
- LightGBM now receives encoded category feature.
