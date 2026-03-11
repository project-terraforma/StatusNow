# `overture_releases/` — Drop Your Overture Parquet Files Here

This folder is the **input** for the training pipeline.  
Place your Overture Maps `places` release parquet files here and the pipeline will handle the rest.

---

## File Naming Convention

Files **must** follow this pattern so the pipeline can sort them into chronological order:

```
YYYY-MM-DD.N_<optional-label>.parquet
```

| Part         | Meaning                                             | Example      |
| ------------ | --------------------------------------------------- | ------------ |
| `YYYY-MM-DD` | Release date                                        | `2026-01-21` |
| `.N`         | Release revision (Overture uses `.0`)               | `.0`         |
| `_<label>`   | Optional human-readable label (ignored by pipeline) | `_places`    |

### Examples

```
2025-11-17.0_places.parquet    ← oldest release
2025-12-16.0_places.parquet
2026-01-21.0_places.parquet
2026-02-18.0_places.parquet    ← newest release
```

The pipeline sorts lexicographically — the date prefix ensures correct ordering automatically.

---

## What Overture Data to Include

Each file should be the **`places`** theme parquet from an Overture release.  
You can download these from the public S3 bucket:

```
s3://overturemaps-us-west-2/release/<YYYY-MM-DD.N>/theme=places/type=place/
```

or filter by bounding box using DuckDB (see `scripts/data_processing/fetch_overture_expanded.py`).

**Required columns** (the pipeline will fail loudly if these are missing):

| Column             | Type        | Notes                                                           |
| ------------------ | ----------- | --------------------------------------------------------------- |
| `id`               | string      | Stable Overture place ID — used to track places across releases |
| `names`            | JSON string | `{"primary": "Starbucks", ...}`                                 |
| `categories`       | JSON string | `{"primary": "coffee_shop", ...}`                               |
| `websites`         | JSON string | list of URL strings                                             |
| `socials`          | JSON string | list of URL strings                                             |
| `phones`           | JSON string | list of phone strings                                           |
| `addresses`        | JSON string | list of address dicts                                           |
| `brand`            | JSON string | brand object or null                                            |
| `sources`          | JSON string | list of source dicts with `update_time`                         |
| `confidence`       | float       | Overture quality score 0–1                                      |
| `operating_status` | string      | `'open'`, `'closed'`, or null                                   |

---

## How Many Releases?

| Releases | What You Get                                                                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2**    | Basic delta features (same as V5 baseline). Note: ~94% of closed places (churned) will have delta=0 by construction — this is a known structural limitation.                          |
| **3+**   | Trajectory features activated: `social_trend`, `website_trend`, `pre_closure_loss`, `consecutive_present`. These are the true leading indicators and substantially improve detection. |

**Recommendation: provide at least 3 releases spanning 2+ months.**

---

## Running the Pipeline

```bash
# From the project root
python pipeline/run_pipeline.py
```

See `pipeline/README.md` for full documentation.
