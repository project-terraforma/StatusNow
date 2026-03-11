# `scripts/research/` — Research & Iteration History

This folder contains the scripts used during the research phase of the project.  
They are preserved as a reference for how the model evolved from V3 → V4 → V5.

**You do not need these to train a new model.** Use the `pipeline/` folder instead.

---

## Contents

| Script                    | Phase | What it did                                                                    |
| ------------------------- | ----- | ------------------------------------------------------------------------------ |
| `experiment_runner_v3.py` | V3    | Label refinement + brand stratification experiments                            |
| `process_data_v3.py`      | V3    | Feature engineering for the V3 model (52 features)                             |
| `fetch_overture_data.py`  | V3/V4 | DuckDB fetch for 2 hardcoded releases (NYC/SF only)                            |
| `build_truth_dataset.py`  | V3/V4 | Truth dataset construction for a single city pair                              |
| `v4_research.py`          | V4    | Full V4 research pipeline: HPO, 12-city expansion, ensemble search             |
| `improve_v4.py`           | V4    | Further V4 iteration experiments                                               |
| `v5_holdout_eval.py`      | V5    | **Best result: 89.41% balanced accuracy** on Chicago + Miami hold-out          |
| `process_data_v5.py`      | V5    | Leak-free feature pipeline (basis for `pipeline/step2_feature_engineering.py`) |

## Key Results

- **V3**: 85.21% (leaky CV, NYC + SF, 18.6k samples)
- **V4**: 89.18% (leaky CV, 12 cities, 123k samples)
- **V5**: **89.41%** (honest geographic hold-out, Chicago + Miami, leak-free)

See the main [README.md](../../README.md) for the full research history.
