"""
Run the Full Training Pipeline
================================

Single entry-point that orchestrates all three pipeline steps:
  1. Build training data from Overture release parquets
  2. Engineer features (multi-release aware, leak-free)
  3. Train the CB+LGBM ensemble and save model artifacts

Usage
-----
  python pipeline/run_pipeline.py [options]

Quick start (minimum setup):
  1. Drop your Overture release parquet files into overture_releases/
     (see overture_releases/README.md for the naming convention)
  2. Run:  python pipeline/run_pipeline.py
  3. Your trained model will be in pipeline_output/models/

Options
-------
  --releases-dir   DIR       Overture release input folder  (default: overture_releases/)
  --output-dir     DIR       Pipeline output folder         (default: pipeline_output/)
  --holdout-cities DATE ...  Hold out these release dates for evaluation
                              e.g. --holdout-cities 2026-02-18.0
  --max-open       N         Max open samples per release pair (default: 9000)
  --max-closed     N         Max closed samples per release pair (default: 3000)
  --no-downsample            Use all rows (may produce very large datasets)
  --cv-folds       N         Cross-validation folds (default: 5)
  --seed           N         Random seed (default: 42)
  --skip-step1               Skip data building (use existing 01_training_data_raw.parquet)
  --skip-step2               Skip feature engineering (use existing 02_features.parquet)
  --skip-step3               Skip training (only run steps 1 and/or 2)
"""

import argparse
import os
import sys
import time


def _banner(title: str) -> None:
    width = 70
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    pad = (width - 2 - len(title)) // 2
    print("║" + " " * pad + title + " " * (width - 2 - pad - len(title)) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "StatusNow — Overture Multi-Release Training Pipeline\n"
            "======================================================\n"
            "Drop Overture release parquets into overture_releases/ and "
            "run this script to train the CB+LGBM open/closed classifier."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required directories
    parser.add_argument(
        "--releases-dir", default="overture_releases",
        help="Folder containing Overture parquet release files (default: overture_releases/)",
    )
    parser.add_argument(
        "--output-dir", default="pipeline_output",
        help="Root folder for all pipeline outputs (default: pipeline_output/)",
    )
    # Step 1 options
    parser.add_argument(
        "--max-open", type=int, default=9000,
        help="Max open-labelled rows per release pair (default: 9000)",
    )
    parser.add_argument(
        "--max-closed", type=int, default=3000,
        help="Max closed-labelled rows per release pair (default: 3000)",
    )
    parser.add_argument(
        "--no-downsample", action="store_true",
        help="Disable downsampling (use all rows)",
    )
    # Step 3 options
    parser.add_argument(
        "--holdout-cities", nargs="+", default=None, metavar="DATE",
        help=(
            "release_date_current values to hold out for honest geographic evaluation. "
            "Example: --holdout-cities 2026-02-18.0"
        ),
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    # Skip flags
    parser.add_argument(
        "--skip-step1", action="store_true",
        help="Skip step 1 — use existing pipeline_output/01_training_data_raw.parquet",
    )
    parser.add_argument(
        "--skip-step2", action="store_true",
        help="Skip step 2 — use existing pipeline_output/02_features.parquet",
    )
    parser.add_argument(
        "--skip-step3", action="store_true",
        help="Skip step 3 — only run steps 1 and/or 2 (no training)",
    )

    args = parser.parse_args()

    raw_path     = os.path.join(args.output_dir, "01_training_data_raw.parquet")
    feats_path   = os.path.join(args.output_dir, "02_features.parquet")
    models_dir   = os.path.join(args.output_dir, "models")
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure we can import from the pipeline package
    project_root = os.path.dirname(pipeline_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    _banner("StatusNow — Overture Multi-Release Training Pipeline")
    print(f"\n  Releases folder : {args.releases_dir}/")
    print(f"  Output folder   : {args.output_dir}/")
    if args.holdout_cities:
        print(f"  Hold-out cities : {args.holdout_cities}")
    print()

    t_total = time.time()

    # ─── Step 1: Build training data ─────────────────────────────────────────
    if args.skip_step1:
        print(f"⏭️  Skipping Step 1 (using existing {raw_path})")
        if not os.path.exists(raw_path):
            print(f"❌  {raw_path} not found. Run without --skip-step1 first.")
            sys.exit(1)
    else:
        from pipeline.step1_build_training_data import build_training_data
        t0 = time.time()
        build_training_data(
            releases_dir=args.releases_dir,
            output_dir=args.output_dir,
            max_open=args.max_open,
            max_closed=args.max_closed,
            no_downsample=args.no_downsample,
        )
        print(f"\n  ⏱️  Step 1 completed in {time.time() - t0:.1f}s")

    # ─── Step 2: Feature engineering ─────────────────────────────────────────
    if args.skip_step2:
        print(f"\n⏭️  Skipping Step 2 (using existing {feats_path})")
        if not os.path.exists(feats_path):
            print(f"❌  {feats_path} not found. Run without --skip-step2 first.")
            sys.exit(1)
    else:
        from pipeline.step2_feature_engineering import build_features
        t0 = time.time()
        build_features(input_path=raw_path, output_path=feats_path)
        print(f"\n  ⏱️  Step 2 completed in {time.time() - t0:.1f}s")

    # ─── Step 3: Train ───────────────────────────────────────────────────────
    if args.skip_step3:
        print(f"\n⏭️  Skipping Step 3 (training). No model was trained.")
    else:
        from pipeline.step3_train import train
        t0 = time.time()
        train(
            input_path=feats_path,
            output_dir=models_dir,
            holdout_cities=args.holdout_cities,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        print(f"\n  ⏱️  Step 3 completed in {time.time() - t0:.1f}s")

    # ─── Done ─────────────────────────────────────────────────────────────────
    _banner("Pipeline Complete")
    print(f"\n  Total time: {time.time() - t_total:.1f}s")
    print(f"\n  Outputs written to: {args.output_dir}/")
    print(f"    {raw_path}")
    print(f"    {feats_path}")
    if not args.skip_step3:
        print(f"    {models_dir}/")
    print()


if __name__ == "__main__":
    main()
