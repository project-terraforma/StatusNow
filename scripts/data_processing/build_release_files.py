"""
One-time setup script: build all 3 release parquets in overture_releases/.

  1. Reads data/overture/{city}_previous.parquet (2026-01-21.0) for all 12 cities
     → combines → overture_releases/2026-01-21.0_places.parquet

  2. Reads data/overture/{city}_current.parquet (2026-02-18.0) for all 12 cities
     → combines → overture_releases/2026-02-18.0_places.parquet

  3. Fetches 2026-03-18.0 for all 12 cities from S3
     → combines → overture_releases/2026-03-18.0_places.parquet

Usage:
    python scripts/data_processing/build_release_files.py
    python scripts/data_processing/build_release_files.py --skip-existing
    python scripts/data_processing/build_release_files.py --only-release 2026-03-18.0
"""

import argparse
import os
import duckdb
import pandas as pd

OVERTURE_S3_BUCKET = "s3://overturemaps-us-west-2/release"

CITIES = [
    "nyc", "sf", "chicago", "la", "houston", "phoenix",
    "philly", "seattle", "denver", "boston", "miami", "atlanta",
]

BBOXES = {
    "nyc":     "-74.2591,40.4774,-73.7003,40.9176",
    "sf":      "-122.52,37.70,-122.35,37.84",
    "chicago": "-87.9401,41.6445,-87.5240,42.0230",
    "la":      "-118.6682,33.7037,-118.1553,34.3373",
    "houston": "-95.7880,29.5228,-95.0143,30.1106",
    "phoenix": "-112.3240,33.2900,-111.9261,33.7694",
    "philly":  "-75.2803,39.8670,-75.0000,40.1380",
    "seattle": "-122.4597,47.4910,-122.2244,47.7341",
    "denver":  "-105.1098,39.6143,-104.5997,39.9142",
    "boston":  "-71.1912,42.2279,-70.9229,42.3969",
    "miami":   "-80.3195,25.7093,-80.1309,25.8557",
    "atlanta": "-84.5512,33.6480,-84.2893,33.8870",
}

OUTPUT_DIR = "overture_releases"
DATA_DIR   = "data/overture"


def build_from_local(release_date: str, suffix: str, output_path: str) -> None:
    """Combine per-city local parquets into a single release file."""
    frames = []
    missing = []
    for city in CITIES:
        path = os.path.join(DATA_DIR, f"{city}_{suffix}.parquet")
        if not os.path.exists(path):
            missing.append(path)
            continue
        df = pd.read_parquet(path)
        df["_city"] = city
        frames.append(df)
        print(f"  ✓ {city}: {len(df):,} rows")

    if missing:
        print(f"  ⚠ Missing files (skipped): {missing}")

    if not frames:
        raise FileNotFoundError(f"No local files found for release {release_date}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"  ✅ Saved {len(combined):,} rows → {output_path}\n")


def fetch_from_s3(release_date: str, output_path: str) -> None:
    """Fetch a new release from S3 for all 12 cities and combine."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    s3_path = f"{OVERTURE_S3_BUCKET}/{release_date}/theme=places/type=place/*"
    frames = []

    for city in CITIES:
        bbox_str = BBOXES[city]
        xmin, ymin, xmax, ymax = map(float, bbox_str.split(","))

        query = f"""
            SELECT
                id,
                to_json(names)      AS names,
                to_json(categories) AS categories,
                confidence,
                to_json(websites)   AS websites,
                to_json(socials)    AS socials,
                to_json(emails)     AS emails,
                to_json(phones)     AS phones,
                to_json(brand)      AS brand,
                to_json(addresses)  AS addresses,
                geometry,
                to_json(sources)    AS sources,
                try_cast(operating_status AS VARCHAR) AS operating_status
            FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
            WHERE bbox.xmin >= {xmin} AND bbox.ymin >= {ymin}
              AND bbox.xmax <= {xmax} AND bbox.ymax <= {ymax}
        """

        print(f"  Fetching {city.upper()} from S3 …")
        try:
            df = con.execute(query).df()
            df["_city"] = city
            frames.append(df)
            print(f"    → {len(df):,} rows")
        except Exception as e:
            print(f"    ❌ Error for {city}: {e}")

    if not frames:
        raise RuntimeError("S3 fetch returned no data for any city")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"  ✅ Saved {len(combined):,} rows → {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Build overture_releases/ parquets")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a release if the output file already exists")
    parser.add_argument("--only-release", type=str, default=None,
                        help="Build only this release (e.g. 2026-03-18.0)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    releases = [
        ("2026-01-21.0", "local", "previous"),
        ("2026-02-18.0", "local", "current"),
        ("2026-03-18.0", "s3",    None),
    ]

    if args.only_release:
        releases = [r for r in releases if r[0] == args.only_release]
        if not releases:
            print(f"Unknown release: {args.only_release}")
            return

    for release_date, source, suffix in releases:
        output_path = os.path.join(OUTPUT_DIR, f"{release_date}_places.parquet")
        print(f"\n{'='*60}")
        print(f"  {release_date}  (source: {source})")
        print(f"{'='*60}")

        if args.skip_existing and os.path.exists(output_path):
            print(f"  ⏭  Already exists, skipping.")
            continue

        if source == "local":
            build_from_local(release_date, suffix, output_path)
        else:
            fetch_from_s3(release_date, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
