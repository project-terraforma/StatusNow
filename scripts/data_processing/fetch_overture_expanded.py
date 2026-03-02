"""
Expanded data fetcher — fetch Overture data for multiple cities.
Adds Chicago, Los Angeles, Houston, Phoenix, Philadelphia to existing NYC and SF.
"""

import duckdb
import os
import argparse

OVERTURE_S3_BUCKET = "s3://overturemaps-us-west-2/release"

# Bounding boxes for major US cities
BBOXES = {
    "nyc": "-74.2591,40.4774,-73.7003,40.9176",
    "sf": "-122.52,37.70,-122.35,37.84",
    "chicago": "-87.9401,41.6445,-87.5240,42.0230",
    "la": "-118.6682,33.7037,-118.1553,34.3373",
    "houston": "-95.7880,29.5228,-95.0143,30.1106",
    "phoenix": "-112.3240,33.2900,-111.9261,33.7694",
    "philly": "-75.2803,39.8670,-75.0000,40.1380",
    "seattle": "-122.4597,47.4910,-122.2244,47.7341",
    "denver": "-105.1098,39.6143,-104.5997,39.9142",
    "boston": "-71.1912,42.2279,-70.9229,42.3969",
    "miami": "-80.3195,25.7093,-80.1309,25.8557",
    "atlanta": "-84.5512,33.6480,-84.2893,33.8870",
}

def fetch_overture_data(city, release_date, output_path):
    bbox_str = BBOXES.get(city)
    if not bbox_str:
        raise ValueError(f"Unknown city: {city}. Available: {list(BBOXES.keys())}")
    
    xmin, ymin, xmax, ymax = map(float, bbox_str.split(","))
    
    print(f"Fetching data for {city.upper()} ({release_date})...")
    print(f"BBox: {bbox_str}")

    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    query = f"""
        SELECT 
            id,
            to_json(names) as names,
            to_json(categories) as categories,
            confidence,
            to_json(websites) as websites,
            to_json(socials) as socials,
            to_json(emails) as emails,
            to_json(phones) as phones,
            to_json(brand) as brand,
            to_json(addresses) as addresses,
            geometry,
            to_json(sources) as sources,
            try_cast(operating_status AS VARCHAR) as operating_status 
        FROM read_parquet('{OVERTURE_S3_BUCKET}/{release_date}/theme=places/type=place/*', filename=true, hive_partitioning=1)
        WHERE bbox.xmin >= {xmin} AND bbox.ymin >= {ymin} AND bbox.xmax <= {xmax} AND bbox.ymax <= {ymax}
    """
    
    print("Executing DuckDB query (streaming from S3)...")
    try:
        df = con.execute(query).df()
        print(f"  Fetched {len(df)} rows")
        df.to_parquet(output_path)
        print(f"✅ Saved to {output_path}")
        return len(df)
    except Exception as e:
        print(f"❌ Error fetching {release_date}: {e}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Overture Data for Multiple Cities.")
    parser.add_argument("--cities", type=str, nargs="+", 
                        default=["chicago", "la", "houston", "phoenix", "philly"],
                        help=f"Cities to fetch. Available: {list(BBOXES.keys())}")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cities that already have data")
    args = parser.parse_args()

    CURRENT_RELEASE = "2026-02-18.0"
    PREVIOUS_RELEASE = "2026-01-21.0"
    output_dir = "data/overture"
    os.makedirs(output_dir, exist_ok=True)

    for city in args.cities:
        current_file = f"{output_dir}/{city}_current.parquet"
        previous_file = f"{output_dir}/{city}_previous.parquet"
        
        if args.skip_existing and os.path.exists(current_file) and os.path.exists(previous_file):
            print(f"\n⏭️  Skipping {city.upper()} (already exists)")
            continue
            
        print(f"\n{'='*60}")
        print(f"  FETCHING {city.upper()}")
        print(f"{'='*60}")
        
        fetch_overture_data(city, CURRENT_RELEASE, current_file)
        fetch_overture_data(city, PREVIOUS_RELEASE, previous_file)
    
    print("\n✅ All cities fetched!")
