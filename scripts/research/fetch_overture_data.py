import duckdb
import os
import argparse

# Amazon S3 bucket for Overture Maps
OVERTURE_S3_BUCKET = "s3://overturemaps-us-west-2/release"

# Bounding Boxes
BBOXES = {
    "nyc": "-74.2591,40.4774,-73.7003,40.9176",
    # Rough SF BBox: West=-122.52, South=37.70, East=-122.35, North=37.84
    "sf": "-122.52,37.70,-122.35,37.84"
}

def fetch_overture_data(city, release_date, output_path):
    """
    Fetches Overture Places data for a specific bounding box and release.
    """
    bbox_str = BBOXES.get(city)
    if not bbox_str:
        raise ValueError(f"Unknown city: {city}. Available: {list(BBOXES.keys())}")
    
    xmin, ymin, xmax, ymax = map(float, bbox_str.split(","))
    
    print(f"Fetching data for {city.upper()} ({release_date})...")
    print(f"BBox: {bbox_str}")

    con = duckdb.connect()
    
    # Configure S3 (No credentials needed for public bucket)
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Construct Query
    # Construct Query
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
        # Execute and save to Parquet
        con.execute(query).df().to_parquet(output_path)
        print(f"✅ Saved to {output_path}")
    except Exception as e:
        print(f"❌ Error fetching {release_date}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Overture Data for a City.")
    parser.add_argument("--city", type=str, default="nyc", choices=["nyc", "sf"], help="City to fetch (nyc or sf)")
    args = parser.parse_args()

    # Define releases
    CURRENT_RELEASE = "2026-02-18.0"
    PREVIOUS_RELEASE = "2026-01-21.0"

    output_dir = "data/overture"
    os.makedirs(output_dir, exist_ok=True)

    # Fetch Current
    fetch_overture_data(args.city, CURRENT_RELEASE, f"{output_dir}/{args.city}_current.parquet")

    # Fetch Previous
    fetch_overture_data(args.city, PREVIOUS_RELEASE, f"{output_dir}/{args.city}_previous.parquet")
