
import duckdb
import os
import sys

# NYC BBox
# xmin, ymin, xmax, ymax
BBOX = (-74.05, 40.65, -73.90, 40.85)
# Explicitly using the releases we found
RELEASE_CURRENT = "2026-02-18.0"
RELEASE_PREVIOUS = "2026-01-21.0"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_places(release, output_filename):
    print(f"Fetching data for release {release}...")
    
    # Overture S3 path
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    
    # SQL query with BBox filter
    # We select specific columns to keep it lightweight but useful for "closed" logic
    query = f"""
        COPY (
            SELECT 
                id,
                names,
                bbox,
                brand,
                websites,
                socials,
                phones,
                confidence,
                categories,
                addresses,
                operating_status
            FROM read_parquet('{s3_path}')
            WHERE 
                bbox.xmin > {BBOX[0]} AND
                bbox.ymin > {BBOX[1]} AND
                bbox.xmax < {BBOX[2]} AND
                bbox.ymax < {BBOX[3]}
        ) TO '{output_filename}' (FORMAT PARQUET);
    """
    
    print(f"Executing Query for {release} -> {output_filename}")
    
    con = duckdb.connect()
    # Install/Load httpfs for S3 access
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Set S3 region (required for Overture)
    con.execute("SET s3_region='us-west-2';")
    
    try:
        con.execute(query)
        print(f"✅ Application success: {output_filename}")
        
        # Verify count
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_filename}')").fetchone()[0]
        print(f"   Rows fetched: {count}")
        
    except Exception as e:
        print(f"❌ Error fetching {release}: {e}")

if __name__ == "__main__":
    fetch_places(RELEASE_CURRENT, os.path.join(DATA_DIR, "overture_current.parquet"))
    fetch_places(RELEASE_PREVIOUS, os.path.join(DATA_DIR, "overture_previous.parquet"))
