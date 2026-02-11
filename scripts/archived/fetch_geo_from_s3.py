import duckdb
import pandas as pd
import time

# Configuration
LOCAL_DATA_FILE = "data/Season 2 Samples 3k Project Updated.parquet"
OUTPUT_FILE = "data/season2_with_geometry.parquet"
# S3_PATH = "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*"
S3_PATH = "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*"

def fetch_geometry():
    print(f"Loading local IDs from {LOCAL_DATA_FILE}...")
    local_df = pd.read_parquet(LOCAL_DATA_FILE)
    
    # Get distinct IDs needed
    ids_to_fetch = local_df['id'].unique().tolist()
    print(f"Total IDs to search for: {len(ids_to_fetch)}")
    
    # Format IDs for SQL IN clause
    # DuckDB handles large lists well, but let's be safe and quote them
    id_list_str = ", ".join([f"'{x}'" for x in ids_to_fetch])
    
    print("Connecting to DuckDB and querying Overture S3 (this may take a minute)...")
    con = duckdb.connect()
    
    # Install httpfs extension for S3 access
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Query
    # Note: We select `id` and `geometry` (binary wkb usually, or we can convert to text)
    # Checking schema first might be good, but standard overture has 'geometry' as blob
    # We'll fetch it as is, or ST_AsText if using spatial extension (which adds download overhead)
    # Let's just fetch the column columns.geometry is usually a STRUCT or BLOB in parquet
    
    start_time = time.time()
    query = f"""
        SELECT 
            id, 
            geometry
        FROM read_parquet('{S3_PATH}')
        WHERE id IN ({id_list_str})
    """
    
    try:
        # Execute
        print("Executing query...")
        result_df = con.execute(query).df()
        
        print(f"Query finished in {time.time() - start_time:.2f} seconds.")
        print(f"Found {len(result_df)} matches.")
        
        if len(result_df) > 0:
            print("Sample geometry:", result_df.iloc[0]['geometry'])
            
            # Merge back to original
            print("Merging with original data...")
            # Drop existing geometry if it exists (it was missing/broken)
            if 'geometry' in local_df.columns:
                local_df = local_df.drop(columns=['geometry'])
                
            merged_df = local_df.merge(result_df, on='id', how='left')
            
            print(f"Saving to {OUTPUT_FILE}...")
            merged_df.to_parquet(OUTPUT_FILE)
            print("Done.")
            
        else:
            print("Warning: No matching IDs found in the S3 dataset. Check release version or ID validity.")
            
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    fetch_geometry()
