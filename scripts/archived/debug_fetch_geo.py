import duckdb
import pandas as pd
import time

# Configuration
LOCAL_DATA_FILE = "data/Season 2 Samples 3k Project Updated.parquet"
# Try one release earlier just in case IDs rotated (unlikely for GERS but possible if dataset changed)
# S3_PATH = "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*" 
S3_PATH = "s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*"

def debug_fetch():
    # Only fetch ONE id to see if it works
    test_id = "08f391ab10c7114d033517ea0e3905a9" 
    
    print(f"Testing fetch for ID: {test_id}")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Check if we can just count rows in one partition to verify access
    # query_check = f"SELECT count(*) FROM read_parquet('s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*') LIMIT 1"
    # print("Checking access...")
    # print(con.execute(query_check).fetchall())

    query = f"""
        SELECT id, geometry
        FROM read_parquet('{S3_PATH}')
        WHERE id = '{test_id}'
    """
    print("Executing single lookup...")
    start = time.time()
    res = con.execute(query).df()
    print(f"Time: {time.time()-start:.2f}s")
    print(res)

if __name__ == "__main__":
    debug_fetch()
