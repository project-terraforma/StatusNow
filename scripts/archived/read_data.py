import duckdb
import os

def main():
    parquet_file = "data/Season 2 Samples 3k Project Updated.parquet"
    
    if not os.path.exists(parquet_file):
        print(f"Error: File '{parquet_file}' not found.")
        return

    print(f"Opening '{parquet_file}' using DuckDB...")
    
    # Create an in-memory DuckDB connection
    con = duckdb.connect()
    
    try:
        # Query to inspect the data
        # We use read_parquet to read the file
        query = f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 5"
        
        # Convert result to a Pandas DataFrame for nicer display
        result = con.execute(query).df()
        
        print("\n--- First 5 rows ---")
        print(result)
        
        print("\n--- Schema ---")

        schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_file}')").df()
        print(schema_df)
        
        print("\n--- Total Row Count ---")
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
        print(f"Total records: {count[0]}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
