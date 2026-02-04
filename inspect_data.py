import duckdb
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def inspect():
    parquet_file = "Project C Samples.parquet"
    con = duckdb.connect()
    
    # 1. Check if one row is a single place (Unique ID check)
    count_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()[0]
    count_distinct_id = con.execute(f"SELECT COUNT(DISTINCT id) FROM read_parquet('{parquet_file}')").fetchone()[0]
    
    print(f"Total rows: {count_total}")
    print(f"Unique IDs: {count_distinct_id}")
    if count_total == count_distinct_id:
        print("-> Conclusion: One row represents exactly one unique place.")
    else:
        print("-> Conclusion: IDs are NOT unique.")

    # 2. Check Class Balance (Open vs Closed)
    print("\n--- Distribution of 'open' column ---")
    dist = con.execute(f"SELECT open, COUNT(*) as cnt FROM read_parquet('{parquet_file}') GROUP BY open").df()
    print(dist)

    # 3. Inspect specific features to understand what we have
    print("\n--- Inspecting complex columns (names, categories, websites, confidence) ---")
    # Fetch a few rows with both open and closed examples
    query = f"""
    SELECT 
        names, 
        categories, 
        websites, 
        socials,
        phones,
        confidence,
        open 
    FROM read_parquet('{parquet_file}') 
    WHERE open IN (0, 1)
    LIMIT 2
    """
    samples = con.execute(query).df()
    print(samples)

    # 4. Check correlations/presence of fields
    # For example, do open places tend to have websites more often?
    print("\n--- Presence of Contact Info vs Open Status ---")
    # DuckDB's list_exact or checking for null/empty lists depends on the structure. 
    # Usually in Overture, these are Structs or Lists.
    # Let's just check non-null counts for now.
    
    stats_query = f"""
    SELECT 
        open,
        COUNT(*) as total,
        COUNT(websites) as has_websites_struct,
        COUNT(socials) as has_socials_struct,
        COUNT(phones) as has_phones_struct,
        AVG(confidence) as avg_confidence
    FROM read_parquet('{parquet_file}')
    GROUP BY open
    """
    stats = con.execute(stats_query).df()
    print(stats)

if __name__ == "__main__":
    inspect()
