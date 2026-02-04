import duckdb
import pandas as pd
import numpy as np

def process_data():
    parquet_file = "Project C Samples.parquet"
    output_file = "processed_data.parquet"
    
    print(f"Loading '{parquet_file}'...")
    con = duckdb.connect()
    
    # Load into DataFrame
    df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").df()
    
    print(f"Original shape: {df.shape}")
    
    def safe_len(x):
        # Handle pandas NAType or None or float('nan')
        if x is None or pd.isna(x):
            return 0
        return len(x)

    def is_present(x):
        return 1 if safe_len(x) > 0 else 0

    print("Engineering features...")
    
    # 1. has_website
    df['has_website'] = df['websites'].apply(is_present)
    
    # 2. has_social
    df['has_social'] = df['socials'].apply(is_present)
    
    # 3. has_phone
    df['has_phone'] = df['phones'].apply(is_present)
    
    # 4. contact_depth: len(websites) + len(socials) + len(emails)
    # We need to make sure we treat these as lengths. 
    # 'emails' was in the schema in the previous turn.
    df['len_websites'] = df['websites'].apply(safe_len)
    df['len_socials'] = df['socials'].apply(safe_len)
    df['len_emails'] = df['emails'].apply(safe_len)
    
    df['contact_depth'] = df['len_websites'] + df['len_socials'] + df['len_emails']
    
    # Drop the temporary length columns if not requested, but keeping them might be useful?
    # The user specifically requested specific columns, I'll remove the intermediates to keep it clean matches request.
    df.drop(columns=['len_websites', 'len_socials', 'len_emails'], inplace=True)
    
    # 5. is_brand: 1 if brand is not None else 0
    # brand column might be a struct or string/dict.
    def check_brand(x):
        if x is None:
            return 0
        # If it's a float nan
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return 0
        # If it's an empty dict/list/string? User said "not None". 
        # Usually brand is a struct {names: ..., wikidata: ...}
        return 1

    df['is_brand'] = df['brand'].apply(check_brand)

    # Select the columns to keep. 
    # Usually we want to keep the label 'open' and maybe 'id' and 'confidence' as well from previous step.
    # And the new features.
    # Let's keep all original columns + new features for now to be safe, unless user wants a strictly filtered dataset.
    # User said "process data", usually implies appending features.
    
    new_features = ['has_website', 'has_social', 'has_phone', 'contact_depth', 'is_brand']
    
    print("\n--- Processed Feature Stats ---")
    print(df[new_features].describe())
    
    print(f"\nSaving to '{output_file}'...")
    df.to_parquet(output_file)
    print("Done.")

    # Show a few examples
    print("\n--- First 5 rows of new features ---")
    print(df[['id'] + new_features + ['open']].head())

if __name__ == "__main__":
    process_data()
