import duckdb
import pandas as pd
import numpy as np

def process_data():
    parquet_file = "Project C Samples.parquet"
    output_file = "processed_for_ml.parquet"
    
    print(f"Loading '{parquet_file}'...")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").df()

    print("Engineering features...")
    
    # --- 1. Basic Digital Presence ---
    def safe_len(x):
        if x is None or pd.isna(x): return 0
        return len(x)

    def is_present(x):
        return 1 if safe_len(x) > 0 else 0

    df['has_website'] = df['websites'].apply(is_present)
    df['has_social'] = df['socials'].apply(is_present)
    df['has_phone'] = df['phones'].apply(is_present)
    
    # Contact Depth
    len_websites = df['websites'].apply(safe_len)
    len_socials = df['socials'].apply(safe_len)
    len_emails = df['emails'].apply(safe_len)
    df['contact_depth'] = len_websites + len_socials + len_emails

    # Is Brand
    # Check if 'brand' struct is not null
    def check_brand(x):
        if x is None or pd.isna(x): return 0
        return 1
    df['is_brand'] = df['brand'].apply(check_brand)

    # --- 2. Confidence ---
    # It seems 'confidence' is numerical. Fill NaN with median or 0.
    # From inspect: confidence is ~0.8-0.9
    df['confidence'] = df['confidence'].fillna(df['confidence'].median())

    # --- 3. Categories (One-Hot) ---
    # Extract 'primary' from the struct/map
    # DuckDB returns struct as dict
    def get_primary_category(x):
        if x is None or pd.isna(x):
            return "unknown"
        # It handles both dict (from pandas/duckdb conversion) and string if parsed differently
        if isinstance(x, dict):
            return x.get('primary', 'unknown')
        return "unknown"
    
    df['category_primary'] = df['categories'].apply(get_primary_category)
    
    # Keep Top 20 categories, map rest to "other"
    top_categories = df['category_primary'].value_counts().nlargest(20).index.tolist()
    df['category_simple'] = df['category_primary'].apply(lambda x: x if x in top_categories else 'other')
    
    # One-Hot Encode
    print(f"One-hot encoding {len(top_categories)} top categories...")
    dummies = pd.get_dummies(df['category_simple'], prefix='cat')
    
    # Concat
    df_ml = pd.concat([df, dummies], axis=1)

    # --- 4. Final Selection ---
    # Select feature columns and target
    feature_cols = [
        'has_website', 'has_social', 'has_phone', 'contact_depth', 'is_brand', 
        'confidence'
    ] + list(dummies.columns)
    
    target_col = 'open'
    
    # Filter final dataframe
    final_df = df_ml[feature_cols + [target_col]].copy()
    
    # Drop rows where target might be NaN (shouldn't be based on inspection, but safety first)
    final_df = final_df.dropna(subset=[target_col])
    final_df[target_col] = final_df[target_col].astype(int)

    print("\n--- Final Dataset Stats ---")
    print(f"Shape: {final_df.shape}")
    print(f"Class Balance (Open=1): {final_df[target_col].mean():.2%}")
    
    print(f"\nSaving to '{output_file}'...")
    final_df.to_parquet(output_file)
    print("Done.")

if __name__ == "__main__":
    process_data()
