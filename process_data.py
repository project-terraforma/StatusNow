import duckdb
import pandas as pd
import numpy as np
import json

def process_data():
    parquet_file = "data/Season 2 Samples 3k Project Updated.parquet"
    output_file = "data/processed_for_ml.parquet"
    
    print(f"Loading '{parquet_file}'...")
    con = duckdb.connect()
    # Read strict to get strings
    df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").df()

    print("Engineering features...")
    
    # ---------------------------
    # Helpers
    # ---------------------------
    def parse_json(x):
        if x is None or pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                if x.strip() == "": return []
                return json.loads(x)
            except:
                return []
        if isinstance(x, (list, dict)):
            return x
        return []

    def get_len(x):
        parsed = parse_json(x)
        if isinstance(parsed, list):
            return len(parsed)
        return 0

    def is_present(x):
        return 1 if get_len(x) > 0 else 0

    # ---------------------------
    # 1. Source Signal (The new heavy hitter)
    # ---------------------------
    def get_source_data(x):
        data = parse_json(x)
        if isinstance(data, list):
            # Extract 'dataset' field
            return [str(item.get('dataset', '')).lower() for item in data if isinstance(item, dict)]
        return []

    # Feature: Number of Sources
    df['num_sources'] = df['sources'].apply(get_len)
    
    # Feature: Has Microsoft (High quality signal)
    df['source_list'] = df['sources'].apply(get_source_data)
    df['source_has_msft'] = df['source_list'].apply(lambda x: 1 if ('microsoft' in x or 'msft' in x) else 0)
    
    # Feature: Source Count > 1 (Cross-verification)
    df['is_cross_verified'] = df['num_sources'].apply(lambda x: 1 if x > 1 else 0)

    # ---------------------------
    # 2. Conflicting Websites
    # ---------------------------
    # User Hypothesis: Different websites -> outdated/confusing -> Closed?
    # Overture websites are usually list of strings.
    def check_website_conflict(x):
        data = parse_json(x)
        if isinstance(data, list) and len(data) > 1:
            # Check if strings are effectively different (ignoring www. or http)
            # Simple heuristic: Set of unique domains
            cleaned = []
            for w in data:
                if isinstance(w, str):
                    # Basic strip to domain-ish
                    s = w.replace("http://", "").replace("https://", "").replace("www.", "").strip().strip("/")
                    cleaned.append(s)
            
            unique_sites = set(cleaned)
            if len(unique_sites) > 1:
                return 1 # CONFLICT
        return 0 # No conflict (0 or 1 site, or same site repeated)

    df['has_conflicting_websites'] = df['websites'].apply(check_website_conflict)

    # ---------------------------
    # 3. Digital Presence
    # ---------------------------
    df['has_website'] = df['websites'].apply(is_present)
    df['has_social'] = df['socials'].apply(is_present)
    df['has_phone'] = df['phones'].apply(is_present)
    
    # We found social length might be inversely correlated for closed, let's keep length
    df['len_socials'] = df['socials'].apply(get_len)
    
    # Contact Depth (Composite)
    def get_email_count(x):
        if pd.isna(x): return 0
        if isinstance(x, (int, float)): return int(x)
        return get_len(x)
        
    len_websites = df['websites'].apply(get_len)
    len_emails = df['emails'].apply(get_email_count)
    df['contact_depth'] = len_websites + df['len_socials'] + len_emails

    # ---------------------------
    # 4. Brand
    # ---------------------------
    def check_brand(x):
        if x is None or pd.isna(x): return 0
        if isinstance(x, str):
            if x.strip() == "" or x == "null" or x == "[]": return 0
            return 1
        return 1
    df['is_brand'] = df['brand'].apply(check_brand)

    # ---------------------------
    # 5. Confidence
    # ---------------------------
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0)

    # ---------------------------
    # 6. Categories (Optimized)
    # ---------------------------
    def get_primary_category(x):
        data = parse_json(x)
        if isinstance(data, dict):
            return data.get('primary', 'unknown')
        return "unknown"
    
    df['category_primary'] = df['categories'].apply(get_primary_category)
    
    # New Feature: Is Unknown Category (Strong closed signal)
    df['cat_is_unknown'] = df['category_primary'].apply(lambda x: 1 if x == 'unknown' else 0)

    # Top 20 One-Hot
    top_categories = df['category_primary'].value_counts().nlargest(20).index.tolist()
    if 'unknown' in top_categories: top_categories.remove('unknown') # handled by specific binary
    
    df['category_simple'] = df['category_primary'].apply(lambda x: x if x in top_categories else 'other')
    dummies = pd.get_dummies(df['category_simple'], prefix='cat')
    
    df_ml = pd.concat([df, dummies], axis=1)

    # ---------------------------
    # 7. Final Output
    # ---------------------------
    target_col = 'label'
    
    # Add new powerful features
    feature_cols = [
        'has_website', 'has_social', 'has_phone', 'contact_depth', 'is_brand', 
        'confidence', 
        'num_sources', 'source_has_msft', 'is_cross_verified', # Source signals
        'has_conflicting_websites', # User Idea
        'len_socials', # Granular social info
        'cat_is_unknown' # Category signal
    ] + list(dummies.columns)
    
    final_df = df_ml[feature_cols + [target_col]].copy()
    final_df = final_df.dropna(subset=[target_col])
    
    final_df.rename(columns={'label': 'open'}, inplace=True)
    final_df['open'] = final_df['open'].astype(int)

    print("\n--- Final Dataset Stats ---")
    print(f"Shape: {final_df.shape}")
    print(f"Class Balance (Open=1): {final_df['open'].mean():.2%}")
    print(f"New Feature 'Conflicting Websites' Mean: {final_df['has_conflicting_websites'].mean():.2%}")
    
    print(f"\nSaving to '{output_file}'...")
    final_df.to_parquet(output_file)
    print("Done.")

if __name__ == "__main__":
    process_data()
