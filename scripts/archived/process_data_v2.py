import duckdb
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA

def process_data_v2(use_project_c_merge=False):
    import os
    
    # 1. Load data - optionally merge Project C
    if use_project_c_merge:
        print("=" * 80)
        print("MERGING DATASETS")
        print("=" * 80)
        
        con = duckdb.connect()
        season2_df = con.execute("SELECT * FROM read_parquet('data/Season 2 Samples 3k Project Updated.parquet')").df()
        project_c_df = con.execute("SELECT * FROM read_parquet('data/Project C Samples.parquet')").df()
        
        # Align schemas - Project C has 'open' instead of 'label'
        if 'open' in project_c_df.columns and 'label' not in project_c_df.columns:
            project_c_df['label'] = project_c_df['open']
            project_c_df = project_c_df.drop(columns=['open', 'geometry', 'bbox', 'type', 'version'], errors='ignore')
        
        # Add base_* columns to Project C (fill with nulls or empty)
        base_cols = [col for col in season2_df.columns if col.startswith('base_')]
        for col in base_cols:
            if col not in project_c_df.columns:
                project_c_df[col] = None
        
        # Merge
        df = pd.concat([season2_df, project_c_df], ignore_index=True)
        print(f"✅ Merged: Season 2 ({len(season2_df)} rows) + Project C ({len(project_c_df)} rows) = {len(df)} total rows")
    else:
        parquet_file = "data/Season 2 Samples 3k Project Updated.parquet"
        if not os.path.exists(parquet_file):
            print(f"Error: File '{parquet_file}' not found.")
            return
        
        print(f"Reading '{parquet_file}'...")
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").df()
    
    print("Engineering features...")
    
    # Helper functions
    def parse_json(x):
        if x is None or pd.isna(x): return []
        if isinstance(x, str):
            try:
                if x.strip() == "": return []
                return json.loads(x)
            except: return []
        if isinstance(x, (list, dict)): return x
        return []
    
    def get_len(x):
        parsed = parse_json(x)
        return len(parsed) if isinstance(parsed, list) else 0
    
    def is_present(x):
        return 1 if get_len(x) > 0 else 0
    
    # ========================================================================
    # EXISTING FEATURES (keeping the good ones)
    # ========================================================================
    
    # Source Data
    def get_source_data(x):
        data = parse_json(x)
        if isinstance(data, list):
            return [str(item.get('dataset', '')).lower() for item in data if isinstance(item, dict)]
        return []
    
    # Recency
    from datetime import datetime
    CURRENT_DATE = datetime(2026, 2, 4)
    
    def get_source_recency_stats(x):
        data = parse_json(x)
        if not isinstance(data, list) or len(data) == 0:
            return pd.Series([9999, 9999, 9999])
        
        dates = []
        for item in data:
            if isinstance(item, dict):
                u_time = item.get('update_time')
                if u_time:
                    try:
                        dt = datetime.strptime(u_time.split('T')[0], "%Y-%m-%d")
                        days_diff = (CURRENT_DATE - dt).days
                        dates.append(days_diff)
                    except: pass
        
        return pd.Series([min(dates), max(dates), sum(dates)/len(dates)] if dates else [9999, 9999, 9999])
    
    print("Calculating source features...")
    df['num_sources'] = df['sources'].apply(get_len)
    df['source_list'] = df['sources'].apply(get_source_data)
    df['source_has_msft'] = df['source_list'].apply(lambda x: 1 if ('microsoft' in x or 'msft' in x) else 0)
    df['is_cross_verified'] = df['num_sources'].apply(lambda x: 1 if x > 1 else 0)
    df[['days_since_latest_update', 'days_since_oldest_update', 'avg_days_since_update']] = df['sources'].apply(get_source_recency_stats)
    
    # Digital Presence
    print("Calculating digital presence...")
    df['has_website'] = df['websites'].apply(is_present)
    df['has_social'] = df['socials'].apply(is_present)
    df['has_phone'] = df['phones'].apply(is_present)
    
    def check_social_platform(x, platform):
        data = parse_json(x)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and platform in item.lower():
                    return 1
        return 0
    
    df['has_facebook'] = df['socials'].apply(lambda x: check_social_platform(x, 'facebook.com'))
    df['len_socials'] = df['socials'].apply(get_len)
    
    def get_email_count(x):
        if pd.isna(x): return 0
        if isinstance(x, (int, float)): return int(x)
        return get_len(x)
    
    len_websites = df['websites'].apply(get_len)
    len_emails = df['emails'].apply(get_email_count)
    df['contact_depth'] = len_websites + df['len_socials'] + len_emails
    
    # Brand
    def check_brand(x):
        if x is None or pd.isna(x): return 0
        if isinstance(x, str):
            if x.strip() == "" or x == "null" or x == "[]": return 0
            return 1
        return 1
    df['is_brand'] = df['brand'].apply(check_brand)
    
    # Confidence
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0)
    
    # Categories
    print("Processing categories...")
    def get_primary_category(x):
        data = parse_json(x)
        if isinstance(data, dict):
            return data.get('primary', 'unknown')
        return "unknown"
    
    df['category_primary'] = df['categories'].apply(get_primary_category)
    df['cat_is_unknown'] = df['category_primary'].apply(lambda x: 1 if x == 'unknown' else 0)
    
    top_categories = df['category_primary'].value_counts().nlargest(20).index.tolist()
    if 'unknown' in top_categories: top_categories.remove('unknown')
    
    df['category_simple'] = df['category_primary'].apply(lambda x: x if x in top_categories else 'other')
    dummies = pd.get_dummies(df['category_simple'], prefix='cat')
    
    # ========================================================================
    # DELTA FEATURES (baseline vs current)
    # ========================================================================
    
    print("Calculating delta features...")
    df['base_confidence'] = pd.to_numeric(df['base_confidence'], errors='coerce').fillna(0)
    df['delta_confidence'] = df['confidence'] - df['base_confidence']
    
    base_websites_len = df['base_websites'].apply(get_len)
    current_websites_len = df['websites'].apply(get_len)
    df['delta_num_websites'] = current_websites_len - base_websites_len
    
    base_socials_len = df['base_socials'].apply(get_len)
    current_socials_len = df['socials'].apply(get_len)
    df['delta_num_socials'] = current_socials_len - base_socials_len
    
    base_phones_len = df['base_phones'].apply(get_len)
    current_phones_len = df['phones'].apply(get_len)
    df['delta_num_phones'] = current_phones_len - base_phones_len
    
    df['has_lost_website'] = (df['delta_num_websites'] < 0).astype(int)
    df['has_gained_social'] = (df['delta_num_socials'] > 0).astype(int)
    df['has_lost_social'] = (df['delta_num_socials'] < 0).astype(int)
    
    df['delta_total_contact'] = df['delta_num_websites'] + df['delta_num_socials'] + df['delta_num_phones']
    df['has_any_loss'] = ((df['delta_num_websites'] < 0) | (df['delta_num_socials'] < 0) | (df['delta_num_phones'] < 0)).astype(int)
    
    # ========================================================================
    # NEW FEATURE 1: INTERACTION FEATURES (Recency × Loss)
    # ========================================================================
    
    print("Creating interaction features...")
    
    # Recency * Loss: Recent losses are stronger signals
    df['recency_x_loss'] = df['days_since_latest_update'] * df['has_any_loss']
    df['recency_x_social_loss'] = df['days_since_latest_update'] * df['has_lost_social']
    
    # Zombie Score: High sources but very old updates = database purgatory
    df['zombie_score'] = df['num_sources'] / (df['avg_days_since_update'] + 1)  # +1 to avoid division by zero
    
    # Decay Velocity: How fast is digital presence declining?
    df['decay_velocity'] = df['delta_total_contact'] / (df['avg_days_since_update'] + 1)
    
    # Confidence Momentum: Confidence change relative to time
    df['confidence_momentum'] = df['delta_confidence'] / (df['avg_days_since_update'] + 1)
    
    # ========================================================================
    # NEW FEATURE 2: CATEGORY CHURN RISK
    # ========================================================================
    
    print("Calculating category-specific churn rates...")
    
    # Calculate churn rate per category from training data
    category_churn = df.groupby('category_primary')['label'].agg(['mean', 'count'])
    category_churn['churn_rate'] = 1 - category_churn['mean']  # 1 - open_rate = churn_rate
    
    # Only use categories with at least 10 samples for reliability
    category_churn_reliable = category_churn[category_churn['count'] >= 10]['churn_rate'].to_dict()
    
    # Map to feature (use median churn for unknown categories)
    median_churn = category_churn[category_churn['count'] >= 10]['churn_rate'].median()
    df['category_churn_risk'] = df['category_primary'].map(category_churn_reliable).fillna(median_churn)
    
    # ========================================================================
    # NEW FEATURE 3: DIGITAL CONGRUENCE (Website/Social consistency)
    # ========================================================================
    
    print("Checking digital congruence...")
    
    # Check if website domain appears in social handles (basic heuristic)
    def check_digital_congruence(websites, socials):
        if pd.isna(websites) or pd.isna(socials):
            return 0
        
        websites_parsed = parse_json(websites)
        socials_parsed = parse_json(socials)
        
        if not websites_parsed or not socials_parsed:
            return 0
        
        # Extract domain from first website
        if isinstance(websites_parsed, list) and len(websites_parsed) > 0:
            website = str(websites_parsed[0])
            domain = website.replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0].split(".")[0]
            
            # Check if domain appears in any social handle
            for social in socials_parsed:
                if isinstance(social, str) and domain.lower() in social.lower():
                    return 1
        
        return 0
    
    df['digital_congruence'] = df.apply(lambda row: check_digital_congruence(row['websites'], row['socials']), axis=1)
    
    # ========================================================================
    # NEW FEATURE 4: PCA ON REDUNDANT RECENCY FEATURES
    # ========================================================================
    
    print("Applying PCA to redundant recency features...")
    
    # Apply PCA to highly correlated recency features
    recency_features = df[['days_since_latest_update', 'avg_days_since_update']].fillna(9999)
    
    pca = PCA(n_components=1)
    df['recency_pca'] = pca.fit_transform(recency_features).flatten()
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_[0]:.2%}")
    
    # Concatenate with dummies
    df_ml = pd.concat([df, dummies], axis=1)
    
    # ========================================================================
    # FEATURE SELECTION (remove redundant features)
    # ========================================================================
    
    print("Selecting final features...")
    
    target_col = 'label'
    
    # Core features (keeping only one from highly correlated pairs)
    feature_cols = [
        # Core metadata
        'confidence', 'is_brand', 'num_sources', 'source_has_msft', 'is_cross_verified',
        
        # Digital presence (removing has_facebook and len_socials due to perfect correlation with has_social)
        'has_website', 'has_social', 'has_phone', 'contact_depth',
        
        # Category
        'cat_is_unknown', 'category_churn_risk',
        
        # Delta features (keeping strong ones, removing weak phone deltas)
        'delta_confidence', 'delta_num_socials', 'delta_total_contact',
        'has_gained_social', 'has_lost_social', 'has_any_loss',
        
        # Recency (using PCA instead of both correlated features)
        'recency_pca',
        
        # NEW: Interaction features
        'recency_x_loss', 'recency_x_social_loss', 'zombie_score', 'decay_velocity', 'confidence_momentum',
        
        # NEW: Digital congruence
        'digital_congruence'
        
    ] + list(dummies.columns)
    
    final_df = df_ml[feature_cols + [target_col]].copy()
    final_df = final_df.dropna(subset=[target_col])
    
    final_df.rename(columns={'label': 'open'}, inplace=True)
    final_df['open'] = final_df['open'].astype(int)
    
    print("\n" + "=" * 80)
    print("FINAL DATASET STATS")
    print("=" * 80)
    print(f"Shape: {final_df.shape}")
    print(f"Class Balance (Open=1): {final_df['open'].mean():.2%}")
    print(f"Feature Count: {len(feature_cols)}")
    print(f"\nNew Features Added: 10")
    print(f"  - Interaction: recency_x_loss, recency_x_social_loss, zombie_score, decay_velocity, confidence_momentum")
    print(f"  - Category Risk: category_churn_risk")
    print(f"  - Digital Congruence: digital_congruence")
    print(f"  - PCA: recency_pca (replacing 2 correlated features)")
    
    output_file = "data/processed_for_ml_v2.parquet"
    print(f"\nSaving to '{output_file}'...")
    final_df.to_parquet(output_file)
    print("Done!")
    
    return final_df

if __name__ == "__main__":
    import sys
    
    # Check if user wants to merge datasets
    merge = '--merge' in sys.argv or '-m' in sys.argv
    
    if merge:
        print("\n🔗 MERGE MODE: Combining Season 2 + Project C datasets")
    else:
        print("\n📊 STANDARD MODE: Using Season 2 dataset only")
        print("   (Use --merge flag to combine with Project C)")
    
    process_data_v2(use_project_c_merge=merge)
