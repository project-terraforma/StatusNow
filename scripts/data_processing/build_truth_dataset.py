import duckdb
import os
import pandas as pd
import numpy as np
import argparse
import json

SEASON2_FILE = "data/Season 2 Samples 3k Project Updated.parquet"

def build_truth_dataset(city):
    print(f"Building Truth Dataset for {city.upper()}...")
    
    # Dynamic Paths
    overture_dir = "data/overture"
    current_file = os.path.join(overture_dir, f"{city}_current.parquet")
    previous_file = os.path.join(overture_dir, f"{city}_previous.parquet")
    output_file = f"data/combined_truth_dataset_{city}.parquet"

    if not os.path.exists(current_file) or not os.path.exists(previous_file):
        print(f"❌ Error: Missing input files for {city}.\nExpected:\n- {current_file}\n- {previous_file}\nRun 'fetch_overture_data.py --city {city}' first.")
        return

    con = duckdb.connect()
    
    # --- 1. Load Overture Data ---
    print(f"Loading Overture Data from {overture_dir}...")
    
    # Logic:
    # CLOSED = (ID in Previous AND ID NOT in Current) OR (ID in Current AND operating_status = 'closed')
    # OPEN = (ID in Current AND operating_status != 'closed')
    
    query = f"""
        WITH prev AS (
            SELECT * FROM read_parquet('{previous_file}')
        ),
        curr AS (
            SELECT * FROM read_parquet('{current_file}')
        )
        SELECT 
            -- Prioritize Current info if available, else Previous
            COALESCE(curr.id, prev.id) as id,
            COALESCE(curr.names, prev.names) as names,
            COALESCE(curr.categories, prev.categories) as categories,
            COALESCE(curr.brand, prev.brand) as brand,
            COALESCE(curr.websites, prev.websites) as websites,
            COALESCE(curr.socials, prev.socials) as socials,
            COALESCE(curr.phones, prev.phones) as phones,
            COALESCE(curr.addresses, prev.addresses) as addresses,
            curr.confidence as confidence,
            
            -- Base columns from Previous (for delta features)
            prev.id as base_id,
            prev.names as base_names,
            prev.categories as base_categories,
            prev.brand as base_brand,
            prev.websites as base_websites,
            prev.socials as base_socials,
            prev.phones as base_phones,
            prev.addresses as base_addresses,
            prev.confidence as base_confidence,
            
            -- Status Logic
            CASE 
                WHEN curr.id IS NULL THEN 0  -- Closed (Missing in Current)
                WHEN curr.operating_status = 'closed' THEN 0 -- Closed (Explicitly Closed)
                ELSE 1 -- Open
            END AS label
            
        FROM prev
        FULL OUTER JOIN curr ON prev.id = curr.id
        WHERE prev.id IS NOT NULL OR curr.id IS NOT NULL
    """
    
    df = con.execute(query).df()
    
    print(f"Total Overture Rows: {len(df)}")
    print(f"Label Dist: {df['label'].value_counts().to_dict()}")
    
    # --- 2. Downsample ---
    TARGET_CLOSED = 3000
    TARGET_OPEN = 6000
    
    print(f"\nDownsampling to {TARGET_OPEN} Open and {TARGET_CLOSED} Closed...")
    
    closed_df = df[df['label'] == 0]
    open_df = df[df['label'] == 1]
    
    if len(closed_df) >= TARGET_CLOSED:
        closed_sampled = closed_df.sample(n=TARGET_CLOSED, random_state=42)
    else:
        print(f"Warning: Only found {len(closed_df)} closed Overture places.")
        closed_sampled = closed_df
        
    if len(open_df) >= TARGET_OPEN:
        open_sampled = open_df.sample(n=TARGET_OPEN, random_state=42)
    else:
         open_sampled = open_df

    overture_final = pd.concat([closed_sampled, open_sampled]).copy()
    overture_final['source_dataset'] = f'overture_{city}'
    
    # --- 3. Align Columns with Season 2 ---
    for col in ['sources', 'emails', 'base_sources', 'base_emails']:
        overture_final[col] = None
        
    print(f"Overture Sampled Shape: {overture_final.shape}")
    
    # --- 4. Merge with Season 2 (NYC Only or Always?) ---
    # User likely wants to cross-validate on SF, but training on Season 2 + NYC is the "Base".
    # For now, let's keep consistent: Merge Season 2 into the training/truth set if available.
    if os.path.exists(SEASON2_FILE):
        print(f"\nLoading Season 2 Data from {SEASON2_FILE}...")
        s2_df = con.execute(f"SELECT * FROM read_parquet('{SEASON2_FILE}')").df()
        s2_df['source_dataset'] = 'season2'
        
        combined_df = pd.concat([s2_df, overture_final], ignore_index=True)
        print(f"Combined Shape: {combined_df.shape}")
        
    else:
        print("Warning: Season 2 file not found. Skipping merge.")
        combined_df = overture_final

    # Convert complex columns to JSON strings for compatibility
    def to_json(x):
        if x is None:
            return None
        if isinstance(x, (dict, list)):
            return json.dumps(x)
        return str(x)

    complex_cols = ['names', 'categories', 'websites', 'socials', 'phones', 'addresses', 'brand', 
                    'base_names', 'base_categories', 'base_websites', 'base_socials', 'base_phones', 'base_addresses', 'base_brand']
    
    print("Converting complex columns to JSON strings...")
    for col in complex_cols:
        if col in combined_df.columns:
             combined_df[col] = combined_df[col].apply(to_json)

    print(f"\nSaving to {output_file}...")
    combined_df.to_parquet(output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Truth Dataset for a City.")
    parser.add_argument("--city", type=str, default="nyc", choices=["nyc", "sf"], help="City to process (nyc or sf)")
    args = parser.parse_args()
    
    build_truth_dataset(args.city)
