
import duckdb
import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
# Overture Files
OVERTURE_CURRENT = os.path.join(DATA_DIR, "overture_current.parquet")
OVERTURE_PREVIOUS = os.path.join(DATA_DIR, "overture_previous.parquet")

# Season 2 File
SEASON2_FILE = "data/Season 2 Samples 3k Project Updated.parquet"

OUTPUT_FILE = os.path.join(DATA_DIR, "combined_truth_dataset.parquet")

def build_truth_dataset():
    print("Building Truth Dataset...")
    
    con = duckdb.connect()
    
    # --- 1. Load Overture Data ---
    print("Loading Overture Data...")
    
    # Logic:
    # CLOSED = (ID in Previous AND ID NOT in Current) OR (ID in Current AND operating_status = 'closed')
    # OPEN = (ID in Current AND operating_status != 'closed')
    
    query = f"""
        WITH prev AS (
            SELECT * FROM read_parquet('{OVERTURE_PREVIOUS}')
        ),
        curr AS (
            SELECT * FROM read_parquet('{OVERTURE_CURRENT}')
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
    
    # Filter to only rows that have EITHER a previous record OR are currently valid
    # Actually, we want:
    # - Closed (label=0): Must be in Previous (to have base stats) OR explicitly closed in Current
    # - Open (label=1): Must be in Current
    
    # Drop rows that are purely new opens (no history)? User said "combine with previous", implies tracking change?
    # But usually a "truth dataset" for closed prediction needs history.
    # If a place is brand new in Current (prev.id is NULL), we can't calculate deltas.
    # Let's keep them if they are Open, but maybe they aren't useful for "closed prediction" if the model relies on history.
    # But for now, let's keep all valid Overture records in NYC area.
    
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
    overture_final['source_dataset'] = 'overture_nyc'
    
    # --- 3. Align Columns with Season 2 ---
    # Season 2 Cols: label, id, names, categories, confidence, websites, socials, phones, brand, addresses, base_*...
    # We constructed most of them in SQL.
    # Missing: sources, emails, base_sources, base_emails (Overture doesn't provide these easily in this simple query, filling with null)
    
    for col in ['sources', 'emails', 'base_sources', 'base_emails']:
        overture_final[col] = None
        
    print(f"Overture Sampled Shape: {overture_final.shape}")
    
    # --- 4. Merge with Season 2 ---
    if os.path.exists(SEASON2_FILE):
        print(f"\nLoading Season 2 Data from {SEASON2_FILE}...")
        s2_df = con.execute(f"SELECT * FROM read_parquet('{SEASON2_FILE}')").df()
        s2_df['source_dataset'] = 'season2'
        
        print(f"Season 2 Shape: {s2_df.shape}")
        
        # Ensure label matches (Season 2: 1=Open, 0=Closed) - verification
        print(f"Season 2 Label Mean: {s2_df['label'].mean():.2%}")
        
        # Align columns
        common_cols = list(set(overture_final.columns) & set(s2_df.columns))
        # We need to ensure we keep all columns that are in S2 used for training
        # If Overture is missing something S2 has, fill with Null
        # If S2 is missing something Overture has, fill with Null
        
        combined_df = pd.concat([s2_df, overture_final], ignore_index=True)
        print(f"Combined Shape: {combined_df.shape}")
        
    else:
        print("Warning: Season 2 file not found. Skipping merge.")
        combined_df = overture_final

    # Convert complex columns to JSON strings for compatibility
    import json
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

    print(f"\nSaving to {OUTPUT_FILE}...")
    combined_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    if not os.path.exists(OVERTURE_CURRENT) or not os.path.exists(OVERTURE_PREVIOUS):
        print("Error: Input files not found. Run fetch_overture_data.py first.")
    else:
        build_truth_dataset()
