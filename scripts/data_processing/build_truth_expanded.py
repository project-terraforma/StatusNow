"""
Build truth datasets for any city, then merge all into one large dataset.
"""

import duckdb
import os
import pandas as pd
import numpy as np
import argparse
import json

SEASON2_FILE = "data/Season 2 Samples 3k Project Updated.parquet"

def build_truth_dataset(city, include_season2=True, max_open=9000, max_closed=3000):
    print(f"\n{'='*60}")
    print(f"Building Truth Dataset for {city.upper()}")
    print(f"{'='*60}")
    
    overture_dir = "data/overture"
    current_file = os.path.join(overture_dir, f"{city}_current.parquet")
    previous_file = os.path.join(overture_dir, f"{city}_previous.parquet")
    output_file = f"data/combined_truth_dataset_{city}.parquet"

    if not os.path.exists(current_file) or not os.path.exists(previous_file):
        print(f"❌ Missing files for {city}. Run fetch first.")
        return None

    con = duckdb.connect()
    
    query = f"""
        WITH prev AS (
            SELECT * FROM read_parquet('{previous_file}')
        ),
        curr AS (
            SELECT * FROM read_parquet('{current_file}')
        )
        SELECT 
            COALESCE(curr.id, prev.id) as id,
            COALESCE(curr.names, prev.names) as names,
            COALESCE(curr.categories, prev.categories) as categories,
            COALESCE(curr.brand, prev.brand) as brand,
            COALESCE(curr.websites, prev.websites) as websites,
            COALESCE(curr.socials, prev.socials) as socials,
            COALESCE(curr.phones, prev.phones) as phones,
            COALESCE(curr.addresses, prev.addresses) as addresses,
            curr.confidence as confidence,
            
            prev.id as base_id,
            prev.names as base_names,
            prev.categories as base_categories,
            prev.brand as base_brand,
            prev.websites as base_websites,
            prev.socials as base_socials,
            prev.phones as base_phones,
            prev.addresses as base_addresses,
            prev.confidence as base_confidence,
            
            CASE 
                WHEN curr.id IS NULL THEN 0
                WHEN curr.operating_status = 'closed' THEN 0
                ELSE 1
            END AS label
            
        FROM prev
        FULL OUTER JOIN curr ON prev.id = curr.id
        WHERE prev.id IS NOT NULL OR curr.id IS NOT NULL
    """
    
    df = con.execute(query).df()
    
    # Stats
    n_closed = (df['label'] == 0).sum()
    n_open = (df['label'] == 1).sum()
    print(f"  Total rows: {len(df)}")
    print(f"  Open: {n_open}, Closed: {n_closed}")
    
    # Downsample
    closed_df = df[df['label'] == 0]
    open_df = df[df['label'] == 1]
    
    if len(closed_df) > max_closed:
        closed_sampled = closed_df.sample(n=max_closed, random_state=42)
    else:
        closed_sampled = closed_df
        
    if len(open_df) > max_open:
        open_sampled = open_df.sample(n=max_open, random_state=42)
    else:
        open_sampled = open_df

    overture_final = pd.concat([closed_sampled, open_sampled]).copy()
    overture_final['source_dataset'] = f'overture_{city}'
    
    # Add missing columns
    for col in ['sources', 'emails', 'base_sources', 'base_emails']:
        overture_final[col] = None
    
    print(f"  Sampled: {len(overture_final)} ({len(closed_sampled)} closed + {len(open_sampled)} open)")
    
    # Merge with Season 2 only for the first city
    if include_season2 and os.path.exists(SEASON2_FILE):
        print(f"  Merging with Season 2 data...")
        s2_df = con.execute(f"SELECT * FROM read_parquet('{SEASON2_FILE}')").df()
        s2_df['source_dataset'] = 'season2'
        combined_df = pd.concat([s2_df, overture_final], ignore_index=True)
    else:
        combined_df = overture_final

    # Convert complex columns to JSON strings
    complex_cols = ['names', 'categories', 'websites', 'socials', 'phones', 'addresses', 'brand',
                    'base_names', 'base_categories', 'base_websites', 'base_socials', 'base_phones', 
                    'base_addresses', 'base_brand']
    
    def to_json(x):
        if x is None: return None
        if isinstance(x, (dict, list)): return json.dumps(x)
        return str(x)
    
    for col in complex_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(to_json)

    combined_df.to_parquet(output_file)
    print(f"  ✅ Saved to {output_file}")
    return output_file


def merge_all_cities(cities, output_path="data/combined_truth_dataset_expanded.parquet"):
    """Merge all city datasets, deduplicating Season 2 rows."""
    print(f"\n{'='*60}")
    print(f"MERGING {len(cities)} CITIES")
    print(f"{'='*60}")
    
    dfs = []
    for city in cities:
        path = f"data/combined_truth_dataset_{city}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            dfs.append(df)
            print(f"  {city}: {len(df)} rows")
        else:
            print(f"  {city}: ❌ not found")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total before dedup: {len(combined)}")
    
    # Deduplicate (Season 2 appears in each city build)
    deduped = combined.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    print(f"  Total after dedup:  {len(deduped)}")
    
    # Stats
    print(f"\n  Source distribution:")
    for src, cnt in deduped['source_dataset'].value_counts().items():
        label_dist = deduped[deduped['source_dataset'] == src]['label'].value_counts().to_dict()
        print(f"    {src:20s}: {cnt:>6d} (Open: {label_dist.get(1.0, 0)}, Closed: {label_dist.get(0.0, 0)})")
    
    print(f"\n  Overall: {len(deduped)} rows, {(deduped['label']==1).sum()} Open, {(deduped['label']==0).sum()} Closed")
    
    deduped.to_parquet(output_path)
    print(f"  ✅ Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=str, nargs="+", 
                        default=["chicago", "la", "houston", "phoenix", "philly"])
    parser.add_argument("--build-only", action="store_true", help="Only build, don't merge")
    parser.add_argument("--merge-all", action="store_true", help="Merge all available city datasets")
    args = parser.parse_args()
    
    if args.merge_all:
        # Find all available city datasets
        all_cities = []
        for f in os.listdir("data"):
            if f.startswith("combined_truth_dataset_") and f.endswith(".parquet"):
                city = f.replace("combined_truth_dataset_", "").replace(".parquet", "")
                if city not in ("all", "expanded"):
                    all_cities.append(city)
        print(f"Found cities: {all_cities}")
        merge_all_cities(all_cities)
    else:
        # Build truth datasets for each city
        for i, city in enumerate(args.cities):
            # Only include season2 for the first city
            build_truth_dataset(city, include_season2=(i == 0))
        
        if not args.build_only:
            # Merge all (including existing nyc, sf)
            all_cities = ["nyc", "sf"] + args.cities
            merge_all_cities(all_cities)
