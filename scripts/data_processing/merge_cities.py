import pandas as pd
import argparse
import os

def merge_cities(cities, output_path):
    print(f"Merging datasets from cities: {cities}...")
    
    dfs = []
    for city in cities:
        path = f"data/combined_truth_dataset_{city}.parquet"
        if not os.path.exists(path):
            print(f"❌ Error: File not found for {city}: {path}")
            print(f"   Please run 'python scripts/data_processing/build_truth_dataset.py --city {city}' first.")
            return
        
        print(f"Loading {path}...")
        df = pd.read_parquet(path)
        dfs.append(df)
    
    if not dfs:
        print("No data loaded.")
        return

    # Concatenate
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before deduplication: {len(combined_df)}")
    
    # Deduplicate based on 'id'
    # Season 2 rows will be duplicated if present in multiple city builds.
    # Overture IDs should be unique per city (unless overlap in BBox, which shouldn't happen for NYC vs SF).
    # We keep the first occurrence.
    
    deduped_df = combined_df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    print(f"Total rows after deduplication: {len(deduped_df)}")
    
    # Verify Season 2 integration
    s2_count = deduped_df[deduped_df['source_dataset'] == 'season2'].shape[0]
    print(f"Season 2 rows preserved: {s2_count}")
    
    # Verify City counts
    for city in cities:
        city_count = deduped_df[deduped_df['source_dataset'] == f'overture_{city}'].shape[0]
        print(f"Overture {city.upper()} rows preserved: {city_count}")

    print(f"Saving merged dataset to {output_path}...")
    deduped_df.to_parquet(output_path)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Truth Datasets from Multiple Cities.")
    parser.add_argument("--cities", type=str, nargs="+", default=["nyc", "sf"], help="List of cities to merge (e.g. nyc sf)")
    parser.add_argument("--output", type=str, default="data/combined_truth_dataset_all.parquet", help="Output filepath")
    
    args = parser.parse_args()
    
    merge_cities(args.cities, args.output)
