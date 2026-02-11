import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import fused
import requests
import json
import time
from tqdm import tqdm
import numpy as np

# Configuration
INPUT_FILE = "data/Season 2 Samples 3k Project Updated.parquet"
OUTPUT_FILE = "data/processed_with_mobility.parquet"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {'User-Agent': 'StatusNow/1.0 (oleg@statusnow.com)'}

def parse_json_safe(x):
    try:
        if isinstance(x, str):
            return json.loads(x)
        return x
    except:
        return {}

def geocode_address(row):
    """
    Geocodes an address using Nominatim (OpenStreetMap).
    Returns Point(lon, lat) or None.
    """
    try:
        # Construct address string
        addrs = parse_json_safe(row['addresses'])
        if isinstance(addrs, list) and len(addrs) > 0:
            addr_obj = addrs[0]
            # Try specific fields
            query_parts = []
            if 'freeform' in addr_obj:
                query_parts.append(addr_obj['freeform'])
            if 'locality' in addr_obj:
                query_parts.append(addr_obj['locality'])
            if 'country' in addr_obj:
                query_parts.append(addr_obj['country'])
            
            query = ", ".join(query_parts)
            
            # API Call
            params = {
                'q': query,
                'format': 'json',
                'limit': 1
            }
            response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    return Point(lon, lat)
            
            # Rate limiting
            time.sleep(1.0) 
            
    except Exception as e:
        pass
    return None

# UDF Definitions (Local execution via Fused)
# @fused.udf
def mobility_proxy_udf(bbox: fused.types.Tile=None):
    """
    Mock Mobility UDF that generates random check-in clusters 
    within the bounding box. In production, this would query 
    snowflake/bigquery for real H3 check-ins.
    """
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Generate random points in bbox
    minx, miny, maxx, maxy = bbox.bounds
    n_points = np.random.randint(0, 50) # Random activity level
    
    lons = np.random.uniform(minx, maxx, n_points)
    lats = np.random.uniform(miny, maxy, n_points)
    
    df = pd.DataFrame({'mobility_intensity': np.random.randint(1, 100, n_points)})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    
    # Buffer to simulate coverage area (e.g. 50meters)
    # Using simple degree approximation (1 deg ~ 111km, 0.0005 ~ 50m)
    gdf['geometry'] = gdf.geometry.buffer(0.0005) 
    return gdf

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Initial row count: {len(df)}")

    # 1. Geocoding (Since geometry is missing)
    # Note: For this demo, we will limit to top 50 to avoid long wait times.
    # In production, we'd batch this async or use a paid geocoder.
    print("Checking for geometry...")
    if 'geometry' not in df.columns:
        print("Geometry missing. Starting Geocoding (Demo Limit: 50 rows)...")
        # Initialize geometry column
        df['geometry'] = None
        
        # Filter for demo
        # subset_indices = df.head(50).index
        subset_indices = df.index # Full processing
        
        tqdm.pandas(desc="Geocoding")
        geometries = df.loc[subset_indices].progress_apply(geocode_address, axis=1)
        df.loc[subset_indices, 'geometry'] = geometries
        
        # Drop rows without geometry for the enrichment step
        df_enriched = df.dropna(subset=['geometry']).copy()
        print(f"Rows with geometry: {len(df_enriched)}")
    else:
        df_enriched = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df_enriched, geometry='geometry', crs="EPSG:4326")
    
    # 2. Batch Processing by Spatial Grid
    # We use simple lat/lon rounding (~10km grid)
    gdf['grid_x'] = gdf.geometry.x.round(1)
    gdf['grid_y'] = gdf.geometry.y.round(1)
    
    groups = gdf.groupby(['grid_x', 'grid_y'])
    print(f"Processing {len(groups)} spatial batches...")
    
    mobility_results = []
    
    for idx, (name, group) in enumerate(groups):
        try:
            # Create bbox for the group
            minx, miny, maxx, maxy = group.total_bounds
            # Add buffer
            bbox_poly = box(minx-0.01, miny-0.01, maxx+0.01, maxy+0.01)
            
            # Call Fused UDF
            # Note: fused.run expects a managed UDF name or a local function.
            # We use our local function directly to avoid Cloud Auth for this offline demo.
            print(f"Calling Mobility UDF for Batch {idx}...")
            # mobility_data = fused.run(mobility_proxy_udf, bbox=bbox_poly) 
            mobility_data = mobility_proxy_udf(bbox=bbox_poly)
            
            if mobility_data is not None and not mobility_data.empty:
                # Spatial Join: Check if POI intersects with Mobility Buffer
                # We join POI (group) with Mobility (mobility_data)
                # Left join to keep all POIs
                merged = gpd.sjoin(group, mobility_data, how='left', predicate='intersects')
                
                # If multiple mobility matches, sum intensity
                if 'mobility_intensity' in merged.columns:
                    grouped_metrics = merged.groupby('id')['mobility_intensity'].sum().reset_index()
                    mobility_results.append(grouped_metrics)
                    
        except Exception as e:
            print(f"Error in batch {idx}: {e}")

    # 3. Consolidate and Feature Engineer
    if mobility_results:
        mobility_full = pd.concat(mobility_results)
        # Merge back to original dataframe (left join)
        df_final = df.merge(mobility_full, on='id', how='left')
    else:
        df_final = df.copy()
        df_final['mobility_intensity'] = 0

    # Fill NaNs
    df_final['mobility_intensity'] = df_final['mobility_intensity'].fillna(0)
    
    # Calculate Source Count (needed for logic)
    def get_len(x):
        try:
            return len(json.loads(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 0
        except:
            return 0
            
    df_final['source_count'] = df_final['sources'].apply(get_len)
    
    # Calculate Derived Features
    # Normalized Score
    df_final['mobility_score'] = df_final['mobility_intensity'] / (df_final['source_count'] + 1)
    
    # Ghost Candidate Flag
    # High Sources (>2) but Zero Activity
    df_final['is_ghost_candidate'] = df_final.apply(
        lambda x: 1 if (x['source_count'] > 2 and x['mobility_intensity'] <= 0.1) else 0, axis=1
    )
    
    # 4. Save
    print(f"Saving enriched data to {OUTPUT_FILE}...")
    # Convert geometry to WKT for parquet compatibility if needed, or drop if not needed for ML
    # We drop geometry for the ML script as it expects tabular data
    cols_to_drop = ['geometry', 'grid_x', 'grid_y', 'index_right']
    for col in cols_to_drop:
        if col in df_final.columns:
            df_final.drop(columns=[col], inplace=True)
            
    df_final.to_parquet(OUTPUT_FILE)
    
    print("Done. Sample Enriched Data:")
    print(df_final[['id', 'source_count', 'mobility_intensity', 'is_ghost_candidate']].head())

if __name__ == "__main__":
    main()
