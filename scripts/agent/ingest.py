import os
import pandas as pd
from typing import List, Dict
from schemas import POIRecord

class IngestManager:
    def __init__(self, data_path: str, confidence_threshold: float = 0.65):
        self.data_path = data_path
        self.threshold = confidence_threshold
        
    def load_and_filter(self) -> List[POIRecord]:
        """
        Loads the V5 parquet output, calculates P(open) vs P(closed) confidence,
        and filters for POIs falling below the required threshold.
        """
        try:
            df = pd.read_parquet(self.data_path)
            
            # Use exported v5_confidence if available, else look for base_confidence
            conf_col = 'v5_confidence' if 'v5_confidence' in df.columns else ('base_confidence' if 'base_confidence' in df.columns else 'confidence')
            
            # Filter down to the low confidence tail
            flagged_df = df[df[conf_col] < self.threshold].copy()
            
            # Helper to robustly pull a string out of whatever pandas gives us (arrays, nan, lists, dict-strings)
            def _extract_string(val):
                if val is None:
                    return "Unknown"
                if isinstance(val, float) and pd.isna(val):
                    return "Unknown"
                
                import numpy as np
                if isinstance(val, (list, tuple, np.ndarray)):
                    # Extract the first non-null element if it's an array
                    for item in val:
                        res = _extract_string(item)
                        if res != "Unknown":
                            return res
                    return "Unknown"
                
                if isinstance(val, dict):
                    # Overture names/categories often appear as dicts like {'primary': 'KFC'}
                    primary = val.get('primary', val.get('main', val.get('common')))
                    if primary:
                        return str(primary)
                    if list(val.values()):
                        return str(list(val.values())[0])
                    return "Unknown"
                    
                v_str = str(val).strip()
                if v_str.lower() in ["nan", "none", "", "null", "<na>"]:
                    return "Unknown"
                    
                # if string representaton of a dict like {"primary": "Bob"}
                if v_str.startswith('{') and v_str.endswith('}'):
                    try:
                        import ast
                        d = ast.literal_eval(v_str)
                        if isinstance(d, dict) and d:
                            return _extract_string(d)
                    except Exception:
                        pass
                        
                return v_str
            
            records = []
            for _, row in flagged_df.iterrows():
                # Extract category safely
                category = _extract_string(row.get('category_primary'))
                if category == "Unknown":
                    category = _extract_string(row.get('base_categories'))
                if category == "Unknown":
                    category = _extract_string(row.get('categories'))
                
                # Extract name safely
                name = _extract_string(row.get('base_names'))
                if name == "Unknown":
                    name = _extract_string(row.get('names'))
                    
                # Extract address safely
                address = _extract_string(row.get('base_addresses'))
                if address == "Unknown":
                    address = _extract_string(row.get('addresses'))
                
                records.append(POIRecord(
                    poi_id=str(row.get('id', row.get('base_id', 'unknown_id'))), 
                    name=str(name), 
                    category=str(category),
                    address=str(address),
                    original_label=int(row.get('label', 0)),
                    original_confidence=float(row.get(conf_col, 0.0)),
                    existing_digital_signals={}
                ))
            return records
            
        except Exception as e:
            print(f"Error reading ingestion parquet: {e}")
            return []
