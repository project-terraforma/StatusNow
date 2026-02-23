"""Quick diagnostic: check address-based matching potential for NYC."""
import pandas as pd
import json
import re

def norm(s):
    if not s: return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def get_zip(x):
    if isinstance(x, str):
        try: d = json.loads(x)
        except: return ""
    else: d = x
    if isinstance(d, list) and d:
        return str(d[0].get("postcode", "")).strip()[:5]
    return ""

def get_addr(x):
    if isinstance(x, str):
        try: d = json.loads(x)
        except: return ""
    else: d = x
    if isinstance(d, list) and d:
        return d[0].get("freeform", "")
    return ""

def get_name(x):
    if isinstance(x, str):
        try: d = json.loads(x)
        except: return ""
    else: d = x
    if isinstance(d, dict):
        return d.get("primary", "")
    return ""

# Load
lic = pd.read_parquet("data/licenses/nyc_licenses.parquet")
ov = pd.read_parquet("data/combined_truth_dataset_all.parquet")
nyc_ov = ov[ov["source_dataset"] == "overture_nyc"].copy()

# Normalize
nyc_ov["_addr"] = nyc_ov["addresses"].apply(get_addr).apply(norm)
nyc_ov["_zip"] = nyc_ov["addresses"].apply(get_zip)
nyc_ov["_name"] = nyc_ov["names"].apply(get_name)

lic["_addr"] = (lic["address"].fillna("")).apply(norm)
lic["_zip"] = lic["zip"].apply(lambda x: str(x)[:5] if pd.notna(x) else "")

# Address+Zip matching
lic_addr_zip = set(zip(lic["_addr"], lic["_zip"]))
matched_addr = sum(1 for _, r in nyc_ov.iterrows() 
                   if (r["_addr"], r["_zip"]) in lic_addr_zip and r["_addr"])
print(f"NYC Overture matched by addr+zip: {matched_addr} / {len(nyc_ov)} ({matched_addr/len(nyc_ov):.1%})")

# Show some matches
lic_lookup = {}
for _, r in lic.iterrows():
    key = (r["_addr"], r["_zip"])
    if key not in lic_lookup:
        lic_lookup[key] = r

count = 0
for _, r in nyc_ov.iterrows():
    key = (r["_addr"], r["_zip"])
    if key in lic_lookup and key[0]:
        lr = lic_lookup[key]
        print(f"  OV: {r['_name']:40s}  addr={r['_addr']:30s}  zip={r['_zip']}")
        print(f"  LIC: {str(lr['business_name']):40s}  addr={lr['_addr']:30s}  zip={lr['_zip']}")
        print(f"       status={lr.get('license_status','?')}")
        print()
        count += 1
        if count >= 8:
            break

print(f"\nShowed {count} example matches")
