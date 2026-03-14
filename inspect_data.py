import pandas as pd
df = pd.read_parquet('data/v5_predictions_export.parquet')
print("Columns:", list(df.columns))
flagged_df = df[df['v5_confidence'] < 0.65].head(5)
for i, row in flagged_df.iterrows():
    print("row:", i)
    print("names:", type(row.get('names')), repr(row.get('names')))
    print("base_names:", type(row.get('base_names')), repr(row.get('base_names')))
    print("categories:", type(row.get('categories')), repr(row.get('categories')))
    print("base_categories:", type(row.get('base_categories')), repr(row.get('base_categories')))
    print("category_primary:", type(row.get('category_primary')), repr(row.get('category_primary')))
    print("---")
