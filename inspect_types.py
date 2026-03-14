import pandas as pd
df = pd.read_parquet('data/v5_predictions_export.parquet')
print("names: ", type(df['names'].iloc[0]), df['names'].iloc[0])
print("base_names: ", type(df.get('base_names', pd.Series([None])).iloc[0]), df.get('base_names', pd.Series([None])).iloc[0])
