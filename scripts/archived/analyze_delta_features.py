import pandas as pd
import duckdb

# Load processed data
con = duckdb.connect()
df = con.execute("SELECT * FROM read_parquet('data/processed_for_ml.parquet')").df()

print("=" * 80)
print("DELTA FEATURES ANALYSIS")
print("=" * 80)

# Identify delta features
delta_features = [col for col in df.columns if 'delta' in col or 'has_lost' in col or 'has_gained' in col or 'has_any' in col]

print(f"\n{len(delta_features)} Delta Features Created:")
for feat in delta_features:
    print(f"  - {feat}")

print("\n" + "=" * 80)
print("CORRELATION WITH TARGET (open)")
print("=" * 80)

# Calculate correlations
correlations = df[delta_features + ['open']].corr()['open'].drop('open').sort_values(key=abs, ascending=False)

print("\nTop Delta Features by Correlation:")
for feat, corr in correlations.items():
    print(f"  {feat:30s}: {corr:+.4f}")

print("\n" + "=" * 80)
print("DISTRIBUTION COMPARISON: Open vs Closed")
print("=" * 80)

# Compare means for open vs closed
open_places = df[df['open'] == 1]
closed_places = df[df['open'] == 0]

print(f"\nSample Sizes: Open={len(open_places)}, Closed={len(closed_places)}")
print("\nFeature Means:")
print(f"{'Feature':<30s} {'Open Mean':>12s} {'Closed Mean':>12s} {'Difference':>12s}")
print("-" * 70)

for feat in delta_features:
    open_mean = open_places[feat].mean()
    closed_mean = closed_places[feat].mean()
    diff = open_mean - closed_mean
    print(f"{feat:<30s} {open_mean:>12.4f} {closed_mean:>12.4f} {diff:>+12.4f}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Most impactful features
print("\nMost Predictive Delta Features (by correlation magnitude):")
for i, (feat, corr) in enumerate(correlations.head(5).items(), 1):
    direction = "MORE likely open" if corr > 0 else "MORE likely closed"
    print(f"{i}. {feat}: {direction} (r={corr:+.4f})")
