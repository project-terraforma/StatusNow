import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# Load data
df = pd.read_parquet("data/processed_for_ml.parquet")
X = df.drop(columns=['open'])
y = df['open']

print("=" * 80)
print("FEATURE SELECTION ANALYSIS")
print("=" * 80)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class Balance: {y.value_counts(normalize=True).to_dict()}")

# 1. Check for near-zero variance features
print("\n" + "=" * 80)
print("1. NEAR-ZERO VARIANCE FEATURES")
print("=" * 80)

variances = X.var()
low_var_features = variances[variances < 0.01].sort_values()
print(f"\nFeatures with variance < 0.01 (consider removing): {len(low_var_features)}")
if len(low_var_features) > 0:
    print(low_var_features)

# 2. Check correlation between features
print("\n" + "=" * 80)
print("2. HIGHLY CORRELATED FEATURES (r > 0.9)")
print("=" * 80)

corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs:")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1:40s} <-> {feat2:40s}: {corr:.3f}")
else:
    print("\nNo highly correlated feature pairs found!")

# 3. Feature Importance using Random Forest
print("\n" + "=" * 80)
print("3. FEATURE IMPORTANCE (Random Forest)")
print("=" * 80)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

print("\nBottom 10 Least Important Features:")
print(importance_df.tail(10).to_string(index=False))

# Features with importance < 0.005
low_importance = importance_df[importance_df['importance'] < 0.005]
print(f"\n{len(low_importance)} features with importance < 0.005 (candidates for removal):")
print(low_importance.to_string(index=False))

# 4. Recursive Feature Selection Test
print("\n" + "=" * 80)
print("4. PERFORMANCE VS NUMBER OF FEATURES")
print("=" * 80)

# Test with different feature counts
feature_counts = [10, 15, 20, 25, 30, 40, 49]
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    auto_class_weights='Balanced',
    random_state=42,
    allow_writing_files=False
)

print("\nTesting with top N features...")
for n_features in feature_counts:
    if n_features > len(importance_df):
        n_features = len(importance_df)
    
    top_features = importance_df.head(n_features)['feature'].tolist()
    X_subset = X[top_features]
    
    cv_results = cross_validate(
        catboost_model, X_subset, y, 
        cv=cv, 
        scoring={'balanced_acc': 'balanced_accuracy'},
        n_jobs=-1
    )
    
    balanced_acc = cv_results['test_balanced_acc'].mean()
    results.append({
        'n_features': n_features,
        'balanced_acc': balanced_acc
    })
    print(f"  {n_features:2d} features: Balanced Acc = {balanced_acc:.4f}")

# Find optimal number of features
results_df = pd.DataFrame(results)
best_idx = results_df['balanced_acc'].idxmax()
best_n = results_df.loc[best_idx, 'n_features']
best_acc = results_df.loc[best_idx, 'balanced_acc']

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print(f"\n✅ Best Performance: {best_n} features with {best_acc:.4f} balanced accuracy")

# Get full feature set performance
full_features_result = results_df[results_df['n_features'] == results_df['n_features'].max()]
if len(full_features_result) > 0:
    full_acc = full_features_result['balanced_acc'].values[0]
    full_n = full_features_result['n_features'].values[0]
    print(f"   Current ({full_n} features): {full_acc:.4f} balanced accuracy")

if best_n < full_n:
    improvement = best_acc - full_acc
    print(f"   Potential improvement: {improvement:+.4f} ({abs(improvement)*100:.2f}%)")
    print(f"\n📋 Recommended feature set ({best_n} features):")
    recommended_features = importance_df.head(best_n)['feature'].tolist()
    for i, feat in enumerate(recommended_features, 1):
        imp = importance_df[importance_df['feature']==feat]['importance'].values[0]
        print(f"   {i:2d}. {feat:40s} (importance: {imp:.4f})")
else:
    print("\n✅ All features are contributing positively. No need to remove features.")
    recommended_features = importance_df['feature'].tolist()

# Save recommendations
print(f"\n💾 Saving recommended features to 'recommended_features.txt'...")
with open("recommended_features.txt", "w") as f:
    f.write(f"# Recommended Feature Set ({best_n} features)\n")
    f.write(f"# Balanced Accuracy: {best_acc:.4f}\n\n")
    for feat in recommended_features:
        f.write(f"{feat}\n")

print("Done!")
