"""
StatusNow — Misclassification Analysis
Generates charts showing:
1. Which place categories have the highest error rates
2. Feature distributions that differ between correct vs incorrect predictions
3. Confusion-style breakdown by category
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from catboost import CatBoostClassifier
import os

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/processed_for_ml_testing.parquet"
OUT_DIR = "/Users/anthonylamas/.gemini/antigravity/brain/6dcd71ed-bf93-479d-b5a1-6ed4db48813c"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

X = df.drop(columns=["open"])
y = df["open"]

# ── Train model + get cross-validated predictions ──────────────────────────
print("Running 5-fold CV predictions with CatBoost...")
model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    verbose=0, auto_class_weights="Balanced",
    random_state=42, allow_writing_files=False,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)

df["predicted"] = preds
df["correct"] = (df["predicted"] == df["open"]).astype(int)
df["pred_prob_open"] = probas[:, 1]

# ── Identify category columns ──────────────────────────────────────────────
cat_cols = [c for c in X.columns if c.startswith("cat_") and c != "cat_is_unknown"]
cat_names = [c.replace("cat_", "").replace("_", " ").title() for c in cat_cols]

overall_acc = df["correct"].mean()
overall_err = 1 - overall_acc
print(f"Overall accuracy: {overall_acc:.4f} ({overall_err:.4f} error rate)")
print(f"Total misclassified: {(df['correct'] == 0).sum()}")

# ── 1. Error rate by place category ─────────────────────────────────────────
print("\nComputing error rates by category...")
cat_stats = []
for col, name in zip(cat_cols, cat_names):
    mask = df[col] == 1
    n = mask.sum()
    if n < 10:
        continue
    err_rate = 1 - df.loc[mask, "correct"].mean()
    n_err = (df.loc[mask, "correct"] == 0).sum()
    pct_closed = 1 - df.loc[mask, "open"].mean()
    cat_stats.append({
        "category": name,
        "n_samples": n,
        "n_errors": n_err,
        "error_rate": err_rate,
        "pct_closed_true": pct_closed,
    })

cat_df = pd.DataFrame(cat_stats).sort_values("error_rate", ascending=False)
print(cat_df.to_string(index=False))

# ── Chart 1: Error rate by category (horizontal bar) ────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#ef4444' if r > overall_err * 1.5 else '#f59e0b' if r > overall_err else '#22c55e'
          for r in cat_df["error_rate"]]
bars = ax.barh(range(len(cat_df)), cat_df["error_rate"], color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(cat_df)))
ax.set_yticklabels(cat_df["category"], fontsize=10)
ax.set_xlabel("Error Rate", fontsize=12)
ax.set_title("Misclassification Rate by Place Category", fontsize=14, fontweight='bold', pad=15)
ax.axvline(overall_err, color='#3b82f6', linestyle='--', linewidth=1.5, label=f'Overall ({overall_err:.1%})')
ax.legend(fontsize=10)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
# Add count labels
for i, (rate, n) in enumerate(zip(cat_df["error_rate"], cat_df["n_samples"])):
    ax.text(rate + 0.003, i, f"{rate:.1%} (n={n})", va='center', fontsize=9, color='#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
path1 = os.path.join(OUT_DIR, "misclass_by_category.png")
fig.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Saved: {path1}")
plt.close()

# ── 2. Error type breakdown (FP vs FN by category) ─────────────────────────
print("\nComputing FP/FN breakdown by category...")
fp_fn_stats = []
for col, name in zip(cat_cols, cat_names):
    mask = df[col] == 1
    n = mask.sum()
    if n < 10:
        continue
    sub = df.loc[mask]
    fp = ((sub["open"] == 0) & (sub["predicted"] == 1)).sum()  # Actually closed, predicted open
    fn = ((sub["open"] == 1) & (sub["predicted"] == 0)).sum()  # Actually open, predicted closed
    fp_fn_stats.append({
        "category": name,
        "n": n,
        "false_positive": fp,
        "false_negative": fn,
        "fp_rate": fp / n if n > 0 else 0,
        "fn_rate": fn / n if n > 0 else 0,
    })

fp_fn_df = pd.DataFrame(fp_fn_stats).sort_values("false_positive", ascending=False)

# Chart 2: Stacked FP/FN by category
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(fp_fn_df))
ax.barh(y_pos, fp_fn_df["false_positive"], color='#ef4444', label='False Positive (Predicted Open, Actually Closed)', edgecolor='white', linewidth=0.5)
ax.barh(y_pos, -fp_fn_df["false_negative"], color='#3b82f6', label='False Negative (Predicted Closed, Actually Open)', edgecolor='white', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(fp_fn_df["category"], fontsize=10)
ax.set_xlabel("Count of Errors", fontsize=12)
ax.set_title("Error Types by Place Category\n(FP = missed closures | FN = false closures)", fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=9, loc='lower right')
ax.invert_yaxis()
ax.axvline(0, color='#94a3b8', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
path2 = os.path.join(OUT_DIR, "fp_fn_by_category.png")
fig.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Saved: {path2}")
plt.close()

# ── 3. Feature distributions: correct vs incorrect ─────────────────────────
print("\nComputing feature correlations with misclassification...")
# Key continuous features to analyze
signal_features = [
    "confidence", "zombie_score", "decay_velocity", "confidence_momentum",
    "digital_congruence", "log_days_since_update", "contact_depth",
    "num_sources", "delta_confidence", "category_churn_risk",
    "recency_x_loss", "nonbrand_stale_risk"
]
signal_features = [f for f in signal_features if f in X.columns]

# Compute difference in means (correct vs incorrect) as "effect size"
effect_sizes = []
for feat in signal_features:
    correct_mean = df.loc[df["correct"] == 1, feat].mean()
    incorrect_mean = df.loc[df["correct"] == 0, feat].mean()
    pooled_std = df[feat].std()
    effect = (incorrect_mean - correct_mean) / pooled_std if pooled_std > 0 else 0
    effect_sizes.append({
        "feature": feat.replace("_", " ").title(),
        "feature_raw": feat,
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "effect_size": effect,
        "abs_effect": abs(effect),
    })

effect_df = pd.DataFrame(effect_sizes).sort_values("abs_effect", ascending=False)
print(effect_df[["feature", "correct_mean", "incorrect_mean", "effect_size"]].to_string(index=False))

# Chart 3: Effect size (Cohen's d-like) of features on misclassification
fig, ax = plt.subplots(figsize=(10, 6))
colors3 = ['#ef4444' if e < 0 else '#22c55e' for e in effect_df["effect_size"]]
ax.barh(range(len(effect_df)), effect_df["effect_size"], color=colors3, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(effect_df)))
ax.set_yticklabels(effect_df["feature"], fontsize=10)
ax.set_xlabel("Effect Size (Standardized Mean Difference)", fontsize=11)
ax.set_title("Features Most Correlated with Misclassification\n(Incorrect − Correct, standardized)", fontsize=13, fontweight='bold', pad=15)
ax.axvline(0, color='#94a3b8', linewidth=1)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
path3 = os.path.join(OUT_DIR, "feature_effect_on_error.png")
fig.savefig(path3, dpi=150, bbox_inches='tight')
print(f"Saved: {path3}")
plt.close()

# ── 4. Box plots for top 4 most correlated features ────────────────────────
top4 = effect_df.head(4)["feature_raw"].tolist()
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for i, feat in enumerate(top4):
    ax = axes[i]
    data_correct = df.loc[df["correct"] == 1, feat].dropna()
    data_incorrect = df.loc[df["correct"] == 0, feat].dropna()
    bp = ax.boxplot([data_correct, data_incorrect],
                    labels=["Correct", "Incorrect"],
                    patch_artist=True,
                    widths=0.6,
                    medianprops=dict(color='#1e293b', linewidth=2))
    bp['boxes'][0].set_facecolor('#22c55e')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#ef4444')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight='600')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.suptitle("Distribution of Top Misclassification-Correlated Features", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
path4 = os.path.join(OUT_DIR, "top_features_boxplot.png")
fig.savefig(path4, dpi=150, bbox_inches='tight')
print(f"Saved: {path4}")
plt.close()

# ── 5. Brand vs Non-brand error rates ──────────────────────────────────────
if "is_brand" in df.columns:
    brand_err = 1 - df.loc[df["is_brand"] == 1, "correct"].mean()
    nonbrand_err = 1 - df.loc[df["is_brand"] == 0, "correct"].mean()
    brand_n = (df["is_brand"] == 1).sum()
    nonbrand_n = (df["is_brand"] == 0).sum()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Brand\nPlaces", "Non-Brand\nPlaces"], [brand_err, nonbrand_err],
                  color=["#3b82f6", "#f59e0b"], edgecolor='white', width=0.5)
    ax.bar_label(bars, [f"{brand_err:.1%}\n(n={brand_n})", f"{nonbrand_err:.1%}\n(n={nonbrand_n})"],
                 padding=5, fontsize=11, fontweight='600')
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("Error Rate: Brand vs Non-Brand Places", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(brand_err, nonbrand_err) * 1.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path5 = os.path.join(OUT_DIR, "brand_vs_nonbrand_error.png")
    fig.savefig(path5, dpi=150, bbox_inches='tight')
    print(f"Saved: {path5}")
    plt.close()

# ── 6. Staleness vs error rate ──────────────────────────────────────────────
if "log_days_since_update" in df.columns:
    df["staleness_bin"] = pd.qcut(df["log_days_since_update"], q=5, duplicates="drop")
    stale_err = df.groupby("staleness_bin", observed=True)["correct"].apply(lambda x: 1 - x.mean())
    stale_n = df.groupby("staleness_bin", observed=True)["correct"].count()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"{iv.left:.1f}–{iv.right:.1f}" for iv in stale_err.index]
    bars = ax.bar(x_labels, stale_err, color='#6366f1', edgecolor='white', width=0.7)
    for i, (rate, n) in enumerate(zip(stale_err, stale_n)):
        ax.text(i, rate + 0.005, f"{rate:.1%}\n(n={n})", ha='center', fontsize=9)
    ax.set_xlabel("Log(Days Since Update) — Quintiles", fontsize=11)
    ax.set_ylabel("Error Rate", fontsize=11)
    ax.set_title("Misclassification Rate by Data Staleness", fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path6 = os.path.join(OUT_DIR, "staleness_vs_error.png")
    fig.savefig(path6, dpi=150, bbox_inches='tight')
    print(f"Saved: {path6}")
    plt.close()

print("\n✅ All charts generated successfully!")
print(f"Output directory: {OUT_DIR}")
