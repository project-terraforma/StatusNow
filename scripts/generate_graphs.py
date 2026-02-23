"""
generate_graphs.py — Visualizations for processed_all_v3.parquet
Produces 10 PNG charts in docs/graphs/
Requirements: pandas, matplotlib, seaborn, numpy, scikit-learn
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fix Windows cp1252 console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# ── Config ────────────────────────────────────────────────────────────────
INPUT   = "data/processed_all_v3.parquet"
OUT_DIR = "docs/graphs"

PALETTE = {"Open": "#22c55e", "Closed": "#ef4444"}
LABEL_MAP = {1: "Open", 0: "Closed"}

# ── Setup ─────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#475569",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"#0f172a",
})

df = pd.read_parquet(INPUT)
df["label_str"] = df["open"].map(LABEL_MAP)
print(f"Loaded {INPUT}  --  {df.shape[0]:,} rows x {df.shape[1]} cols")

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK]  {path}")


# ══════════════════════════════════════════════════════════════════════════
# 1. CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/10] Class balance ...")
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["label_str"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[l] for l in counts.index],
              edgecolor="#0f172a", linewidth=1.5, width=0.55)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}\n({val/len(df):.1%})", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#e2e8f0")
ax.set_title("Class Balance  (Open vs Closed)", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.2)
_save(fig, "class_balance.png")

# ══════════════════════════════════════════════════════════════════════════
# 2. CONFIDENCE BY LABEL
# ══════════════════════════════════════════════════════════════════════════
print("[2/10] Confidence by label ...")
fig, ax = plt.subplots(figsize=(8, 5))
for label_str, color in PALETTE.items():
    subset = df[df["label_str"] == label_str]["confidence"].dropna()
    ax.hist(subset, bins=40, alpha=0.65, color=color, label=label_str, edgecolor="#0f172a")
ax.set_title("Confidence Score Distribution by Label", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Confidence")
ax.set_ylabel("Count")
ax.legend(framealpha=0.9)
_save(fig, "confidence_by_label.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════
print("[3/10] Correlation heatmap ...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude the one-hot cat_ columns for readability; keep core features
core_numeric = [c for c in numeric_cols if not c.startswith("cat_") and c != "label_str"]
corr = df[core_numeric].corr()
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation Matrix  (excl. category dummies)", fontsize=15, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
_save(fig, "correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. FEATURE DISTRIBUTIONS (6-panel)
# ══════════════════════════════════════════════════════════════════════════
print("[4/10] Feature distributions ...")
dist_features = ["confidence", "num_sources", "contact_depth",
                  "log_days_since_update", "zombie_score", "category_churn_risk"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, feat in zip(axes, dist_features):
    for label_str, color in PALETTE.items():
        subset = df[df["label_str"] == label_str][feat].dropna()
        ax.hist(subset, bins=35, alpha=0.6, color=color, label=label_str, edgecolor="#0f172a")
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
fig.suptitle("Key Feature Distributions by Label", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
_save(fig, "feature_distributions.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. DELTA FEATURES BY LABEL
# ══════════════════════════════════════════════════════════════════════════
print("[5/10] Delta features ...")
delta_cols = ["delta_confidence", "delta_num_socials", "delta_total_contact"]
means = df.groupby("label_str")[delta_cols].mean()
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(delta_cols))
w = 0.32
for i, (label_str, color) in enumerate(PALETTE.items()):
    vals = means.loc[label_str]
    bars = ax.bar(x + i * w - w/2, vals, w, color=color, label=label_str,
                  edgecolor="#0f172a", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color="#e2e8f0")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("delta_", "d_") for c in delta_cols], fontsize=11)
ax.axhline(0, color="#475569", linewidth=0.8)
ax.set_title("Mean Delta Features by Label", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Mean Value")
ax.legend(framealpha=0.9)
_save(fig, "delta_features.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. RECENCY vs LABEL (box plot)
# ══════════════════════════════════════════════════════════════════════════
print("[6/10] Recency vs label ...")
fig, ax = plt.subplots(figsize=(7, 5))
data_for_box = [df[df["open"] == 1]["log_days_since_update"].dropna(),
                df[df["open"] == 0]["log_days_since_update"].dropna()]
bp = ax.boxplot(data_for_box, labels=["Open", "Closed"], patch_artist=True,
                widths=0.5,
                medianprops=dict(color="#fbbf24", linewidth=2),
                whiskerprops=dict(color="#94a3b8"),
                capprops=dict(color="#94a3b8"),
                flierprops=dict(marker="o", markerfacecolor="#64748b", markersize=3, alpha=0.5))
for patch, color in zip(bp["boxes"], [PALETTE["Open"], PALETTE["Closed"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("#0f172a")
ax.set_title("Log Days Since Update by Label", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("log(days + 1)")
_save(fig, "recency_vs_label.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. STALENESS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════
print("[7/10] Staleness breakdown ...")
fig, ax = plt.subplots(figsize=(8, 5))
stale_cols = ["is_stale_6mo", "is_stale_1yr", "is_stale_2yr"]
stale_data = []
for label_str in ["Open", "Closed"]:
    sub = df[df["label_str"] == label_str]
    n = len(sub)
    fresh = ((sub["is_stale_6mo"] == 0)).sum() / n
    s6 = ((sub["is_stale_6mo"] == 1) & (sub["is_stale_1yr"] == 0)).sum() / n
    s1 = ((sub["is_stale_1yr"] == 1) & (sub["is_stale_2yr"] == 0)).sum() / n
    s2 = (sub["is_stale_2yr"] == 1).sum() / n
    stale_data.append({"label": label_str, "Fresh (<6mo)": fresh,
                       "6mo–1yr": s6, "1yr–2yr": s1, ">2yr": s2})

stale_df = pd.DataFrame(stale_data).set_index("label")
colors_stale = ["#22c55e", "#facc15", "#f97316", "#ef4444"]
stale_df.plot(kind="bar", stacked=True, ax=ax, color=colors_stale,
              edgecolor="#0f172a", linewidth=1.2, width=0.5)
ax.set_title("Staleness Breakdown by Label", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Proportion")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
_save(fig, "staleness_breakdown.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. DIGITAL PRESENCE BY LABEL
# ══════════════════════════════════════════════════════════════════════════
print("[8/10] Digital presence ...")
presence_cols = ["has_website", "has_social", "has_phone", "is_brand", "has_facebook"]
presence_cols = [c for c in presence_cols if c in df.columns]
fig, ax = plt.subplots(figsize=(9, 5))
means_p = df.groupby("label_str")[presence_cols].mean()
x = np.arange(len(presence_cols))
w = 0.32
for i, (label_str, color) in enumerate(PALETTE.items()):
    vals = means_p.loc[label_str]
    bars = ax.bar(x + i * w - w/2, vals, w, color=color, label=label_str,
                  edgecolor="#0f172a", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.0%}", ha="center", va="bottom", fontsize=9, color="#e2e8f0")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("has_", "").replace("is_", "").title() for c in presence_cols], fontsize=11)
ax.set_title("Digital Presence & Brand Rate by Label", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.12)
ax.legend(framealpha=0.9)
_save(fig, "digital_presence.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. TOP FEATURE IMPORTANCE (mutual information)
# ══════════════════════════════════════════════════════════════════════════
print("[9/10] Feature importance (mutual information) ... this may take a moment")
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "open"]
X = df[feature_cols].fillna(0)
y = df["open"]
mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=True)
top = mi_series.tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
colors_mi = plt.cm.coolwarm(np.linspace(0.2, 0.95, len(top)))
ax.barh(top.index, top.values, color=colors_mi, edgecolor="#0f172a", linewidth=1)
for i, (val, name) in enumerate(zip(top.values, top.index)):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9, color="#e2e8f0")
ax.set_title("Top 15 Features - Mutual Information with Label",
             fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("MI Score")
_save(fig, "top_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════
# 10. BUSINESS TYPE BREAKDOWN (% of each category)
# ══════════════════════════════════════════════════════════════════════════
print("[10/10] Business type breakdown ...")
cat_cols = [c for c in df.columns if c.startswith("cat_")]
if cat_cols:
    cat_sums = df[cat_cols].sum().sort_values(ascending=False)
    cat_pct  = cat_sums / len(df)
    # Clean names: cat_restaurant -> Restaurant
    clean_names = [c.replace("cat_", "").replace("_", " ").title() for c in cat_pct.index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(cat_cols) * 0.4)))
    colors_cat = plt.cm.viridis(np.linspace(0.15, 0.85, len(cat_pct)))
    bars = ax.barh(clean_names[::-1], cat_pct.values[::-1],
                   color=colors_cat[::-1], edgecolor="#0f172a", linewidth=1)
    for bar, v in zip(bars, cat_pct.values[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.1%}", va="center", fontsize=9, color="#e2e8f0")
    ax.set_title("Business Type Distribution (% of Dataset)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Proportion of all rows")
    _save(fig, "business_type_breakdown.png")
else:
    print("  [WARN]  No cat_* columns found -- skipping.")

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  Done!  {len(os.listdir(OUT_DIR))} graphs saved to {OUT_DIR}/")
print(f"{'='*60}")
