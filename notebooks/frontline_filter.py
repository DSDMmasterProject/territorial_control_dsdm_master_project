#!/usr/bin/env python3
"""
Frontline Cell Filter for Myanmar Territorial Control

Removes cells that:
- Never changed control status during the full time period, AND
- Are not a direct neighbor (8-connected PRIO-GRID) of any cell that changed

This forces the model to focus on contested frontline zones rather than
stable background territory, making the evaluation more meaningful.

Usage:
    python frontline_filter.py

Output:
    data/processed/myanmar_frontline_cells.csv     list of kept cell GIDs
    data/processed/10_target_variable_filtered.parquet
    data/processed/myanmar_priogrid_features_filtered.csv

Requirements:
    pip install pandas pyarrow
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_PROC     = PROJECT_ROOT / "data" / "processed"
TARGET_PATH   = DATA_PROC / "10_target_variable.parquet"
FEATURES_PATH = DATA_PROC / "myanmar_priogrid_features.csv"

# ── Temporal split ────────────────────────────────────────────────────────
VAL_START = "2025-10"

# ── Step 1: Load target variable ──────────────────────────────────────────
print("Loading target variable...")
target = pd.read_parquet(TARGET_PATH)
target["year_month"] = target["year_month"].astype(str)
target = target.sort_values(["priogrid_gid", "year_month"]).reset_index(drop=True)

all_cells = set(target["priogrid_gid"].unique())
print(f"  {len(target):,} rows | {len(all_cells)} cells | "
      f"{target['year_month'].nunique()} months")

# ── Step 2: Identify cells that changed at least once ────────────────────
print("\nIdentifying changing cells...")
target["prev_target"] = target.groupby("priogrid_gid")["target"].shift(1)
target["changed"] = (
    target["prev_target"].notna() &
    (target["target"] != target["prev_target"])
)

changed_cells = set(target[target["changed"]]["priogrid_gid"].unique())
print(f"  Cells that changed at least once: {len(changed_cells)}")

# ── Step 3: Find neighbors of changing cells ──────────────────────────────
print("Finding neighbors of changing cells...")

def get_neighbors(gid):
    """8-connected PRIO-GRID neighbors."""
    return {gid-721, gid-720, gid-719,
            gid-1,           gid+1,
            gid+719, gid+720, gid+721}

neighbor_cells = set()
for gid in changed_cells:
    neighbor_cells.update(get_neighbors(gid) & all_cells)

keep_cells   = changed_cells | neighbor_cells
remove_cells = all_cells - keep_cells

print(f"  Cells that changed:              {len(changed_cells)}")
print(f"  Neighbor cells to keep:          {len(neighbor_cells - changed_cells)}")
print(f"  Total kept:                      {len(keep_cells)} / {len(all_cells)}")
print(f"  Removed (stable, isolated):      {len(remove_cells)}")

# ── Step 4: Filter target variable ───────────────────────────────────────
print("\nFiltering target variable...")
target_filtered = target[target["priogrid_gid"].isin(keep_cells)].copy()
target_filtered = target_filtered.drop(columns=["prev_target", "changed"])

print(f"  Full dataset:     {len(target):,} rows")
print(f"  Filtered dataset: {len(target_filtered):,} rows")

# Save
out_target = DATA_PROC / "10_target_variable_filtered.parquet"
target_filtered.to_parquet(out_target, index=False)
print(f"  Saved -> {out_target}")

# ── Step 5: Filter features ───────────────────────────────────────────────
if FEATURES_PATH.exists():
    print("\nFiltering features...")
    features = pd.read_csv(FEATURES_PATH)
    features["year_month"] = features["year_month"].astype(str)
    features_filtered = features[features["priogrid_gid"].isin(keep_cells)].copy()
    print(f"  Full dataset:     {len(features):,} rows")
    print(f"  Filtered dataset: {len(features_filtered):,} rows")
    out_features = DATA_PROC / "myanmar_priogrid_features_filtered.csv"
    features_filtered.to_csv(out_features, index=False)
    print(f"  Saved -> {out_features}")
else:
    print("\nFeatures file not found — skipping feature filter")

# ── Step 6: Save cell list ────────────────────────────────────────────────
cells_df = pd.DataFrame({
    "priogrid_gid": sorted(keep_cells),
    "changed":      [gid in changed_cells for gid in sorted(keep_cells)],
    "is_neighbor":  [gid not in changed_cells for gid in sorted(keep_cells)],
})
out_cells = DATA_PROC / "myanmar_frontline_cells.csv"
cells_df.to_csv(out_cells, index=False)
print(f"\nCell list saved -> {out_cells}")

# ── Step 7: No-change baseline on filtered validation set ─────────────────
print(f"\n{'='*60}")
print("NO-CHANGE BASELINE — FILTERED DATASET")
print(f"{'='*60}")

target_filtered = target_filtered.sort_values(["priogrid_gid", "year_month"])
target_filtered["prev"] = target_filtered.groupby("priogrid_gid")["target"].shift(1)
target_filtered_with_pred = target_filtered.dropna(subset=["prev"]).copy()

val = target_filtered_with_pred[
    target_filtered_with_pred["year_month"] >= VAL_START
].copy()

print(f"\nValidation rows (filtered): {len(val):,}")
print(f"Label distribution:")
print(val["target"].value_counts().to_string())

val["actually_changed"] = val["target"] != val["prev"]
n_changed = val["actually_changed"].sum()
print(f"\nCells that changed in validation: {n_changed} / {len(val)} "
      f"({n_changed/len(val)*100:.1f}%)")

y_true = val["target"].values
y_pred = val["prev"].values

acc      = accuracy_score(y_true, y_pred)
report   = classification_report(y_true, y_pred,
                                  labels=["gov", "opo", "uncertain"],
                                  target_names=["gov", "opo", "uncertain"],
                                  digits=3, output_dict=True)
macro_f1 = report["macro avg"]["f1-score"]

print(f"\nNo-change accuracy:  {acc:.3f}")
print(f"No-change macro F1:  {macro_f1:.3f}")
print(f"\nPer-class F1:")
for cls in ["gov", "opo", "uncertain"]:
    print(f"  {cls:<10}: {report[cls]['f1-score']:.3f}")

# ── Step 8: Compare with original unfiltered baseline ────────────────────
print(f"\n{'='*60}")
print("COMPARISON: Filtered vs Unfiltered No-Change Baseline")
print(f"{'='*60}")
print(f"\n{'Dataset':<25} {'Val rows':>9} {'Change%':>8} {'Accuracy':>9} {'Macro F1':>9}")
print(f"{'-'*55}")
print(f"{'Unfiltered':<25} {'1,110':>9} {'2.8%':>8} {'0.972':>9} {'0.967':>9}")
print(f"{'Filtered':<25} {len(val):>9,} "
      f"{n_changed/len(val)*100:>7.1f}% "
      f"{acc:>9.3f} {macro_f1:>9.3f}")
print(f"\nU-Net (200 epochs):           accuracy=0.918  macro F1=0.899")
print(f"\nFor U-Net to be meaningful it needs to beat the filtered no-change baseline.")
