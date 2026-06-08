#!/usr/bin/env python3
"""
No-Change Baseline for Myanmar Territorial Control

For each cell in the validation set, predicts the same label
as the previous month. Compares against U-Net results.

This directly addresses the professor's concern:
"Does the U-Net beat a model that just says nothing changed?"

Usage:
    python no_change_baseline.py

Requires:
    data/processed/10_target_variable.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".").resolve()
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
TARGET_PATH  = DATA_PROC / "10_target_variable.parquet"

# ── Temporal split (must match U-Net notebook exactly) ───────────────────
TRAIN_CUTOFF = "2025-09"   # train: up to and including Sep 2025
VAL_START    = "2025-10"   # validate: Oct 2025 onwards

# ── U-Net results for comparison (from notebook output) ──────────────────
UNET_RESULTS = {
    "accuracy":   0.918,
    "macro_f1":   0.899,
    "gov_f1":     0.940,
    "opo_f1":     0.887,
    "uncertain_f1": 0.869,
}

# ── Load target variable ──────────────────────────────────────────────────
print("Loading target variable...")
target = pd.read_parquet(TARGET_PATH)
target["year_month"] = target["year_month"].astype(str)
target = target[["priogrid_gid", "year_month", "target"]].copy()
target = target.sort_values(["priogrid_gid", "year_month"]).reset_index(drop=True)

print(f"  {len(target):,} rows | {target['priogrid_gid'].nunique()} cells | "
      f"{target['year_month'].nunique()} months")
print(f"  Range: {target['year_month'].min()} to {target['year_month'].max()}")

# ── Build no-change prediction ────────────────────────────────────────────
# For each cell x month, the no-change prediction = label from previous month
print("\nBuilding no-change predictions...")

target["pred_no_change"] = target.groupby("priogrid_gid")["target"].shift(1)

# Drop rows where we have no previous month (first month per cell)
target_with_pred = target.dropna(subset=["pred_no_change"]).copy()

# ── Split into train and validate ─────────────────────────────────────────
val = target_with_pred[target_with_pred["year_month"] >= VAL_START].copy()
train = target_with_pred[target_with_pred["year_month"] <= TRAIN_CUTOFF].copy()

print(f"\nTrain rows (with no-change pred): {len(train):,}")
print(f"Val rows   (with no-change pred): {len(val):,}")
print(f"\nValidation label distribution:")
print(val["target"].value_counts().to_string())

# ── Evaluate no-change baseline on validation set ─────────────────────────
y_true = val["target"].values
y_pred = val["pred_no_change"].values

acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred,
                                labels=["gov", "opo", "uncertain"],
                                target_names=["gov", "opo", "uncertain"],
                                digits=3, output_dict=True)
cm     = confusion_matrix(y_true, y_pred, labels=["gov", "opo", "uncertain"])

macro_f1    = report["macro avg"]["f1-score"]
gov_f1      = report["gov"]["f1-score"]
opo_f1      = report["opo"]["f1-score"]
uncertain_f1 = report["uncertain"]["f1-score"]

# ── How many cells actually changed? ─────────────────────────────────────
val["actually_changed"] = val["target"] != val["pred_no_change"]
n_changed = val["actually_changed"].sum()
pct_changed = n_changed / len(val) * 100

print(f"\n{'='*60}")
print("NO-CHANGE BASELINE RESULTS (validation set)")
print(f"{'='*60}")
print(f"\nCells that actually changed in validation: {n_changed} / {len(val)} ({pct_changed:.1f}%)")
print(f"\nAccuracy:  {acc:.3f}")
print(f"Macro F1:  {macro_f1:.3f}")
print(f"\nPer-class F1:")
print(f"  gov:       {gov_f1:.3f}")
print(f"  opo:       {opo_f1:.3f}")
print(f"  uncertain: {uncertain_f1:.3f}")

print(f"\nConfusion matrix (rows=actual, cols=predicted):")
print(f"              gov    opo  uncertain")
for i, cls in enumerate(["gov", "opo", "uncertain"]):
    print(f"  {cls:<10}  {cm[i][0]:4d}   {cm[i][1]:4d}   {cm[i][2]:4d}")

# ── Comparison table ──────────────────────────────────────────────────────
majority_baseline = val["target"].value_counts(normalize=True).iloc[0]

print(f"\n{'='*60}")
print("COMPARISON: No-Change vs U-Net vs Majority Class")
print(f"{'='*60}")
print(f"\n{'Model':<25} {'Accuracy':>9} {'Macro F1':>9}")
print(f"{'-'*45}")
print(f"{'Majority class':<25} {majority_baseline:>9.3f} {'N/A':>9}")
print(f"{'No-change baseline':<25} {acc:>9.3f} {macro_f1:>9.3f}")
print(f"{'U-Net (200 epochs)':<25} {UNET_RESULTS['accuracy']:>9.3f} {UNET_RESULTS['macro_f1']:>9.3f}")

print(f"\n{'Model':<25} {'gov F1':>8} {'opo F1':>8} {'uncertain F1':>13}")
print(f"{'-'*55}")
print(f"{'No-change':<25} {gov_f1:>8.3f} {opo_f1:>8.3f} {uncertain_f1:>13.3f}")
print(f"{'U-Net':<25} {UNET_RESULTS['gov_f1']:>8.3f} "
      f"{UNET_RESULTS['opo_f1']:>8.3f} "
      f"{UNET_RESULTS['uncertain_f1']:>13.3f}")

unet_beats = UNET_RESULTS['macro_f1'] > macro_f1
print(f"\n{'='*60}")
print(f"U-Net beats no-change baseline: {'YES' if unet_beats else 'NO'}")
if unet_beats:
    gain = UNET_RESULTS['macro_f1'] - macro_f1
    print(f"Macro F1 gain: +{gain:.3f}")
else:
    gap = macro_f1 - UNET_RESULTS['macro_f1']
    print(f"No-change is better by: {gap:.3f} macro F1")
    print("This suggests the U-Net may be memorising geography rather than learning dynamics.")
print(f"{'='*60}")

# ── Breakdown: accuracy on changed vs unchanged cells ────────────────────
print(f"\nBreakdown by cell stability:")
changed_cells   = val[val["actually_changed"] == True]
unchanged_cells = val[val["actually_changed"] == False]

# No-change accuracy on cells that actually changed (should be 0% by definition)
print(f"\n  Cells that changed   ({len(changed_cells):4d} rows):")
print(f"    No-change accuracy: 0.000 (by definition)")
if len(changed_cells) > 0:
    unet_on_changed = "unknown — run U-Net predictions on this subset"
    print(f"    U-Net accuracy:     {unet_on_changed}")

print(f"\n  Cells that did NOT change ({len(unchanged_cells):4d} rows):")
nc_acc_stable = accuracy_score(unchanged_cells["target"],
                                unchanged_cells["pred_no_change"])
print(f"    No-change accuracy: {nc_acc_stable:.3f} (perfect by definition)")