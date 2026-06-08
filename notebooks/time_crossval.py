#!/usr/bin/env python3
"""
Expanding Window Time Cross-Validation for Myanmar Territorial Control

Two versions:
    1. Full dataset (all 185 cells)
    2. Changing cells only (94 cells that change at least once)

For each fold:
    - Train on all months up to cutoff
    - Validate on all remaining months
    - Compute no-change baseline metrics

Usage:
    python time_crossval.py

Requires:
    data/processed/10_target_variable.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
TARGET_PATH  = DATA_PROC / "10_target_variable.parquet"

# ── Parameters ────────────────────────────────────────────────────────────
MIN_TRAIN_MONTHS = 12   # minimum months before first validation fold

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading target variable...")
target = pd.read_parquet(TARGET_PATH)
target["year_month"] = target["year_month"].astype(str)
target = target.sort_values(["priogrid_gid", "year_month"]).reset_index(drop=True)

all_months = sorted(target["year_month"].unique())
print(f"  {len(target):,} rows | {target['priogrid_gid'].nunique()} cells | "
      f"{len(all_months)} months")
print(f"  Range: {all_months[0]} to {all_months[-1]}")

# ── Identify changing cells ───────────────────────────────────────────────
target["prev"] = target.groupby("priogrid_gid")["target"].shift(1)
target["changed"] = target["prev"].notna() & (target["target"] != target["prev"])
changed_cells = set(target[target["changed"]]["priogrid_gid"].unique())
print(f"\n  Cells that change at least once: {len(changed_cells)}")

# ── Helper: evaluate no-change baseline on a validation set ───────────────
def evaluate_no_change(val_df):
    """
    For each cell-month in val_df, predict same label as previous month.
    Returns dict of metrics.
    """
    if len(val_df) == 0:
        return None

    y_true = val_df["target"].values
    y_pred = val_df["prev"].values

    # Only evaluate rows where we have a previous month
    mask = val_df["prev"].notna()
    if mask.sum() == 0:
        return None

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=["gov", "opo", "uncertain"], zero_division=0)
    n_changed = (y_true != y_pred).sum()
    n_total   = len(y_true)

    return {
        "n_rows":     n_total,
        "n_changed":  int(n_changed),
        "change_pct": round(n_changed / n_total * 100, 1),
        "accuracy":   round(acc, 3),
        "macro_f1":   round(macro_f1, 3),
    }


# ── Run expanding window cross-validation ─────────────────────────────────
def run_crossval(data, label="Full dataset"):
    print(f"\n{'='*60}")
    print(f"EXPANDING WINDOW CROSS-VALIDATION — {label}")
    print(f"{'='*60}")
    print(f"Minimum training months: {MIN_TRAIN_MONTHS}")
    print(f"Total months available:  {len(all_months)}")

    results = []

    for i in range(MIN_TRAIN_MONTHS, len(all_months)):
        train_months = all_months[:i]
        val_months   = all_months[i:]
        cutoff       = all_months[i-1]   # last training month

        train = data[data["year_month"].isin(train_months)]
        val   = data[data["year_month"].isin(val_months)]

        metrics = evaluate_no_change(val)
        if metrics is None:
            continue

        metrics["cutoff"]       = cutoff
        metrics["n_train_months"] = len(train_months)
        metrics["n_val_months"]   = len(val_months)
        metrics["n_train_rows"]   = len(train)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Print fold-by-fold results
    print(f"\n{'Cutoff':<10} {'TrainM':>7} {'ValM':>5} {'ValRows':>8} "
          f"{'Changed':>8} {'Change%':>8} {'Accuracy':>9} {'MacroF1':>9}")
    print("-" * 75)
    for _, r in results_df.iterrows():
        print(f"{r['cutoff']:<10} {r['n_train_months']:>7} {r['n_val_months']:>5} "
              f"{r['n_rows']:>8} {r['n_changed']:>8} {r['change_pct']:>7.1f}% "
              f"{r['accuracy']:>9.3f} {r['macro_f1']:>9.3f}")

    # Summary statistics
    print(f"\n{'Summary':}")
    print(f"  Mean accuracy:   {results_df['accuracy'].mean():.3f} "
          f"(+/- {results_df['accuracy'].std():.3f})")
    print(f"  Mean macro F1:   {results_df['macro_f1'].mean():.3f} "
          f"(+/- {results_df['macro_f1'].std():.3f})")
    print(f"  Mean change%:    {results_df['change_pct'].mean():.1f}%")
    print(f"  Min macro F1:    {results_df['macro_f1'].min():.3f} "
          f"(fold: {results_df.loc[results_df['macro_f1'].idxmin(), 'cutoff']})")
    print(f"  Max macro F1:    {results_df['macro_f1'].max():.3f} "
          f"(fold: {results_df.loc[results_df['macro_f1'].idxmax(), 'cutoff']})")

    return results_df


# ── Version 1: Full dataset ───────────────────────────────────────────────
results_full = run_crossval(target, label="All 185 cells")

# ── Version 2: Changing cells only ───────────────────────────────────────
target_changing = target[target["priogrid_gid"].isin(changed_cells)].copy()
results_changing = run_crossval(target_changing,
                                label=f"Changing cells only ({len(changed_cells)} cells)")

# ── Final comparison ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY COMPARISON")
print(f"{'='*60}")
print(f"\n{'Dataset':<30} {'Mean Acc':>9} {'Mean F1':>9} {'Mean Change%':>13}")
print(f"{'-'*63}")
print(f"{'Full (185 cells)':<30} "
      f"{results_full['accuracy'].mean():>9.3f} "
      f"{results_full['macro_f1'].mean():>9.3f} "
      f"{results_full['change_pct'].mean():>12.1f}%")
print(f"{'Changing only (94 cells)':<30} "
      f"{results_changing['accuracy'].mean():>9.3f} "
      f"{results_changing['macro_f1'].mean():>9.3f} "
      f"{results_changing['change_pct'].mean():>12.1f}%")
print(f"\nU-Net single split:            acc=0.918   macro F1=0.899")
print(f"\nKey question: does U-Net beat the no-change baseline")
print(f"on the changing-cells-only dataset?")
