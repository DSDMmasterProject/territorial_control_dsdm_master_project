# ============================================================
# build_training_dataset.py
# Join X features + y labels → training dataset
# This is the final step before modelling.
#
# Inputs:  data/processed/myanmar_priogrid_features.csv  (X)
#          data/processed/myanmar_priogrid_labels.csv    (y)
# Outputs: data/processed/myanmar_training_dataset.csv  ← for model training
#          data/processed/myanmar_prediction_dataset.csv ← full panel for inference
# ============================================================

# The script loads the two files we've built — the 12,338-row feature matrix and the 5,238-row label vector — and joins them on their shared key of cell ID and year-month.
# It produces two outputs: the prediction dataset contains every cell-month with features attached and labels wherever they exist (NaN otherwise), which is what you run a trained model against to generate control maps for the full 2021–2026 period.
# The training dataset contains only the rows where both features and labels are present, split temporally into a training set (November 2023 to September 2025) and a held-out validation set (October 2025 to March 2026), following the conflict research convention of simulating real deployment by training on history and testing on the future.
# It closes by printing the exact Python code your teammates need to load and use the data.

import pathlib  # pathlib: file path handling

import numpy as np  # numpy: numerical operations
import pandas as pd  # pandas: data manipulation

# ── 1. PATHS ──────────────────────────────────────────────────

X_PATH = pathlib.Path(
    "data/processed/myanmar_priogrid_features.csv"
)  # Feature matrix (X)
Y_PATH = pathlib.Path("data/processed/myanmar_priogrid_labels.csv")  # Label vector (y)
OUT_DIR = pathlib.Path("data/processed")
TRAIN_PATH = (
    OUT_DIR / "myanmar_training_dataset.csv"
)  # Labelled rows only (for model training)
PRED_PATH = (
    OUT_DIR / "myanmar_prediction_dataset.csv"
)  # All rows (for generating predictions)

# ── 2. LOAD X AND Y ───────────────────────────────────────────

print("=" * 60)
print("Building training dataset (X + y)")
print("=" * 60)

X = pd.read_csv(X_PATH, encoding="utf-8")  # Load feature matrix (12,338 rows)
y = pd.read_csv(Y_PATH, encoding="utf-8")  # Load label vector (5,238 rows)

print(f"\n✅ Feature matrix (X): {len(X):,} rows × {len(X.columns)} columns")
print(f"✅ Label vector   (y): {len(y):,} rows × {len(y.columns)} columns")

# Convert year_month to string in both (ensure consistent join key format)
X["year_month"] = X["year_month"].astype(str)  # Ensure string type for joining
y["year_month"] = y["year_month"].astype(str)  # Same for labels

# ── 3. BUILD PREDICTION DATASET (X + y where available) ───────
#
# The prediction dataset = ALL 12,338 cell-months with features.
# Where a Wikipedia label exists, we attach it.
# Where no label exists (pre-Nov 2023 or unlabelled cells), label = NaN.
# This is what you run the model on to generate predictions.

print("\n[A] Building prediction dataset (all cell-months)...")

pred = X.merge(  # Left join: keep all X rows
    y[
        [
            "priogrid_gid",
            "year_month",
            "control_status",  # Select only the columns we need from y
            "label_confidence",
            "n_towns_in_cell",
            "is_direct_snapshot",
        ]
    ],
    on=["priogrid_gid", "year_month"],  # Join key: cell AND month must match
    how="left",  # left: keep all X rows, NaN where no y label
)

labelled_mask = pred[
    "control_status"
].notna()  # Boolean mask: True where we have a label
print(f"   Total rows:          {len(pred):,}")
print(
    f"   Rows with label:     {labelled_mask.sum():,} ({100 * labelled_mask.mean():.1f}%)"
)
print(
    f"   Rows without label:  {(~labelled_mask).sum():,} ({100 * (~labelled_mask).mean():.1f}%)"
)

# ── 4. BUILD TRAINING DATASET (labelled rows only) ────────────
#
# The training dataset = only rows where BOTH features AND labels exist.
# This is what you pass to the model for fitting and evaluation.

print("\n[B] Building training dataset (labelled rows only)...")

train = pred[labelled_mask].copy()  # Keep only rows with a label

print(f"   Training rows: {len(train):,}")
print(f"   PRIOGRID cells: {train['priogrid_gid'].nunique()}")
print(f"   Months covered: {train['year_month'].nunique()}")

# ── 5. ADD TEMPORAL TRAIN/VALIDATION SPLIT ────────────────────
#
# For conflict models, a TEMPORAL split is more rigorous than random:
# - Train on earlier months → test on later months
# - This simulates real deployment: you train on history, predict the future
# - Avoids data leakage (future information bleeding into training)
#
# Split point: train = Nov 2023 – Sep 2025 (first ~22 months, ~75%)
#              validate = Oct 2025 – Mar 2026 (last ~6 months, ~25%)
#
# This gives ~6 months of held-out validation — enough for meaningful evaluation.

SPLIT_DATE = "2025-10"  # First month of validation set

train["split"] = np.where(  # np.where: conditional assignment
    train["year_month"] < SPLIT_DATE,  # If month is before split date
    "train",  # Label it "train"
    "validate",  # Otherwise label it "validate"
)

n_train = (train["split"] == "train").sum()  # Count training rows
n_validate = (train["split"] == "validate").sum()  # Count validation rows
print(f"\n   Temporal split at {SPLIT_DATE}:")
print(f"   Train set:    {n_train:,} rows ({100 * n_train / len(train):.1f}%)")
print(f"   Validate set: {n_validate:,} rows ({100 * n_validate / len(train):.1f}%)")

# ── 6. PRINT COMPREHENSIVE SUMMARY ────────────────────────────

print("\n── Training Dataset Summary ──")

print("\n   Class distribution (full training set):")
for status, count in train["control_status"].value_counts().items():
    pct = 100 * count / len(train)
    bar = "█" * int(pct / 2)
    print(f"   {status:<22} {count:>5,}  ({pct:4.1f}%)  {bar}")

print("\n   Class distribution — TRAIN split:")
tr = train[train["split"] == "train"]["control_status"].value_counts()
for status, count in tr.items():
    print(f"   {status:<22} {count:>5,}")

print("\n   Class distribution — VALIDATE split:")
val = train[train["split"] == "validate"]["control_status"].value_counts()
for status, count in val.items():
    print(f"   {status:<22} {count:>5,}")

print("\n   Label confidence in training set:")
print(train["label_confidence"].value_counts().to_string())

print("\n   Feature columns (X):")
feature_cols = [
    c
    for c in train.columns
    if c  # Identify feature columns
    not in [
        "priogrid_gid",
        "year_month",
        "year_month_dt",
        "adm_1",  # Exclude metadata columns
        "control_status",
        "label_confidence",
        "n_towns_in_cell",  # Exclude label columns
        "high_conf_towns",
        "is_direct_snapshot",
        "split",
    ]
]
print(f"   {len(feature_cols)} features: {feature_cols}")

print("\n   Class imbalance check:")
majority_class_pct = train["control_status"].value_counts(normalize=True).iloc[0] * 100
print(f"   Majority class (government) = {majority_class_pct:.1f}% of training data")
print(
    f"   → Baseline accuracy (always predict 'government'): {majority_class_pct:.1f}%"
)
print("   → Your model must beat this baseline to add value")

# ── 7. SAVE BOTH DATASETS ─────────────────────────────────────

train.to_csv(TRAIN_PATH, index=False, encoding="utf-8")  # Save training dataset
pred.to_csv(PRED_PATH, index=False, encoding="utf-8")  # Save prediction dataset

print(f"\n✅ Training dataset:   {TRAIN_PATH}  ({len(train):,} rows)")
print(f"✅ Prediction dataset: {PRED_PATH}  ({len(pred):,} rows)")

# ── 8. PRINT SHARE INSTRUCTIONS ───────────────────────────────

print(f"""
{"=" * 60}
COMPLETE — READY TO SHARE WITH TEAMMATES
{"=" * 60}

FILES TO SHARE (from data/processed/):
  myanmar_training_dataset.csv    ← model training + validation
  myanmar_prediction_dataset.csv  ← generate predictions for all months
  myanmar_priogrid_features.csv   ← raw X matrix (if they want to rebuild)
  myanmar_priogrid_labels.csv     ← raw y vector (if they want to rebuild)

HOW TO USE (tell teammates):
  X columns = all features listed above ({len(feature_cols)} total)
  y column  = 'control_status'  (4 classes)
  split     = 'train' | 'validate'  (temporal split at {SPLIT_DATE})

  Standard usage:
    train_df = pd.read_csv('myanmar_training_dataset.csv')
    X_train  = train_df[train_df['split']=='train'][feature_cols]
    y_train  = train_df[train_df['split']=='train']['control_status']
    X_val    = train_df[train_df['split']=='validate'][feature_cols]
    y_val    = train_df[train_df['split']=='validate']['control_status']

BASELINE TO BEAT:
  Always-predict-government accuracy: {majority_class_pct:.1f}%
  Your model must exceed this to demonstrate value.
{"=" * 60}
""")
