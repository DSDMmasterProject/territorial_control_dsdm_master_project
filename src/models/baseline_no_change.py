#!/usr/bin/env python
# baseline_no_change.py
# Persistence baseline: for each cell, predict the last known control state
# carried forward from the FIRST available month in the full time series —
# the carry-forward does NOT restart at TRAIN_CUTOFF.
#
# The persistence history uses the full target series (all months, all cells),
# but only cells in the valid-GID set (= same cells focal_mask writes) are
# written to predictions.csv and evaluated.
#
# Fallback for cells with no prior observation: training majority class.
#
# Inputs:
#   data/processed/10_target_variable.parquet
#   data/processed/myanmar_feature_store.csv   (for valid-GID computation only)
#
# Outputs:
#   models/baseline_no_change/predictions.csv

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
SEED         = 20269999
TRAIN_CUTOFF = '2025-06'

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DATA_PROC        = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR          = PROJECT_ROOT / 'models' / 'baseline_no_change'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH = OUT_DIR / 'predictions.csv'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.feature_selection import build_valid_gids

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_MAP   = {'gov': 0, 'opo': 1, 'uncertain': 2}
CLASS_NAMES = ['gov', 'opo', 'uncertain']

# ── Load data ──────────────────────────────────────────────────────────────────
target = pd.read_parquet(DATA_PROC / '10_target_variable.parquet')
target['year_month'] = target['year_month'].astype(str)

features_meta = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv',
                             usecols=['priogrid_gid'])

# ── Valid GID set (matches focal_mask's export exactly) ───────────────────────
valid_gids = build_valid_gids(
    set(features_meta['priogrid_gid'].unique()),
    set(target['priogrid_gid'].unique()),
)
n_before = len(target)
print(f'Target rows (full series): {n_before:,}'
      f'  |  months: {target["year_month"].nunique()}'
      f'  |  cells: {target["priogrid_gid"].nunique()}')

# ── Majority class from training set (fallback for unseen cells) ───────────────
train_data     = target[target['year_month'] <= TRAIN_CUTOFF]
majority_class = train_data['target'].value_counts().idxmax()
print(f'Majority class (training fallback): {majority_class}')

# ── Build persistence predictions over the FULL time series ───────────────────
# Sort the complete series by cell then time.
# shift(1) within each cell gives the immediately preceding month's state;
# ffill() propagates earlier states across any gaps.
# fillna() handles a cell's very first observation (no prior state).
df = target[['priogrid_gid', 'year_month', 'target']].copy()
df = df.sort_values(['priogrid_gid', 'year_month']).reset_index(drop=True)

df['pred_class'] = df.groupby('priogrid_gid')['target'].shift(1)
df['pred_class'] = df.groupby('priogrid_gid')['pred_class'].ffill()
df['pred_class'] = df['pred_class'].fillna(majority_class)

# ── Restrict output to the same GID set as focal_mask ─────────────────────────
df = df[df['priogrid_gid'].isin(valid_gids)].copy()
print(f'Dropped {n_before - len(df)} out-of-bounds rows'
      f'  |  remaining: {len(df):,}')

df['pred_class_int'] = df['pred_class'].map(LABEL_MAP)
df['true_class']     = df['target']

for cls in CLASS_NAMES:
    df[f'prob_{cls}'] = (df['pred_class'] == cls).astype(float)
df['confidence']     = 1.0
df['has_label']      = True
df['is_train_month'] = df['year_month'] <= TRAIN_CUTOFF

pred_df = df[[
    'priogrid_gid', 'year_month', 'pred_class', 'pred_class_int',
    'prob_gov', 'prob_opo', 'prob_uncertain', 'confidence',
    'true_class', 'has_label', 'is_train_month',
]].copy()

pred_df.to_csv(PREDICTIONS_PATH, index=False)
print(f'Predictions saved: {PREDICTIONS_PATH}  ({len(pred_df):,} rows)')
print(f'Val rows: {(~pred_df["is_train_month"]).sum():,}')

val_df = pred_df[~pred_df['is_train_month']]
print('\nPrediction class distribution (val):')
print(val_df['pred_class'].value_counts())
print('\nTrue class distribution (val):')
print(val_df['true_class'].value_counts())
print('COMPLETE.')
