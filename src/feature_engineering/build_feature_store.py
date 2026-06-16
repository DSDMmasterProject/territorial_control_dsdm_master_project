#!/usr/bin/env python
# build_feature_store.py
# Build the full feature store — all available features in one place.
# The model selects a subset via INPUT_FEATURES; this file produces everything.
#
# Inputs:
#   data/processed/myanmar_priogrid_features.csv     ← base UCDP aggregations
#   data/processed/priogrid_static_myanmar.csv       ← terrain/resources (optional)
#   data/processed/priogrid_yearly_myanmar.csv       ← yearly geo features (optional)
#
# Output:
#   data/processed/myanmar_feature_store.csv

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / 'data' / 'processed'
OUT_PATH     = DATA_PROC / 'myanmar_feature_store.csv'

# ── Load base feature matrix ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PROC / 'myanmar_priogrid_features.csv')
df = df.sort_values(['priogrid_gid', 'year_month']).reset_index(drop=True)
df['year_month'] = df['year_month'].astype(str)
print(f'Base feature matrix: {df.shape}  ({df["priogrid_gid"].nunique()} cells, '
      f'{df["year_month"].nunique()} months)')

# ── Geo enrichment (static + yearly PRIOGRID features) ───────────────────────
# Skipped silently if catalog files have not been generated yet.
static_path = DATA_PROC / 'priogrid_static_myanmar.csv'
yearly_path = DATA_PROC / 'priogrid_yearly_myanmar.csv'

if static_path.exists():
    static = pd.read_csv(static_path)
    static_cols = [c for c in static.columns if c != 'priogrid_gid']
    df = df.merge(static, on='priogrid_gid', how='left')
    df[static_cols] = df[static_cols].fillna(0.0)
    print(f'Static geo features added ({len(static_cols)}): {static_cols}')
else:
    print('Static geo features skipped — priogrid_static_myanmar.csv not found.')

if yearly_path.exists():
    yearly = pd.read_csv(yearly_path)
    yearly_cols = [c for c in yearly.columns if c not in ['priogrid_gid', 'year']]
    yearly_for_merge = yearly.drop(columns=['year'], errors='ignore')
    df = df.merge(yearly_for_merge, on='priogrid_gid', how='left')
    df[yearly_cols] = df[yearly_cols].fillna(0.0)
    print(f'Yearly geo features added ({len(yearly_cols)}): {yearly_cols}')
else:
    print('Yearly geo features skipped — priogrid_yearly_myanmar.csv not found.')

# ── Extended temporal features ────────────────────────────────────────────────
# The base matrix already has lag1/3/6 and roll3m/6m for a handful of columns.
# We extend here with: lag9/12, roll12m, rolling std, trend, actor-level rolling.

# Columns to build extended lags and rolling stats on
LAG_COLS  = [
    'total_events', 'total_fatalities',
    'events_gov', 'events_nug', 'gov_vs_civilians', 'is_onesided',
]
# Columns to add actor-level rolling features (avg + std)
ACTOR_ROLL_COLS = [
    'events_gov', 'events_nug', 'events_kio', 'events_ula',
    'events_knu', 'events_mndaa', 'events_pslf',
]
ROLL_WINDOWS = [3, 6, 12]

def group_shift(df, col, lag):
    return df.groupby('priogrid_gid')[col].shift(lag).fillna(0.0)

def group_roll_mean(df, col, window):
    return (
        df.groupby('priogrid_gid')[col]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

def group_roll_std(df, col, window):
    return (
        df.groupby('priogrid_gid')[col]
        .transform(lambda x: x.rolling(window, min_periods=2).std().fillna(0.0))
    )

# Extended lags: 9 and 12 months (base already covers 1/3/6)
print('\nAdding extended lags (9, 12 months)...')
for lag in [9, 12]:
    for col in LAG_COLS:
        col_name = f'{col}_lag{lag}'
        if col_name not in df.columns:
            df[col_name] = group_shift(df, col, lag)
print(f'  Done — {len(LAG_COLS) * 2} new lag columns')

# Rolling 12m mean for total_events and total_fatalities (base covers 3m/6m)
print('Adding roll12m mean for total_events and total_fatalities...')
for col in ['total_events', 'total_fatalities']:
    col_name = f'{col}_roll12m'
    if col_name not in df.columns:
        df[col_name] = group_roll_mean(df, col, 12)

# Actor-level rolling averages at 3m, 6m, 12m
print('Adding actor-level rolling averages (3m, 6m, 12m)...')
n_actor_roll = 0
for col in ACTOR_ROLL_COLS:
    for window in ROLL_WINDOWS:
        col_name = f'{col}_roll{window}m'
        if col_name not in df.columns:
            df[col_name] = group_roll_mean(df, col, window)
            n_actor_roll += 1
print(f'  Done — {n_actor_roll} new columns')

# Rolling standard deviation — captures conflict volatility
# High std = contested/transitioning area; low std = stable
print('Adding rolling std (3m, 6m) for key columns...')
STD_COLS = ['total_events', 'total_fatalities', 'events_gov', 'events_nug']
for col in STD_COLS:
    for window in [3, 6]:
        df[f'{col}_std{window}m'] = group_roll_std(df, col, window)
print(f'  Done — {len(STD_COLS) * 2} new columns')

# Trend features: roll3m − roll6m (positive = escalating, negative = de-escalating)
print('Adding trend features (roll3m − roll6m)...')
TREND_COLS = ['total_events', 'total_fatalities', 'events_gov', 'events_nug']
for col in TREND_COLS:
    df[f'{col}_trend'] = group_roll_mean(df, col, 3) - group_roll_mean(df, col, 6)
print(f'  Done — {len(TREND_COLS)} new columns')

# Actor event shares (extend beyond gov/nug which are already in base)
print('Adding actor event shares (kio, ula, knu)...')
for actor in ['kio', 'ula', 'knu']:
    col = f'events_{actor}'
    df[f'{actor}_event_share'] = np.where(
        df['total_events'] > 0,
        df[col] / df['total_events'],
        0.0,
    )

# ── Summary ───────────────────────────────────────────────────────────────────
base_cols  = pd.read_csv(DATA_PROC / 'myanmar_priogrid_features.csv', nrows=0).columns.tolist()
new_cols   = [c for c in df.columns if c not in base_cols]
meta_cols  = ['priogrid_gid', 'year_month', 'year_month_dt', 'adm_1']
all_feat   = [c for c in df.columns if c not in meta_cols]

print(f'\n{"=" * 60}')
print('FEATURE STORE SUMMARY')
print(f'{"=" * 60}')
print(f'  Rows             : {len(df):,}')
print(f'  Cells            : {df["priogrid_gid"].nunique()}')
print(f'  Months           : {df["year_month"].nunique()}  '
      f'({df["year_month"].min()} → {df["year_month"].max()})')
print(f'  Total features   : {len(all_feat)}')
print(f'  Base features    : {len(all_feat) - len(new_cols)}')
print(f'  New features     : {len(new_cols)}')
print(f'\nNew columns added:')
for c in new_cols:
    print(f'  {c}')

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f'\nFeature store saved: {OUT_PATH}')
print(f'  {len(df.columns)} total columns  |  {len(df):,} rows')
print('\nCOMPLETE.')
