#!/usr/bin/env python3
"""
Build feature matrix for Myanmar conflict prediction using PRIO-GRID.

Target: did this cell change faction next month? (1=yes, 0=no)

Joins Wikipedia panel and UCDP events using exact priogrid_gid.

Usage:
    python build_feature_matrix.py

Output:
    model/feature_matrix.csv

Requirements:
    pip install pandas numpy
"""

import os
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(BASE_DIR, 'extracting_ground_truth', 'panel_dataset_labelled.csv')
UCDP_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'ucdp', 'ucdp_labelled.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'model')
OUT_PATH   = os.path.join(OUT_DIR, 'feature_matrix.csv')
os.makedirs(OUT_DIR, exist_ok=True)

ACTOR_FLAGS = ['Junta','PDF','KIA','AA','Karen','Karenni','TNLA','MNDAA','Chin Resistance']

print("=" * 60)
print("Building feature matrix (PRIO-GRID)")
print("=" * 60)

# ── Step 1: Load panel ────────────────────────────────────────────────────
print("\nStep 1: Loading panel data...")
panel = pd.read_csv(PANEL_PATH)
panel['date'] = pd.to_datetime(panel['date'].str.replace('myanmar_', ''))
panel['year_month'] = panel['date'].dt.to_period('M')
panel = panel[panel['level1_side'] != 'Other'].copy()
print(f"  {len(panel):,} rows, {panel['priogrid_gid'].nunique()} unique PRIO-GRID cells")

cells = panel[['priogrid_gid','lat','lon']].drop_duplicates()
all_months = pd.period_range(panel['year_month'].min(), panel['year_month'].max(), freq='M')
print(f"  Monthly range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

# ── Step 2: Forward-fill monthly grid ────────────────────────────────────
print("\nStep 2: Building monthly grid (forward-fill)...")
cell_months = pd.MultiIndex.from_product(
    [cells['priogrid_gid'].values, all_months],
    names=['priogrid_gid','year_month']
).to_frame(index=False)
cell_months = cell_months.merge(cells, on='priogrid_gid', how='left')

panel_monthly = panel[['priogrid_gid','year_month','faction','level1_side','level2_group']].copy()
grid = cell_months.merge(panel_monthly, on=['priogrid_gid','year_month'], how='left')
grid = grid.sort_values(['priogrid_gid','year_month'])
grid[['faction','level1_side','level2_group']] = grid.groupby('priogrid_gid')[
    ['faction','level1_side','level2_group']
].ffill()
grid = grid.dropna(subset=['level1_side']).copy()
print(f"  {len(grid):,} rows after forward-fill")

# ── Step 3: Load UCDP ─────────────────────────────────────────────────────
print("\nStep 3: Loading UCDP events...")
ucdp = pd.read_csv(UCDP_PATH, parse_dates=['date_start'])
ucdp['year_month'] = ucdp['date_start'].dt.to_period('M')
print(f"  {len(ucdp):,} events")

# ── Step 4: Aggregate UCDP by priogrid_gid x month ───────────────────────
print("\nStep 4: Aggregating UCDP by PRIO-GRID cell x month...")
for actor in ACTOR_FLAGS:
    col = f'flag_{actor.lower().replace(" ","_")}'
    ucdp[col] = ((ucdp['wiki_label_a']==actor)|(ucdp['wiki_label_b']==actor)).astype(int)
flag_cols = [f'flag_{a.lower().replace(" ","_")}' for a in ACTOR_FLAGS]
ucdp['all_actors'] = ucdp['wiki_label_a'].fillna('')+'|'+ucdp['wiki_label_b'].fillna('')

cell_month_feats = ucdp.groupby(['priogrid_gid','year_month']).agg(
    n_events=('id','count'),
    total_deaths=('best','sum'),
    n_unique_groups=('all_actors', lambda x: len(set('|'.join(x).split('|'))-{'','None'})),
    junta_events=('flag_junta','sum'),
    anti_junta_events=('wiki_label_a', lambda x: (ucdp.loc[x.index,'level1_side_a']=='Anti-Junta').sum()),
    **{c:(c,'sum') for c in flag_cols}
).reset_index()

cell_month_feats['junta_event_share'] = (
    cell_month_feats['junta_events'] / cell_month_feats['n_events'].replace(0,np.nan)
).fillna(0)
cell_month_feats['anti_junta_event_share'] = (
    cell_month_feats['anti_junta_events'] / cell_month_feats['n_events'].replace(0,np.nan)
).fillna(0)

print(f"  UCDP covers {cell_month_feats['priogrid_gid'].nunique()} unique cells")

# ── Step 5: Merge onto grid ───────────────────────────────────────────────
print("\nStep 5: Merging features (exact priogrid_gid join)...")
df = grid.merge(cell_month_feats, on=['priogrid_gid','year_month'], how='left')
event_cols = ['n_events','total_deaths','n_unique_groups','junta_events',
              'anti_junta_events','junta_event_share','anti_junta_event_share'] + flag_cols
df[event_cols] = df[event_cols].fillna(0)
df['is_pro_junta'] = (df['level1_side']=='Pro-Junta').astype(int)

# ── Step 6: Lag and rolling features ─────────────────────────────────────
print("\nStep 6: Computing lag and rolling features...")
df = df.sort_values(['priogrid_gid','year_month'])
for col in ['n_events','total_deaths']:
    for lag in [1,3,6]:
        df[f'{col}_lag{lag}'] = df.groupby('priogrid_gid')[col].shift(lag)
    for window in [3,6]:
        df[f'{col}_roll{window}'] = df.groupby('priogrid_gid')[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

# ── Step 7: Conflict trend ────────────────────────────────────────────────
print("\nStep 7: Computing conflict trend...")
def rolling_slope(series):
    slopes = [np.nan]*len(series)
    arr = series.values
    for i in range(2, len(arr)):
        window = arr[max(0,i-2):i+1]
        mask = ~np.isnan(window)
        if mask.sum() >= 2:
            x = np.arange(len(window))
            slopes[i] = np.polyfit(x[mask], window[mask], 1)[0]
    return pd.Series(slopes, index=series.index)

df['event_trend_3m'] = df.groupby('priogrid_gid')['n_events'].transform(rolling_slope)

# ── Step 8: Months since last event ──────────────────────────────────────
print("\nStep 8: Computing months since last event...")
def months_since_event(series):
    result = []; count = np.nan
    for val in series:
        if val > 0: count = 0
        elif count is not np.nan: count += 1
        result.append(count)
    return pd.Series(result, index=series.index)

df['months_since_last_event'] = df.groupby('priogrid_gid')['n_events'].transform(months_since_event)

# ── Step 9: Months in current faction ────────────────────────────────────
print("\nStep 9: Computing months in current faction...")
def months_in_faction(series):
    result = []; count = 0; prev = None
    for val in series:
        if val != prev: count = 1
        else: count += 1
        result.append(count); prev = val
    return pd.Series(result, index=series.index)

df['months_in_current_faction'] = df.groupby('priogrid_gid')['level1_side'].transform(months_in_faction)

# ── Step 10: Neighbor features using PRIO-GRID adjacency ─────────────────
print("\nStep 10: Computing neighbor features (PRIO-GRID adjacency)...")

# In PRIO-GRID: neighbors differ by +-1 col (+-1 gid) or +-1 row (+-720 gid)
def get_pg_neighbors(gid):
    return [gid-721, gid-720, gid-719,
            gid-1,           gid+1,
            gid+719, gid+720, gid+721]

event_lookup   = df.set_index(['priogrid_gid','year_month'])['n_events'].to_dict()
faction_lookup = df.set_index(['priogrid_gid','year_month'])['level1_side'].to_dict()

neighbor_events  = []
neighbor_changed = []
n = len(df)

for idx, (_, row) in enumerate(df.iterrows()):
    if idx % 5000 == 0:
        print(f"  Progress: {idx}/{n}", end='\r')
    gid    = row['priogrid_gid']
    ym     = row['year_month']
    ym_prev  = ym - 1
    ym_prev2 = ym - 2
    neighbors = get_pg_neighbors(gid)

    n_ev = sum(event_lookup.get((ng, ym_prev), 0) for ng in neighbors)
    neighbor_events.append(n_ev)

    changed = 0
    for ng in neighbors:
        curr = faction_lookup.get((ng, ym_prev), None)
        prev = faction_lookup.get((ng, ym_prev2), None)
        if curr and prev and curr != prev:
            changed = 1; break
    neighbor_changed.append(changed)

print(f"  Progress: done{' '*20}")
df['neighbor_n_events']       = neighbor_events
df['neighbor_changed_faction'] = neighbor_changed

# ── Step 11: Attacker matches controller ─────────────────────────────────
print("\nStep 11: Computing faction match feature...")
df['attacker_matches_controller'] = (
    ((df['junta_event_share']>0.5) & (df['level1_side']=='Pro-Junta')) |
    ((df['anti_junta_event_share']>0.5) & (df['level1_side']=='Anti-Junta'))
).astype(int)

# ── Step 12: Target variable ──────────────────────────────────────────────
print("\nStep 12: Building target variable (changed)...")
df['level1_side_next'] = df.groupby('priogrid_gid')['level1_side'].shift(-1)
df['changed'] = (df['level1_side'] != df['level1_side_next']).astype(int)
df = df.dropna(subset=['level1_side_next']).copy()

print(f"\n  Change rate: {df['changed'].mean()*100:.2f}%")
print(f"  Changed=1 : {df['changed'].sum():,} rows")
print(f"  Changed=0 : {(df['changed']==0).sum():,} rows")

# Check UCDP overlap with changing cells
changed_cells = df[df['changed']==1]
print(f"\n  Among cells that changed:")
print(f"    With any events (this month): {(changed_cells['n_events']>0).mean()*100:.1f}%")
print(f"    With any lag1 events:         {(changed_cells['n_events_lag1']>0).mean()*100:.1f}%")
print(f"    With any lag3 events:         {(changed_cells['n_events_lag3']>0).mean()*100:.1f}%")

# ── Step 13: Train/validate split ─────────────────────────────────────────
print("\nStep 13: Creating temporal train/validate split...")
split_date = pd.Period('2025-10', freq='M')
df['split'] = np.where(df['year_month'] <= split_date, 'train', 'validate')

# ── Save ──────────────────────────────────────────────────────────────────
df['year_month'] = df['year_month'].astype(str)
df.to_csv(OUT_PATH, index=False)

print(f"\nDone.")
print(f"  Rows     : {len(df):,}")
print(f"  Columns  : {len(df.columns)}")
print(f"  Saved -> {OUT_PATH}")
print(f"\n  Train/validate split:")
print(df['split'].value_counts().to_string())
