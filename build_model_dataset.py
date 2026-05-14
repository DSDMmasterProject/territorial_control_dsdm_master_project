#!/usr/bin/env python3
"""
Build model dataset for binary classification proof of concept.

For each grid cell and each snapshot transition (T -> T+1), creates one row with:
- Target: did the cell change faction? (1/0)
- Features: UCDP conflict events within 0.5 degrees and 30 days before snapshot T

Usage:
    python build_model_dataset.py

Output:
    model/model_dataset.csv

Requirements:
    pip install pandas numpy
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH  = os.path.join(BASE_DIR, 'extracting_ground_truth', 'panel_dataset_labelled.csv')
UCDP_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'ucdp', 'ucdp_labelled.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'model')
OUT_PATH    = os.path.join(OUT_DIR, 'model_dataset.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────
RADIUS_DEG  = 0.5   # spatial radius for UCDP event matching (~55km)
WINDOW_DAYS = 30    # days before snapshot T to look for events

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading panel dataset...")
panel = pd.read_csv(PANEL_PATH)
panel['date'] = pd.to_datetime(panel['date'].str.replace('myanmar_', ''))
print(f"  {len(panel):,} rows, {panel['date'].nunique()} snapshots")

print("Loading UCDP events...")
ucdp = pd.read_csv(UCDP_PATH, parse_dates=['date_start'])
print(f"  {len(ucdp):,} events")

# ── Get sorted snapshot dates ─────────────────────────────────────────────
snapshots = sorted(panel['date'].unique())
print(f"\nSnapshots: {len(snapshots)} dates from {snapshots[0].date()} to {snapshots[-1].date()}")

# ── Build transition pairs (T -> T+1) ────────────────────────────────────
print("\nBuilding model dataset...")
print(f"  Radius: {RADIUS_DEG} degrees | Window: {WINDOW_DAYS} days")

all_rows = []
n_transitions = len(snapshots) - 1

for i in range(n_transitions):
    date_t  = snapshots[i]
    date_t1 = snapshots[i + 1]
    window_start = date_t - timedelta(days=WINDOW_DAYS)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Transition {i+1}/{n_transitions}: {date_t.date()} -> {date_t1.date()}")

    # Grid at T and T+1
    grid_t  = panel[panel['date'] == date_t][['lat', 'lon', 'faction', 'level1_side', 'level2_group']].copy()
    grid_t1 = panel[panel['date'] == date_t1][['lat', 'lon', 'faction']].copy()
    grid_t1 = grid_t1.rename(columns={'faction': 'faction_t1'})

    # Merge T and T+1 on cell coordinates
    merged = grid_t.merge(grid_t1, on=['lat', 'lon'], how='inner')
    merged = merged.rename(columns={
        'faction':     'faction_t',
        'level1_side': 'level1_side_t',
        'level2_group':'level2_group_t',
    })
    merged['date_t']  = date_t
    merged['date_t1'] = date_t1
    merged['changed'] = (merged['faction_t'] != merged['faction_t1']).astype(int)

    # UCDP events in the 30-day window before T
    ucdp_window = ucdp[
        (ucdp['date_start'] >= window_start) &
        (ucdp['date_start'] <  date_t)
    ][['latitude', 'longitude', 'best', 'level1_side_a']].copy()

    if len(ucdp_window) == 0:
        merged['n_events']          = 0
        merged['total_deaths']      = 0
        merged['pro_junta_events']  = 0
        merged['anti_junta_events'] = 0
    else:
        # For each cell, count events within RADIUS_DEG
        cell_lats = merged['lat'].values
        cell_lons = merged['lon'].values
        ev_lats   = ucdp_window['latitude'].values
        ev_lons   = ucdp_window['longitude'].values
        ev_deaths = ucdp_window['best'].values
        ev_sides  = ucdp_window['level1_side_a'].values

        n_events          = np.zeros(len(merged), dtype=int)
        total_deaths      = np.zeros(len(merged), dtype=float)
        pro_junta_events  = np.zeros(len(merged), dtype=int)
        anti_junta_events = np.zeros(len(merged), dtype=int)

        for j, (clat, clon) in enumerate(zip(cell_lats, cell_lons)):
            # Distance in degrees (approximate, fine for 0.5 degree radius)
            dist = np.sqrt((ev_lats - clat)**2 + (ev_lons - clon)**2)
            nearby = dist <= RADIUS_DEG

            n_events[j]     = nearby.sum()
            total_deaths[j] = ev_deaths[nearby].sum()
            pro_junta_events[j]  = ((ev_sides[nearby] == 'Pro-Junta')).sum()
            anti_junta_events[j] = ((ev_sides[nearby] == 'Anti-Junta')).sum()

        merged['n_events']          = n_events
        merged['total_deaths']      = total_deaths
        merged['pro_junta_events']  = pro_junta_events
        merged['anti_junta_events'] = anti_junta_events

    all_rows.append(merged)

# ── Combine and save ──────────────────────────────────────────────────────
print("\nCombining all transitions...")
df = pd.concat(all_rows, ignore_index=True)

# Reorder columns cleanly
df = df[[
    'lat', 'lon', 'date_t', 'date_t1',
    'faction_t', 'faction_t1', 'level1_side_t', 'level2_group_t',
    'changed',
    'n_events', 'total_deaths', 'pro_junta_events', 'anti_junta_events'
]]

df.to_csv(OUT_PATH, index=False)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"  Total rows       : {len(df):,}")
print(f"  Transitions      : {n_transitions}")
print(f"  Cells per snap   : {len(df) // n_transitions:,}")
print(f"  Saved -> {OUT_PATH}")

print(f"\n  Target variable (changed):")
vc = df['changed'].value_counts()
print(f"    No change (0)  : {vc.get(0,0):,}  ({vc.get(0,0)/len(df)*100:.1f}%)")
print(f"    Changed   (1)  : {vc.get(1,0):,}  ({vc.get(1,0)/len(df)*100:.1f}%)")

print(f"\n  UCDP feature summary:")
print(f"    Rows with any events     : {(df['n_events']>0).sum():,}  ({(df['n_events']>0).mean()*100:.1f}%)")
print(f"    Mean events per row      : {df['n_events'].mean():.3f}")
print(f"    Max events in one cell   : {df['n_events'].max()}")
print(f"    Total deaths recorded    : {df['total_deaths'].sum():,.0f}")
