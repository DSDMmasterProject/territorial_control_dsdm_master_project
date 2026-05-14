#!/usr/bin/env python3
"""
Visualize a Myanmar grid map with UCDP conflict events overlaid.
Shows territorial control at a snapshot date with events from the
30 days prior plotted on top.

Usage:
    python visualize_with_events.py 2024-03-16
    python visualize_with_events.py 2024-03-16 --days 60

Requirements:
    pip install matplotlib pandas
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from datetime import datetime, timedelta

# ── Paths — adjust if your folder structure differs ───────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GRIDS_DIR  = os.path.join(BASE_DIR, 'extracting_ground_truth', 'output_grids')
UCDP_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'ucdp', 'ucdp_labelled.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'extracting_ground_truth', 'output_maps')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Faction colors (grid background) ─────────────────────────────────────
FACTION_COLORS = {
    'Junta':            '#ebc0b3',
    'PDF':              '#cae7c4',
    'KIA':              '#c1c1c1',
    'Karen':            '#88c0f2',
    'Karenni':          '#96efef',
    'MNDAA':            '#ff7f2a',
    'NDAA':             '#d8abe7',
    'TNLA':             '#cd7c7c',
    'UWSA':             '#ed9595',
    'AA':               '#dad17e',
    'RCSS':             '#e2e21d',
    'SSPP':             '#eacccc',
    'Chin Resistance':  '#a9e89b',
    'ZRA':              '#fade69',
    'Mon':              '#f0c0e0',
    'Danu':             '#c0e0c0',
    'Outside Myanmar':  '#d0e8f0',
}

# ── Event dot colors by level1_side ───────────────────────────────────────
EVENT_COLORS = {
    'Pro-Junta':  '#8B0000',   # dark red
    'Anti-Junta': '#00008B',   # dark blue
    'Other':      '#444444',   # dark grey
}


def find_nearest_snapshot(target_date, grids_dir):
    """Find the grid CSV whose date is closest to and <= target_date."""
    import glob
    files = sorted(glob.glob(os.path.join(grids_dir, 'myanmar_*_grid.csv')))
    best = None
    best_date = None
    for f in files:
        date_str = os.path.basename(f).replace('myanmar_', '').replace('_grid.csv', '')
        try:
            d = datetime.strptime(date_str, '%Y-%m-%d')
            if d <= target_date and (best_date is None or d > best_date):
                best = f
                best_date = d
        except ValueError:
            continue
    return best, best_date


def main():
    parser = argparse.ArgumentParser(
        description='Visualize grid map with UCDP events overlaid'
    )
    parser.add_argument('date', help='Snapshot date YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=30,
                        help='How many days before snapshot to include events (default 30)')
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, '%Y-%m-%d')
    window_start = target_date - timedelta(days=args.days)

    # ── Load grid ─────────────────────────────────────────────────────────
    grid_path, snap_date = find_nearest_snapshot(target_date, GRIDS_DIR)
    if grid_path is None:
        print(f"ERROR: No grid found on or before {args.date}")
        sys.exit(1)
    print(f"Grid snapshot : {snap_date.strftime('%Y-%m-%d')} ({grid_path})")

    grid = pd.read_csv(grid_path)

    # ── Load UCDP events ──────────────────────────────────────────────────
    ucdp = pd.read_csv(UCDP_PATH, parse_dates=['date_start'])
    events = ucdp[
        (ucdp['date_start'] >= window_start) &
        (ucdp['date_start'] <= target_date)
    ].copy()
    print(f"UCDP events   : {len(events)} events between "
          f"{window_start.strftime('%Y-%m-%d')} and {target_date.strftime('%Y-%m-%d')}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 11))
    resolution = 0.1

    # Background: grid cells
    for _, row in grid.iterrows():
        color = FACTION_COLORS.get(row['faction'], '#cccccc')
        rect = mpatches.Rectangle(
            (row['lon'] - resolution/2, row['lat'] - resolution/2),
            resolution, resolution,
            linewidth=0, facecolor=color
        )
        ax.add_patch(rect)

    # Overlay: UCDP events
    if len(events) > 0:
        # Scale dot size by deaths (best estimate), min size 20
        max_deaths = events['best'].max()
        if max_deaths > 0:
            events['dot_size'] = 20 + (events['best'] / max_deaths) * 150
        else:
            events['dot_size'] = 30

        for _, ev in events.iterrows():
            side = ev.get('level1_side_a', 'Other')
            color = EVENT_COLORS.get(side, '#444444')
            ax.scatter(
                ev['longitude'], ev['latitude'],
                s=ev['dot_size'],
                c=color,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.4,
                zorder=5
            )

    # ── Axes & labels ─────────────────────────────────────────────────────
    ax.set_xlim(92.0, 101.5)
    ax.set_ylim(9.5, 28.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(
        f'Myanmar Conflict — {snap_date.strftime("%Y-%m-%d")}\n'
        f'Events: {window_start.strftime("%Y-%m-%d")} to {target_date.strftime("%Y-%m-%d")} '
        f'(n={len(events)})',
        fontsize=11
    )
    ax.set_aspect('equal')

    # ── Legend: factions ─────────────────────────────────────────────────
    present_factions = grid[grid['faction'] != 'Outside Myanmar']['faction'].unique()
    faction_patches = [
        mpatches.Patch(color=FACTION_COLORS.get(f, '#cccccc'), label=f)
        for f in sorted(present_factions)
    ]

    # ── Legend: events ────────────────────────────────────────────────────
    event_handles = [
        mlines.Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=EVENT_COLORS['Pro-Junta'],
                      markersize=7, label='Event: Pro-Junta'),
        mlines.Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=EVENT_COLORS['Anti-Junta'],
                      markersize=7, label='Event: Anti-Junta'),
        mlines.Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=EVENT_COLORS['Other'],
                      markersize=7, label='Event: Other/Civilian'),
        mlines.Line2D([0],[0], marker='o', color='w',
                      markerfacecolor='grey', markersize=4,
                      label='Dot size = deaths'),
    ]

    # Two-column legend: factions left, events right
    legend1 = ax.legend(handles=faction_patches, loc='lower left',
                        fontsize=6, framealpha=0.9, title='Territory',
                        title_fontsize=7)
    ax.add_artist(legend1)
    ax.legend(handles=event_handles, loc='lower right',
              fontsize=6, framealpha=0.9, title='Events',
              title_fontsize=7)

    plt.tight_layout()

    out_name = f"myanmar_{args.date}_with_events_{args.days}d.png"
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
