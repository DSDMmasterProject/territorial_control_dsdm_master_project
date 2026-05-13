#!/usr/bin/env python3
"""
Visualize Myanmar conflict grid CSV as a map.

Usage:
    python visualize_grid.py myanmar_2024-01-09_grid.csv

Requirements:
    pip install matplotlib pandas
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

# Faction colors matching the Wikipedia SVG
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
    'Outside Myanmar':  '#d0e8f0',  # light blue for sea/neighbors
}

def get_color(faction):
    return FACTION_COLORS.get(faction, '#aaaaaa')  # grey for unknown factions

def plot_grid(csv_path):
    df = pd.read_csv(csv_path)
    date = df['date'].iloc[0]

    # Map each faction to a color
    df['color'] = df['faction'].apply(get_color)

    fig, ax = plt.subplots(figsize=(7, 11))

    # Plot each cell as a colored square
    resolution = 0.1
    for _, row in df.iterrows():
        rect = mpatches.Rectangle(
            (row['lon'] - resolution/2, row['lat'] - resolution/2),
            resolution, resolution,
            linewidth=0,
            facecolor=row['color']
        )
        ax.add_patch(rect)

    # Axis limits
    ax.set_xlim(92.0, 101.5)
    ax.set_ylim(9.5, 28.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Myanmar Conflict Map — {date}', fontsize=13)
    ax.set_aspect('equal')

    # Legend (only factions that appear in this snapshot)
    present_factions = df[df['faction'] != 'Outside Myanmar']['faction'].unique()
    legend_patches = [
        mpatches.Patch(color=get_color(f), label=f)
        for f in sorted(present_factions)
    ]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=7,
              framealpha=0.9, title='Faction')

    plt.tight_layout()
    out_path = csv_path.replace('_grid.csv', '_map.png')
    plt.savefig(out_path, dpi=150)
    print(f"Map saved → {out_path}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_grid.py path/to/myanmar_YYYY-MM-DD_grid.csv")
        sys.exit(1)
    plot_grid(sys.argv[1])
