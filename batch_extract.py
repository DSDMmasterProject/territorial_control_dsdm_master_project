#!/usr/bin/env python3
"""
Batch processor for Myanmar conflict SVG snapshots.
Processes all SVGs into PRIO-GRID aligned CSVs and combines into panel dataset.

Place in: extracting_ground_truth/

Usage:
    python -B batch_extract.py
"""

import os, sys, csv, glob, time, subprocess

FACTION_IDS = {
    'Junta':0,'junta exclaves':0,'KNA':0,'ZRA':1,'PDF':2,
    'KIA':3,'Kachin':3,'Karen':4,'Karenni':5,'AA':6,'Rakhine':6,
    'TNLA':7,'MNDAA':8,'Chin Resistance':9,'Chin':9,'Chin Brotherhood':9,
    'Mon':10,'Danu':11,'SSPP':12,'RCSS':13,'NDAA':14,'UWSA':15,'MNLA':16,
}
NON_FACTIONS = {'roads','Mandalay','Myanmar','Yangon','Kayah n','Aya','circles'}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SVG_DIR     = os.path.join(BASE_DIR, 'wikipedia_snapshots')
GEOJSON_DIR = os.path.join(BASE_DIR, 'output_geojson')
GRID_DIR    = os.path.join(BASE_DIR, 'output_grids')
PANEL_PATH  = os.path.join(BASE_DIR, 'panel_dataset.csv')
EXTRACT     = os.path.join(BASE_DIR, 'extract_myanmar_svg.py')

os.makedirs(GEOJSON_DIR, exist_ok=True)
os.makedirs(GRID_DIR,    exist_ok=True)

svg_files = sorted(glob.glob(os.path.join(SVG_DIR, 'myanmar_*.svg')))
if not svg_files:
    print(f"ERROR: No SVG files found in {SVG_DIR}")
    sys.exit(1)

print(f"Found {len(svg_files)} SVG files")
print(f"Date range: {os.path.basename(svg_files[0])} -> {os.path.basename(svg_files[-1])}")
print(f"Grid: PRIO-GRID (0.5 degree, official cell IDs)")
print("-" * 60)

failed = []
for i, svg_path in enumerate(svg_files):
    filename = os.path.splitext(os.path.basename(svg_path))[0]
    print(f"[{i+1:2d}/{len(svg_files)}] {filename}", end='  ', flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, '-B', EXTRACT, svg_path],
            capture_output=True, text=True, encoding='utf-8',
            env={**os.environ, 'PYTHONUTF8': '1'}
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip().split('\n')[-1])

        src_geojson = os.path.join(SVG_DIR, filename+'.geojson')
        src_grid    = os.path.join(SVG_DIR, filename+'_grid.csv')
        if os.path.exists(src_geojson):
            os.replace(src_geojson, os.path.join(GEOJSON_DIR, filename+'.geojson'))
        if os.path.exists(src_grid):
            os.replace(src_grid, os.path.join(GRID_DIR, filename+'_grid.csv'))

        print(f"OK  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append((filename, str(e)))

print("-" * 60)
print("Combining into panel dataset...")

grid_files = sorted(glob.glob(os.path.join(GRID_DIR, '*_grid.csv')))
all_rows = []; unknown_factions = set()

for grid_path in grid_files:
    date_str = os.path.basename(grid_path).replace('myanmar_','').replace('_grid.csv','')
    with open(grid_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            faction = row['faction']
            if faction == 'Outside Myanmar' or faction in NON_FACTIONS:
                continue
            faction_id = FACTION_IDS.get(faction)
            if faction_id is None:
                unknown_factions.add(faction)
                faction_id = 99
            all_rows.append((
                row['priogrid_gid'], row['lat'], row['lon'],
                faction, faction_id, date_str
            ))

with open(PANEL_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['priogrid_gid','lat','lon','faction','faction_id','date'])
    w.writerows(all_rows)

snapshots = len(grid_files)
print(f"\nDone.")
print(f"  Snapshots processed : {snapshots - len(failed)}/{len(svg_files)}")
print(f"  Total rows in panel : {len(all_rows):,}")
print(f"  Cells per snapshot  : ~{len(all_rows)//snapshots if snapshots else 0:,}")
print(f"  Panel dataset       : {PANEL_PATH}")

if unknown_factions:
    print(f"\n  WARNING: Unknown factions (faction_id=99): {unknown_factions}")
if failed:
    print(f"\n  Failed snapshots ({len(failed)}):")
    for name, err in failed:
        print(f"    {name}: {err}")
