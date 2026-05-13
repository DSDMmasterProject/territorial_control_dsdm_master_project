#!/usr/bin/env python3
"""
Myanmar Conflict Map Extractor
Converts Wikipedia SVG conflict maps into georeferenced GeoJSON and
a raster grid CSV.

Usage:
    python extract_myanmar_svg.py path/to/myanmar_YYYY-MM-DD.svg

Requirements:
    pip install shapely scipy numpy
"""

import xml.etree.ElementTree as ET
import numpy as np
import json, re, argparse, os, csv
from collections import Counter
from scipy.interpolate import LinearNDInterpolator
from numpy.linalg import lstsq
from shapely.geometry import Polygon, Point, mapping, shape
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# Ground control points: SVG pixel (x,y) -> (lat, lon)
GCPS = [
    (581.1, 1235.5, 16.866, 96.195),  # Yangon
    (577.1,  724.1, 21.975, 96.084),  # Mandalay
    (578.6,  948.9, 19.745, 96.129),  # Naypyidaw
    (724.0, 1269.7, 16.490, 97.628),  # Mawlamyine
    (699.3,  378.0, 25.383, 97.400),  # Myitkyina
    (736.7,  623.0, 22.934, 97.752),  # Lashio
    (268.0,  895.5, 20.152, 92.900),  # Sittwe
    (466.4,  901.4, 20.149, 94.932),  # Magway
    (447.6, 1238.6, 16.774, 94.732),  # Pathein
    (668.3,  840.0, 20.789, 97.037),  # Taunggyi
    (748.8,  518.5, 23.988, 97.657),  # Muse
    (776.8, 1508.0, 14.083, 98.196),  # Dawei
    (551.1,  827.4, 20.879, 95.859),  # Meiktila
    (485.0,  697.8, 22.108, 95.135),  # Monywa
    (685.3,  928.4, 19.674, 97.210),  # Loikaw
    (607.2, 1178.8, 17.335, 96.480),  # Bago
    (512.6, 1032.0, 18.818, 95.218),  # Pyay
    (394.3,  599.0, 23.212, 94.014),  # Kalay
]

def _load_myanmar_border():
    border_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myanmar_border.geojson')
    if os.path.exists(border_path):
        with open(border_path) as f:
            gj = json.load(f)
        geom = shape(gj['features'][0]['geometry'])
        print("  Using precise Myanmar border from myanmar_border.geojson")
        return geom
    else:
        print("  WARNING: myanmar_border.geojson not found, using rough approximation.")
        return Polygon([
            [92.30, 28.00], [92.60, 28.30], [93.20, 28.60], [94.00, 28.50],
            [95.00, 28.20], [96.00, 28.40], [96.60, 28.50], [97.00, 28.30],
            [97.40, 28.50], [97.80, 28.20], [98.40, 28.00], [98.70, 27.50],
            [99.00, 26.80], [99.40, 26.20], [99.80, 25.00], [100.20, 23.80],
            [100.80, 23.00],[101.20, 22.50],[101.60, 22.20],[101.50, 21.50],
            [100.80, 21.00],[100.30, 20.50],[100.10, 20.10],[ 99.50, 19.00],
            [ 99.20, 18.50],[ 99.00, 17.50],[ 98.60, 16.50],[ 98.80, 15.50],
            [ 98.60, 15.00],[ 98.30, 14.00],[ 99.00, 13.00],[ 99.20, 11.50],
            [ 99.00, 10.50],[ 98.50, 10.00],[ 98.00, 10.50],[ 97.50, 11.00],
            [ 98.00, 13.00],[ 97.80, 15.00],[ 97.50, 16.00],[ 97.00, 17.00],
            [ 96.50, 17.50],[ 96.00, 18.00],[ 95.50, 19.00],[ 94.50, 19.50],
            [ 94.00, 20.00],[ 93.50, 20.50],[ 93.00, 21.00],[ 92.80, 21.50],
            [ 92.30, 22.00],[ 92.10, 23.00],[ 92.00, 24.00],[ 92.20, 25.00],
            [ 92.00, 26.00],[ 92.10, 27.00],[ 92.30, 28.00],
        ])

MYANMAR_BORDER = _load_myanmar_border()


def build_transform():
    px = np.array([[g[0], g[1]] for g in GCPS])
    lats = np.array([g[2] for g in GCPS])
    lons = np.array([g[3] for g in GCPS])
    lat_interp = LinearNDInterpolator(px, lats)
    lon_interp = LinearNDInterpolator(px, lons)
    X = np.column_stack([px[:, 0], px[:, 1], np.ones(len(px))])
    c_lat, _, _, _ = lstsq(X, lats, rcond=None)
    c_lon, _, _, _ = lstsq(X, lons, rcond=None)
    return lat_interp, lon_interp, c_lat, c_lon


def px_to_geo(points, lat_interp, lon_interp, c_lat, c_lon):
    pts = np.array(points)
    lats = lat_interp(pts)
    lons = lon_interp(pts)
    nm = np.isnan(lats)
    if nm.any():
        X2 = np.column_stack([pts[nm, 0], pts[nm, 1], np.ones(nm.sum())])
        lats[nm] = X2 @ c_lat
        lons[nm] = X2 @ c_lon
    return list(zip(lons.tolist(), lats.tolist()))


def parse_svg_path(d):
    rings = []
    current_ring = []
    tokens = re.findall(
        r'[MLCZSQTAHVmlczsqtahv]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d
    )
    i = 0; cmd = None; cx, cy = 0.0, 0.0

    while i < len(tokens):
        t = tokens[i]
        if t in 'MLCZSQTAHVmlczsqtahv':
            cmd = t; i += 1; continue
        if cmd in ('M', 'm'):
            if current_ring and len(current_ring) > 2:
                rings.append(current_ring)
            x, y = float(tokens[i]), float(tokens[i+1])
            if cmd == 'm': x += cx; y += cy
            cx, cy = x, y; current_ring = [(cx, cy)]; i += 2
            cmd = 'L' if cmd == 'M' else 'l'
        elif cmd in ('L', 'l'):
            x, y = float(tokens[i]), float(tokens[i+1])
            if cmd == 'l': x += cx; y += cy
            cx, cy = x, y; current_ring.append((cx, cy)); i += 2
        elif cmd in ('H', 'h'):
            x = float(tokens[i])
            if cmd == 'h': x += cx
            cx = x; current_ring.append((cx, cy)); i += 1
        elif cmd in ('V', 'v'):
            y = float(tokens[i])
            if cmd == 'v': y += cy
            cy = y; current_ring.append((cx, cy)); i += 1
        elif cmd in ('C', 'c'):
            x1,y1 = float(tokens[i]),float(tokens[i+1])
            x2,y2 = float(tokens[i+2]),float(tokens[i+3])
            x, y  = float(tokens[i+4]),float(tokens[i+5])
            if cmd == 'c': x1+=cx;y1+=cy;x2+=cx;y2+=cy;x+=cx;y+=cy
            for tv in np.linspace(0,1,8)[1:]:
                bx=(1-tv)**3*cx+3*(1-tv)**2*tv*x1+3*(1-tv)*tv**2*x2+tv**3*x
                by=(1-tv)**3*cy+3*(1-tv)**2*tv*y1+3*(1-tv)*tv**2*y2+tv**3*y
                current_ring.append((bx, by))
            cx, cy = x, y; i += 6
        elif cmd in ('S', 's'):
            x2,y2 = float(tokens[i]),float(tokens[i+1])
            x, y  = float(tokens[i+2]),float(tokens[i+3])
            if cmd == 's': x2+=cx;y2+=cy;x+=cx;y+=cy
            for tv in np.linspace(0,1,6)[1:]:
                bx=(1-tv)**2*cx+2*(1-tv)*tv*x2+tv**2*x
                by=(1-tv)**2*cy+2*(1-tv)*tv*y2+tv**2*y
                current_ring.append((bx, by))
            cx, cy = x, y; i += 4
        elif cmd in ('Z', 'z'):
            if current_ring and len(current_ring) > 2:
                rings.append(current_ring)
            current_ring = []; cmd = None
        else:
            i += 1

    if current_ring and len(current_ring) > 2:
        rings.append(current_ring)
    return rings


def extract_factions(svg_path):
    lat_interp, lon_interp, c_lat, c_lon = build_transform()
    label_attr = '{http://www.inkscape.org/namespaces/inkscape}label'
    tree = ET.parse(svg_path)
    root = tree.getroot()

    faction_polys = {}
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        label = path.get(label_attr, '').strip()
        if not label or 'junta' in label.lower():
            continue
        d = path.get('d', '')
        if not d:
            continue
        style = path.get('style', '')
        fm = re.search(r'fill:#([0-9a-fA-F]{6})', style)
        color = ('#' + fm.group(1)) if fm else path.get('fill', 'none')
        # Skip paths with no fill — these are roads, borders, outlines, not territory
        if not color or color == 'none':
            continue
        for ring in parse_svg_path(d):
            if len(ring) < 3:
                continue
            geo = px_to_geo(ring, lat_interp, lon_interp, c_lat, c_lon)
            try:
                poly = Polygon(geo)
                if poly.is_valid and poly.area > 0:
                    faction_polys.setdefault((label, color), []).append(poly)
            except Exception:
                pass

    features = []
    merged_others = []

    for (label, color), polys in faction_polys.items():
        try:
            merged = unary_union(polys)
            if merged.is_empty:
                continue
            canonical = re.sub(r'\s+exclave[s]?$', '', label, flags=re.IGNORECASE).strip()
            merged_others.append(merged)
            features.append({
                "type": "Feature",
                "properties": {
                    "faction":   canonical,
                    "label_raw": label,
                    "color":     color,
                    "area_deg2": round(merged.area, 4),
                    "is_junta":  False,
                },
                "geometry": mapping(merged)
            })
        except Exception as e:
            print(f"  Warning merging '{label}': {e}")

    # Reconstruct Junta = Myanmar border minus all other factions
    try:
        others_union = unary_union(merged_others) if merged_others else Polygon()
        junta_geom = MYANMAR_BORDER.difference(others_union)
        if not junta_geom.is_empty:
            features.append({
                "type": "Feature",
                "properties": {
                    "faction":   "Junta",
                    "label_raw": "Junta (reconstructed)",
                    "color":     "#ebc0b3",
                    "area_deg2": round(junta_geom.area, 4),
                    "is_junta":  True,
                },
                "geometry": mapping(junta_geom)
            })
    except Exception as e:
        print(f"  Warning reconstructing Junta: {e}")

    return features


def rasterize(features, date_str, resolution=0.1):
    lat_min, lat_max = 9.5, 28.5
    lon_min, lon_max = 92.0, 101.5
    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)

    geoms_primary, junta_geom = [], None
    for feat in features:
        try:
            g = shape(feat['geometry'])
            name = feat['properties']['faction']
            if name == 'Junta':
                junta_geom = g
            else:
                geoms_primary.append((name, g))
        except Exception:
            pass

    rows = []
    n = len(lats)
    for i, lat in enumerate(lats):
        if i % 20 == 0:
            print(f"  Rasterizing... {i}/{n} rows", end='\r')
        for lon in lons:
            pt = Point(lon, lat)
            faction = 'Outside Myanmar'
            for name, geom in geoms_primary:
                if geom.contains(pt):
                    faction = name
                    break
            if faction == 'Outside Myanmar' and junta_geom and junta_geom.contains(pt):
                faction = 'Junta'
            rows.append((round(lat, 4), round(lon, 4), faction, date_str))
    print(f"  Rasterizing... done{' '*20}")
    return rows


def main():
    parser = argparse.ArgumentParser(description='Extract Myanmar SVG -> GeoJSON + grid CSV')
    parser.add_argument('svg', help='Input SVG file path')
    parser.add_argument('--resolution', type=float, default=0.1)
    args = parser.parse_args()

    svg_path = args.svg
    filename = os.path.splitext(os.path.basename(svg_path))[0]
    out_dir  = os.path.dirname(os.path.abspath(svg_path))
    if not os.access(out_dir, os.W_OK):
        out_dir = os.getcwd()
    geojson_path = os.path.join(out_dir, filename + '.geojson')
    csv_path     = os.path.join(out_dir, filename + '_grid.csv')
    date_str     = filename

    print(f"\nProcessing: {svg_path}")
    features = extract_factions(svg_path)

    with open(geojson_path, 'w') as f:
        json.dump({"type":"FeatureCollection","features":features}, f, indent=2)

    print(f"\n  GeoJSON saved -> {geojson_path}")
    print(f"  {'Faction':<26} {'Area (deg2)':>12}")
    print(f"  {'-'*40}")
    for feat in sorted(features, key=lambda f: -f['properties']['area_deg2']):
        p = feat['properties']
        tag = ' <- reconstructed' if p['is_junta'] else ''
        print(f"  {p['faction']:<26} {p['area_deg2']:>12}{tag}")

    rows = rasterize(features, date_str, args.resolution)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lat', 'lon', 'faction', 'date'])
        w.writerows(rows)

    counts = Counter(r[2] for r in rows)
    n_mmr = sum(v for k,v in counts.items() if k != 'Outside Myanmar')
    print(f"\n  Grid CSV saved -> {csv_path}")
    print(f"  {len(rows):,} total cells | {n_mmr:,} inside Myanmar\n")
    print(f"  {'Faction':<26} {'Cells':>6}  {'%Myanmar':>9}")
    print(f"  {'-'*44}")
    for faction, n in counts.most_common():
        if faction == 'Outside Myanmar': continue
        print(f"  {faction:<26} {n:>6}  {n/n_mmr*100:>8.1f}%")


if __name__ == '__main__':
    main()