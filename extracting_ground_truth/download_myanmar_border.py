#!/usr/bin/env python3
"""
One-time setup: downloads the proper Myanmar border shapefile
from Natural Earth and saves it as myanmar_border.geojson

Run this once before using extract_myanmar_svg.py

Requirements:
    pip install requests
"""

import requests
import json
import os

# Natural Earth 50m country boundaries — filtered to Myanmar only
# We download the full GeoJSON and extract Myanmar
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"

OUT_PATH = os.path.join(os.path.dirname(__file__), "myanmar_border.geojson")

print("Downloading Natural Earth country boundaries...")
response = requests.get(URL, timeout=60)
response.raise_for_status()

data = response.json()

# Find Myanmar
myanmar = None
for feature in data['features']:
    props = feature.get('properties', {})
    name = props.get('NAME', '') or props.get('name', '') or props.get('ADMIN', '')
    if name in ('Myanmar', 'Burma'):
        myanmar = feature
        break

if myanmar is None:
    print("ERROR: Could not find Myanmar in the dataset.")
    print("Names found:", [f['properties'].get('NAME') for f in data['features']][:20])
    exit(1)

# Save just Myanmar
out = {
    "type": "FeatureCollection",
    "features": [myanmar]
}

with open(OUT_PATH, 'w') as f:
    json.dump(out, f)

print(f"Saved Myanmar border → {OUT_PATH}")
print("You can now run extract_myanmar_svg.py as normal.")
