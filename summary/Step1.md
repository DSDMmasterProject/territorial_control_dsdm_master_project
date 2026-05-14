# Myanmar Conflict Prediction — Data Pipeline Summary

## Project Goal
Build a model (starting with a Hidden Markov Model) that uses conflict event data
from UCDP to predict territorial control in Myanmar, as captured by Wikipedia
conflict maps over time.

---

## Data Sources

### 1. Wikipedia SVG Conflict Maps
- 62 snapshots from 2023-12-30 to 2026-05-07
- Downloaded from Wikimedia Commons (File: Myanmar_civil_war_map.svg, file history)
- Each SVG shows which armed faction controls which geographic zone in Myanmar
- Stored in: `extracting_ground_truth/wikipedia_snapshots/`

### 2. UCDP GED Events (GEDEvent_myanmar_merged.csv)
- UCDP Georeferenced Event Dataset for Myanmar
- Full range: 1989–2026, but we use 2023+ (4,041 events)
- Each row is one conflict event with location, date, actors, and deaths
- Stored in: `data/raw/ucdp/`

---

## Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `download_myanmar_border.py` | `extracting_ground_truth/` | One-time download of precise Myanmar border from Natural Earth |
| `extract_myanmar_svg.py` | `extracting_ground_truth/` | Converts one SVG into a GeoJSON + grid CSV |
| `batch_extract.py` | `extracting_ground_truth/` | Runs extraction on all 62 SVGs, produces panel dataset |
| `visualize_grid.py` | `extracting_ground_truth/` | Visualizes a grid CSV as a map (for sanity checking) |
| `add_faction_labels.py` | project root | Adds faction hierarchy columns to both datasets |

---

## How the SVG Extraction Works

1. **Read SVG as XML** — each faction territory is a `<path>` element with an
   Inkscape label (e.g. `label="KIA"`) and a fill color. Paths with `fill: none`
   are roads/borders/outlines and are skipped.

2. **Georeference pixel -> lat/lon** — SVG paths use pixel coordinates. We convert
   them to real geography using 18 ground control points: city label positions in
   the SVG matched to their known real-world coordinates (e.g. Yangon is at pixel
   (581, 1235) = lat/lon 16.87, 96.20). A linear interpolation transform is fitted
   to these 18 anchor points. Accuracy: ~4-6 km mean error.

3. **Reconstruct Junta territory** — the Junta (SAC/Tatmadaw) controls the central
   heartland but it is never drawn as an explicit polygon in the SVG — it is just
   the grey background. We reconstruct it as:
   `Junta territory = Myanmar national border minus all other faction zones`
   The Myanmar border comes from the Natural Earth dataset (downloaded once by
   `download_myanmar_border.py`).

4. **Rasterize to grid** — Myanmar is divided into a 0.1° x 0.1° grid (~11km per
   cell). Each cell is labelled with the faction whose polygon contains that cell's
   center point. Cells outside Myanmar's border are labelled "Outside Myanmar" and
   dropped from the output.

---

## Output Files

### Per SVG snapshot (62 files each):
- `extracting_ground_truth/output_geojson/myanmar_YYYY-MM-DD.geojson`
  Geographic polygons per faction in real lat/lon coordinates.
- `extracting_ground_truth/output_grids/myanmar_YYYY-MM-DD_grid.csv`
  Raster grid for that snapshot: `lat, lon, faction, date`

### Combined:
- `extracting_ground_truth/panel_dataset.csv`
  All 62 snapshots stacked: `lat, lon, faction, faction_id, date`
  364,319 rows | ~5,876 Myanmar cells per snapshot

- `extracting_ground_truth/panel_dataset_labelled.csv`
  Same as above with three added columns:
  `wiki_label, level2_group, level1_side`

- `data/raw/ucdp/ucdp_labelled.csv`
  UCDP events filtered to 2023+, with added columns:
  `wiki_label_a/b, level2_group_a/b, level1_side_a/b`

---

## Faction Hierarchy

Both datasets use the same three-level vocabulary:

### Level 1 — side (coarsest)
- `Pro-Junta` — SAC/Tatmadaw and allied militias
- `Anti-Junta` — all opposition groups
- `Other` — UWSA (neutral), ARSA (Rohingya), etc.

### Level 2 — group
- `Junta` — SAC/Tatmadaw
- `Pro-democracy` — PDF, NUG-aligned groups
- `Ethnic armed` — KIA, AA, Karen, Karenni, TNLA, MNDAA, Chin, ZRA, Mon
- `Ceasefire` — SSPP, RCSS, NDAA (non-combatant but mapped)
- `Other` — UWSA, ARSA, etc.

### Level 3 — wiki_label (finest, 1-to-1 match)

| Wikipedia label | UCDP actor | Notes |
|----------------|------------|-------|
| Junta | Government of Myanmar (Burma) | Reconstructed as residual territory |
| PDF | NUG, ABSDF | People's Defence Force |
| KIA | KIO | Kachin Independence Army |
| AA | ULA | Arakan Army |
| Karen | KNU | Karen National Liberation Army |
| Karenni | KNPP | Karenni resistance |
| TNLA | PSLF | Ta'ang National Liberation Army |
| MNDAA | MNDAA | Myanmar National Democratic Alliance Army |
| Chin Resistance | CNF | Chin National Army and allies |
| ZRA | PNLO | Zomi Revolutionary Army / Pa-O |
| RCSS | RCSS | Shan State Army South (non-combatant) |
| SSPP | SSPP | Shan State Army North (non-combatant) |
| NDAA | — | No UCDP events recorded |
| UWSA | — | Wa State Army, no UCDP events |
| Mon | — | Too small/recent for UCDP coverage |
| Danu | — | Very small local group |

---

## Known Limitations

- **Georeferencing accuracy**: ~4-6 km mean error. Sufficient for 0.1° grid but
  not for street-level analysis.
- **Junta territory is approximate**: reconstructed as a residual, so any
  inaccuracy in the Myanmar border or other faction polygons affects Junta cells.
- **Grid resolution**: 0.1° (~11 km). Can be changed by passing `--resolution`
  to `extract_myanmar_svg.py` and re-running `batch_extract.py`.
- **Unequal time spacing**: snapshots are not evenly spaced in time. This needs
  to be handled in the HMM (e.g. time-weighted transitions or treating each
  snapshot as one discrete step).
- **Wikipedia maps are manually edited**: the SVG files are maintained by
  Wikipedia volunteers and may lag real events by days or weeks.
- **UCDP and Wikipedia use different spatial units**: UCDP gives point locations
  per event; Wikipedia gives polygon zones. The join between them (next step)
  uses a spatial snap of each UCDP event to its nearest grid cell.

---

## Next Steps

1. **Spatial join**: snap each UCDP event (lat/lon point) to the nearest grid
   cell so events and panel rows share the same spatial unit.
2. **Exploratory analysis**: which cells change faction most often? What does the
   transition matrix look like?
3. **HMM**: model each cell's faction sequence as a hidden Markov chain, using
   UCDP event features (deaths, actor type, event count) as observations.