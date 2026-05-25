# Myanmar Conflict Prediction — Project Status

## Project Goal
Build a model that uses conflict event data (UCDP/ACLED) to predict or explain
territorial control in Myanmar, as captured by Wikipedia conflict maps over time.
The project is a World Bank FCV Challenge Project in partnership with the World Bank's
Fragility, Conflict & Violence team.

---

## What We Have Built

### 1. Wikipedia SVG Extraction Pipeline
A complete reproducible pipeline that converts Wikipedia conflict map SVG files
into structured geographic data.

**How it works:**
- Parses SVG faction polygons using Inkscape label attributes
- Georeferences pixel coordinates to lat/lon using 18 city anchor points (~4-6km accuracy)
- Reconstructs Junta territory as Myanmar border minus all other faction polygons
  (Junta is the SVG background, not an explicit polygon)
- Assigns each PRIO-GRID cell a faction label using majority vote across 9 sample points
- Output: one GeoJSON + one grid CSV per snapshot

**Key design decisions:**
- Paths with `fill: none` are skipped (roads, borders, outlines — not territory)
- Faction aliases are normalized (Kachin→KIA, Rakhine→AA etc.)
- PRIO-GRID (0.5° standard grid) used for spatial alignment with UCDP

**Scripts:**
- `extracting_ground_truth/extract_myanmar_svg.py` — single SVG extraction
- `extracting_ground_truth/batch_extract.py` — batch process all SVGs
- `extracting_ground_truth/download_myanmar_border.py` — one-time border download
- `extracting_ground_truth/visualize_grid.py` — visualize a grid CSV as a map
- `visualize_with_events.py` — overlay UCDP events on grid map

---

### 2. Wikipedia Panel Dataset
**File:** `extracting_ground_truth/panel_dataset_labelled.csv`

- 62 snapshots from 2023-12-30 to 2026-05-07
- 286 PRIO-GRID cells inside Myanmar
- 17,748 rows (snapshots × cells, not perfectly regular due to majority vote)
- Columns: `priogrid_gid, lat, lon, faction, faction_id, level1_side, level2_group, wiki_label, date`

**Faction hierarchy:**
- Level 1 (`level1_side`): Pro-Junta / Anti-Junta / Other
- Level 2 (`level2_group`): Junta / Ethnic armed / Pro-democracy / Ceasefire / Other
- Level 3 (`wiki_label`): direct faction name (KIA, AA, PDF etc.)

**Distribution (level 1):**
- Pro-Junta: 59% (Junta heartland)
- Anti-Junta: 39%
- Other: 2% (UWSA etc.)

---

### 3. UCDP Dataset
**File:** `data/raw/ucdp/ucdp_labelled.csv`

- Source: UCDP GED 25.1 + Candidate 2025/2026
- Filtered to 2023+: 4,041 events
- Added columns: `wiki_label_a/b, level2_group_a/b, level1_side_a/b`
- Already contains `priogrid_gid` for exact spatial join
- Side A is Pro-Junta in 99% of events (government always coded as initiator)

**Actor crosswalk (UCDP → Wikipedia):**
| UCDP | Wiki label |
|------|------------|
| Government of Myanmar | Junta |
| NUG | PDF |
| ULA | AA |
| KIO | KIA |
| KNU | Karen |
| KNPP | Karenni |
| PSLF | TNLA |
| MNDAA | MNDAA |
| CNF | Chin Resistance |
| PNLO | ZRA |
| ABSDF | PDF |

---

### 4. Feature Matrix
**File:** `model/feature_matrix.csv`

- 17,567 rows (286 cells × 30 months, forward-filled, Other cells dropped)
- 43 columns total, 33 features
- Temporal split: train (Oct 2025 and before) / validate (Nov 2025 onwards)

**Features:**
- Current faction encoded (`is_pro_junta`, `months_in_current_faction`)
- Core UCDP: `n_events`, `total_deaths`, `n_unique_groups`
- Actor flags: `flag_kia`, `flag_pdf`, `flag_aa` etc. (9 factions)
- Actor ratios: `junta_event_share`, `anti_junta_event_share`
- Lag features: 1/3/6 months for events and deaths
- Rolling means: 3/6 months for events and deaths
- Trend: `event_trend_3m`, `months_since_last_event`
- Neighbor features: `neighbor_n_events`, `neighbor_changed_faction` (PRIO-GRID adjacency)
- Faction match: `attacker_matches_controller`

---

### 5. Model Results
**Target:** did this cell change from Pro-Junta to Anti-Junta (or vice versa) next month?

| Model | Accuracy | ROC-AUC | Recall(change) |
|-------|----------|---------|----------------|
| Baseline (no change) | 0.9948 | 0.5000 | 0.000 |
| Logistic Regression | 0.9499 | 0.6354 | 0.222 |
| Random Forest | 0.9732 | 0.3458 | 0.000 |
| Gradient Boosting | 0.9948 | 0.5889 | 0.000 |

**Key findings:**
- Signal exists — Logistic Regression ROC-AUC of 0.635 is above random
- Most predictive features: `flag_kia`, `flag_tnla`, `flag_mndaa`, `junta_event_share`
- Core problem: only 174 change events in 17,567 rows (1% change rate)
- In validate set: only 9 change events — too few for reliable evaluation

---

## Core Problem

**Label scarcity.** We have 30 months of Wikipedia snapshots (Dec 2023 - May 2026).
Territorial control is highly persistent — most cells never change. This gives us
very few positive examples for the change prediction task.

**Attempted solutions:**
- Switched from 0.1° custom grid to PRIO-GRID (0.5°) for exact UCDP join — improved
  UCDP coverage from 0% to 13-17% of changing cells
- Tried predicting faction label instead of change — dominated by persistence
- Considered extending labels back to 2021 — Wikipedia SVGs don't go back that far
- Considered ACLED town capture events as labels — too inconsistently coded

---

## Comparison with Colleague's Approach

| | Our approach | Colleague's approach |
|--|-------------|---------------------|
| Label source | Wikipedia SVG polygons | Wikipedia Lua module (towns) |
| Spatial unit | PRIO-GRID cells (full territory) | Towns → PRIO-GRID (populated places) |
| Label type | Pro-Junta / Anti-Junta (binary) | 4-class control status |
| Time range | Dec 2023 - May 2026 (30 months) | Nov 2023 - Mar 2026 (29 months) |
| Training rows | ~15,851 | ~3,335 |
| Change events | 174 | more (4-class gives more variation) |
| UCDP features | Events + actor flags + lags | Events + actor flags + lags + ratios |

**Key difference:** polygon-based labels capture rural/wilderness territorial control
that town-based labels miss. Town-based labels capture who controls strategic populated
places. Neither is strictly better — they measure different aspects of control.

---

## Open Questions

1. **Does ACLED have internal territorial control data** (polygons/shapefiles) not
   in the public dataset? — Ask at tomorrow's meeting
2. **How reliable are ACLED sub-events** ("Government regains territory", "Non-state
   actor overtakes territory") as territorial control labels?
3. **Is the boundary hypothesis testable** with our data — do conflict events
   concentrate near faction boundaries? This doesn't require change events.
4. **Can we use spatial analysis** rather than prediction as the main contribution —
   explaining where conflict happens relative to territorial boundaries?

---

## Next Steps (to decide)

**Option A — Improve prediction:**
- Get better labels from ACLED sub-events (pending meeting)
- Extend to Somalia/Nigeria (World Bank FCV countries) for more data
- Try longer lag structures (12-18 months)

**Option B — Spatial analysis:**
- Measure distance from conflict events to nearest faction boundary
- Test boundary hypothesis: does violence concentrate at borders?
- Does pre-boundary conflict predict boundary movement?
- Compare conflict signatures across factions

**Option C — Methodological contribution:**
- Systematic comparison of polygon vs town-based labels
- Where do they agree/disagree and what does that tell us?
- Pipeline generalisation to other conflicts

---

## File Structure

```
territorial_control_dsdm_master_project/
├── add_faction_labels.py
├── build_feature_matrix.py
├── train_models.py
├── visualize_with_events.py
├── extracting_ground_truth/
│   ├── extract_myanmar_svg.py
│   ├── batch_extract.py
│   ├── download_myanmar_border.py
│   ├── visualize_grid.py
│   ├── myanmar_border.geojson
│   ├── panel_dataset.csv
│   ├── panel_dataset_labelled.csv
│   ├── wikipedia_snapshots/       (62 SVG files)
│   ├── output_geojson/            (62 GeoJSON files)
│   └── output_grids/              (62 grid CSV files)
├── data/raw/ucdp/
│   ├── GEDEvent_myanmar_merged.csv
│   └── ucdp_labelled.csv
└── model/
    ├── feature_matrix.csv
    └── results/
        ├── model_comparison.csv
        ├── feature_importance.csv
        └── predictions.csv
```
