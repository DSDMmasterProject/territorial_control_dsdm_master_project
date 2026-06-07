# Data Collection Pipeline — Wikipedia Scraping Summary
**Project:** Myanmar Territorial Control (FCV / World Bank)
**As of:** 2026-05-26
**Pipeline:** `src/data_collection/` steps 1–9
**This document covers:** the Wikipedia ground-truth arm (steps 1–6) and how it feeds the label vector (steps 8–9).

---

## Important framing before reading

The pipeline does **not** scrape Wikipedia article text, war infoboxes, or belligerent lists. It targets a single Wikipedia resource that is fundamentally different: **`Module:Myanmar_Civil_War_detailed_map`**, a Lua data module that stores the underlying coordinate and icon data powering the interactive territorial control map embedded in Wikipedia's Myanmar Civil War article. There are no `<infobox>` templates, no wikitext belligerent fields, no `mwparserfromhell` usage. The scraping mechanism is a direct MediaWiki REST API call that returns the raw Lua source text as JSON, which is then parsed with hand-written regular expressions. This distinction matters for everything that follows.

---

## Executive Summary

Steps 1–6 fetch `Module:Myanmar_Civil_War_detailed_map` from the English Wikipedia API, parse its Lua table syntax with regex to extract location entries (each a town or township tagged with a coloured SVG icon), decode each icon filename to a `(control_status, actor, confidence)` triple using a hard-coded lookup dictionary, and build a temporal panel by replaying the module's full revision history — yielding 24 monthly snapshots spanning November 2023 to March 2026. Each snapshot contains between 303 and 352 labeled locations, all with exact decimal coordinates. Steps 8–9 aggregate these point labels to PRIOGRID 0.5° cells via majority vote and forward-fill to complete the monthly panel. The final label vector covers 185 of the 199 UCDP-active Myanmar cells (93% coverage) across 29 months (Nov 2023–Mar 2026), with 85.2% high-confidence assignments. Actor grouping is not string-based: it is purely a color-code-to-category lookup with no belligerent list parsing, no fuzzy matching, and no alliance expansion — because the source data encodes actor identity entirely through icon choice, not text.

---

## A) EDA of the Scraped Data

### A.1 Source pages

The pipeline targets **exactly one Wikipedia resource**, accessed via two separate API call patterns:

| Resource | Wikipedia URL / API target | What it contains |
|---|---|---|
| Lua data module (current) | `Module:Myanmar_Civil_War_detailed_map` via `action=query&titles=Module:…&prop=revisions&rvprop=content` | 822 lines of Lua; location entries with lat/lon/icon/label |
| Same module (revision history) | Same endpoint with `rvlimit=500` and pagination | All historical revisions (timestamps + content per revision) |
| SVG map file history (reference only) | `File:Myanmar_civil_war.svg` on Wikimedia Commons via `prop=imageinfo` | File upload timestamps and editor comments; **read but not parsed** |

No Wikipedia article pages, talk pages, infoboxes, or category pages are fetched.

### A.2 What the source data actually looks like

The Lua module is a single `return { marks = { ... } }` table. Each entry looks like:

```lua
{ lat = "17.646", long = "95.460", mark = "Location dot red.svg",
  marksize = "18", label = "[[Hinthada]]", link = "Hinthada",
  label_size = "100", position = "bottom" }
```

Eight fields appear across entries: `lat`, `long`, `mark`, `marksize`, `label`, `link`, `label_size`, `position`. The pipeline extracts `lat`, `long`, `mark`, and `label` only. `marksize`, `link`, `label_size`, and `position` are read but discarded.

The module also organises entries into a two-level geographic hierarchy using **Lua comments** (e.g., `-- Sagaing Region`, `--Monywa District`). These comments are **not parsed** by the code — the admin1/admin2 hierarchy exists in the source but is not present in any output file.

### A.3 Temporal coverage of the data

| | Value |
|---|---|
| **Earliest labeled location date** | 2023-11-30 (first monthly snapshot captured) |
| **Latest labeled location date** | 2026-03-30 (last monthly snapshot captured) |
| **Total temporal span** | 28 months |
| **Number of monthly snapshots** | 24 (not every calendar month has an edit — see gaps below) |
| **Snapshot selection rule** | Last Wikipedia edit within each calendar month |
| **Earliest Wikipedia module edit captured** | 2023-11-30 19:38 UTC (revid 1187678714) |
| **Latest Wikipedia module edit captured** | 2026-03-30 06:15 UTC (revid 1346151570) |

The module existed before November 2023, but step 5 captures only 24 monthly snapshots. Why the range starts in November 2023 is not explained in the code — it appears to be a practical decision coinciding with Operation 1027 (Oct 27 2023), when territorial control began changing rapidly.

**Gaps — months with no Wikipedia editor update:**

| Gap | Period | Duration |
|---|---|---|
| Gap 1 | Feb 2025 → Apr 2025 | 3 months (Feb & Mar 2025 missing) |
| Gap 2 | Jun 2025 → Jul 2025 | 2 months (Jun 2025 missing) |
| Gap 3 | Nov 2025 → Jan 2026 | 3 months (Nov & Dec 2025 missing) |

These gaps are filled by forward-fill (LOCF) in step 9.

### A.4 Temporal coverage of the scrape (cache status)

| | Value |
|---|---|
| Lua module raw text saved to | `data/raw/validation/myanmar_module_raw.lua` |
| Module revision metadata saved to | `data/raw/validation/myanmar_module_revision_metadata.json` |
| Temporal snapshots CSV saved to | `data/raw/validation/myanmar_wikipedia_temporal_groundtruth.csv` |
| Snapshot date of the saved raw `.lua` | 2026-03-30 (latest revision) |
| **Is the data live or cached?** | **Fully cached on disk.** Steps 1–5 fetch from Wikipedia at run time and write to disk; downstream steps (6–10) read local files only. |
| SVG map last updated (reference only) | 2026-05-07 (not re-fetched, only logged) |
| Lag between latest module snapshot and SVG map | ~5 weeks |

### A.5 Volume

**Current snapshot (2026-03-30), step-2 output:**

| | Count |
|---|---|
| Total location entries in current Lua module | 369 `{}` blocks with `lat=` |
| Entries skipped (non-location marks: airports, mountain passes, etc.) | 23 |
| Valid location records in step-2 CSV | 346 |

**Temporal panel (step-5 output, all 24 snapshots):**

| | Count |
|---|---|
| Total rows (locations × snapshots) | 8,035 |
| Unique (lat, lon) pairs across all snapshots | 352 |
| Unique location names | 365 (some names were edited across snapshots) |
| Locations that changed control status ≥ once | 127 / 352 (36.1%) |

**Locations per snapshot:**

| Snapshot | Total | Gov | Contested | EAO | Resistance | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| 2023-11-30 | 303 | 193 | 94 | 0 | 3 | 13 |
| 2023-12-31 | 304 | 190 | 97 | 0 | 5 | 12 |
| 2024-01-31 | 304 | 184 | 97 | 1 | 7 | 15 |
| 2024-02-27 | 304 | 189 | 82 | 9 | 11 | 13 |
| 2024-03-29 | 314 | 196 | 78 | 17 | 14 | 9 |
| 2024-04-26 | 322 | 199 | 76 | 19 | 19 | 9 |
| 2024-05-22 | 324 | 195 | 76 | 23 | 22 | 8 |
| 2024-06-27 | 328 | 194 | 78 | 26 | 22 | 8 |
| 2024-07-31 | 340 | 196 | 74 | 32 | 26 | 12 |
| 2024-08-27 | 345 | 226 | 48 | 31 | 28 | 12 |
| 2024-09-15 | 345 | 226 | 48 | 31 | 28 | 12 |
| 2024-10-25 | 348 | 221 | 51 | 34 | 29 | 13 |
| 2024-11-29 | 346 | 219 | 50 | 36 | 30 | 11 |
| 2024-12-28 | 348 | 219 | 46 | 39 | 32 | 12 |
| 2025-01-28 | 346 | 218 | 46 | 40 | 32 | 10 |
| 2025-04-07 | 346 | 218 | 44 | 41 | 32 | 11 |
| 2025-05-10 | 346 | 219 | 44 | 41 | 32 | 10 |
| 2025-07-08 | 346 | 214 | 44 | 42 | 32 | 14 |
| 2025-08-31 | 346 | 215 | 43 | 42 | 32 | 14 |
| 2025-09-25 | 346 | 215 | 43 | 42 | 34 | 12 |
| 2025-10-18 | 346 | 219 | 40 | 42 | 34 | 11 |
| 2026-01-08 | 346 | 218 | 41 | 42 | 34 | 11 |
| 2026-02-21 | 346 | 219 | 41 | 41 | 34 | 11 |
| 2026-03-30 | 346 | 219 | 40 | 41 | 34 | 12 |

![Records per snapshot date](figs/fig1_records_per_snapshot.png)

**Key trend:** Contested locations dropped from 94 → 40; EAOs rose from 0 → 41; resistance rose from 3 → 34. The EAO category was absent from the first two snapshots — editors began using the relevant icons only from late 2023 / early 2024 as EAO territorial gains became unambiguous.

### A.6 Field completeness

| Field | In step-2 output? | Completeness | Notes |
|---|---|---|---|
| `lat` | Yes | **100%** (346/346) | All entries have explicit `lat=` in Lua |
| `lon` | Yes | **100%** (346/346) | All entries have explicit `long=` in Lua |
| `location_name` | Yes | **100%** (346/346) | 365/367 entries in module have `label=` with a `[[wikilink]]`; 2 have plain-text labels; none are blank in the output |
| `mark` (icon filename) | Yes | **100%** | Entries without `mark=` are not extracted at all |
| `control_status` | Yes | **100%** | Decoded from mark via MARK_MAPPING; no NaN in output |
| `actor` | Yes | **100%** | Decoded from mark; UNMAPPED marks get `"UNMAPPED: <mark>"` as actor |
| `confidence` | Yes | **100%** | Set to `"high"`, `"medium"`, or `"low"` by the mapping dict |
| `date` | Yes (step 3+) | 100% | Stamped with the module revision date in step 3 |
| `admin1` / `admin2` | **No** | 0% | Present as Lua comments in source; **not extracted** |
| `link` (Wikipedia article) | No | — | Present in source; read but dropped |
| `marksize` | No | — | Present in source; read but dropped |

### A.7 Geographic coverage

All 346 locations have exact decimal coordinates extracted directly from the Lua module.

| | Value |
|---|---|
| Coordinate coverage | 100% — every location has lat/lon |
| Lat range | 9.98°N – 27.50°N |
| Lon range | 92.37°E – 100.37°E |
| Admin level | Town/township (individual named settlements) |
| Admin1/Admin2 | **Not extracted.** Embedded in Lua source comments; code does not parse them. |
| State/region available? | Only implicitly — can be derived by spatial join with GADM (not done in the pipeline) |

---

## B) Actor Grouping Logic

### The key fact: there is no belligerent string parsing

Raw actor names like "Tatmadaw", "People's Defence Force", or "Three Brotherhood Alliance" do **not appear anywhere in the scraped data**. The Lua module encodes actor identity entirely through icon choice (which SVG filename to use), not through text fields. Consequently, the code's actor grouping is not string matching, not fuzzy matching, not Wikidata lookup, and not infobox column analysis. It is a single Python dictionary that maps icon filename → `(control_status, actor_label, confidence)`.

### B.1 The MARK_MAPPING dictionary (complete)

Defined identically in both `step2_parse_lua_module.py` and `step5_temporal_snapshots.py` (copy-pasted for self-containment). There is no external file — the mapping is hard-coded.

**Solid dot icons — unambiguous single-actor control:**

| Icon filename | control_status | actor label | confidence |
|---|---|---|---|
| `Location dot red.svg` | government | Military Forces of Myanmar (SAC) | high |
| `Location dot blue.svg` | ethnic_armed_group | Kachin Independence Army (KIO/KIA) | high |
| `Location dot green.svg` | resistance | Arakan Army / NUG-aligned | high |
| `Location dot orange.svg` | ethnic_armed_group | Chin ethnic armed groups (CDF/CNF) | high |
| `Location dot magenta.svg` | resistance | Chin PDF / resistance forces | high |
| `Location dot cyan.svg` | ethnic_armed_group | Arakan Army (AA) | high |
| `Location dot teal.svg` | ethnic_armed_group | Arakan Army (AA) | high |
| `Location dot limegreen.svg` | resistance | NUG/PDF resistance forces | high |
| `Location dot purple.svg` | ethnic_armed_group | United Wa State Army (UWSA) | medium |
| `Location dot yellow.svg` | contested | Unknown actors | low |
| `Location dot black.svg` | contested | Unknown/symbolic | low |
| `Location dot white.svg` | contested | Unknown/symbolic | low |
| `Location dot coral.svg` | ethnic_armed_group | EAO allied with SAC (unconfirmed) | low |
| `Location dot ochre200.png` | ethnic_armed_group | Ethnic armed group (unspecified) | low |
| `Location dot darkslategray.svg` | unknown | Actor unidentified | low |
| `Location dot lightgrey.svg` | unknown | Actor unidentified | low |
| `Location dot red-black.svg` | contested | SAC vs unknown armed group | medium |

**Alternate dot variants (same meaning, different SVG):**

| Icon filename | control_status | actor label | confidence |
|---|---|---|---|
| `Dot green 0d0.svg` | resistance | NUG/PDF resistance forces | high |
| `Dot yellow ff4.svg` | contested | Unclear/contested situation | low |
| `Map-dot-grey-68a.svg` | unknown | Actor unidentified | low |

**Animated GIFs — contested between two specific named actors:**

| Icon filename | control_status | actor label | confidence |
|---|---|---|---|
| `80x80-red-blue-anim.gif` | contested | SAC vs KIO/KIA | high |
| `80x80-red-lime-anim.gif` | contested | SAC vs NUG/PDF | high |
| `80x80-red-magenta-anim.gif` | contested | SAC vs Chin PDF | high |
| `80x80-red-cyan-anim.gif` | contested | SAC vs Arakan Army (AA) | high |
| `80x80-red-black-anim.gif` | contested | SAC vs unknown armed group | low |
| `80x80-red-orange-anim.gif` | contested | SAC vs ethnic armed group | high |
| `80x80-red-green-anim.gif` | contested | SAC vs Arakan Army | high |
| `80x80-blue-lime-anim.gif` | contested | KIO/KIA vs NUG/PDF | high |

**Icons intentionally skipped (not control-status markers):**
`Myanmar Roadmap.png`, `Fighter-jet-blue-icon.svg`, `Fighter-jet-green-icon.svg`,
`Map-circle-*.svg`, `Map-arcNE-green.svg`, `WhiteDot.svg`,
`Anchor pictogram green.svg`, `Mountain pass 12x12 *.svg` (6 variants)

### B.2 Counts per control_status and actor (current snapshot, 2026-03-30)

| control_status | actor label | Icon(s) | Count |
|---|---|---|---:|
| government | Military Forces of Myanmar (SAC) | red dot | 219 |
| contested | SAC vs NUG/PDF | 80x80-red-lime-anim | 22 |
| ethnic_armed_group | KIO/KIA | blue dot | 18 |
| resistance | Arakan Army / NUG-aligned | green dot | 16 |
| resistance | NUG/PDF resistance forces | limegreen dot + Dot green 0d0 | 13 |
| unknown | Actor unidentified | grey dots + Map-dot-grey | 12 |
| ethnic_armed_group | Ethnic armed group (unspecified) | ochre dot | 8 |
| contested | Unclear/contested situation | Dot yellow ff4 | 8 |
| ethnic_armed_group | Chin ethnic armed groups | orange dot | 7 |
| ethnic_armed_group | Arakan Army (AA) | cyan + teal dot | 6 |
| resistance | Chin PDF / resistance forces | magenta dot | 5 |
| contested | SAC vs Arakan Army (AA) | 80x80-red-cyan-anim | 3 |
| unknown | Unknown/symbolic | black dot | 2 |
| contested | SAC vs unknown armed group | red-black dot + 80x80-red-black | 2 |
| contested | SAC vs KIO/KIA | 80x80-red-blue-anim | 2 |
| ethnic_armed_group | UWSA | purple dot | 1 |
| contested | SAC vs Chin PDF | 80x80-red-magenta-anim | 1 |
| ethnic_armed_group | EAO allied with SAC (unconfirmed) | coral dot | 1 |

![Mark → control status mapping](figs/fig2_mark_to_actor_mapping.png)

### B.3 Alliance handling

**Not implemented.** "Three Brotherhood Alliance" (MNDAA + PSLF/TNLA + KIO/KIA) does not appear as a text field anywhere. Wikipedia editors represent the three member groups individually on the map using the same blue dot icon — all three map to `ethnic_armed_group / KIO/KIA` in the current code (see the blue dot entry above), which is **incorrect** for MNDAA and PSLF/TNLA locations. This is a known limitation: the blue dot is code-defined as "KIO/KIA" but is used by Wikipedia editors to represent any northern EAO.

### B.4 Ungrouped / unmatched actors

In the current snapshot there are **zero UNMAPPED entries** — every mark type present in the 2026-03-30 module is in either MARK_MAPPING or SKIP_MARKS.

However, across all 24 historical snapshots, Wikipedia editors used **9 additional mark types** that the MARK_MAPPING dictionary does not cover. These produced `"UNMAPPED: <mark>"` actor labels in the temporal dataset:

| Unrecognised mark | Occurrences (all snapshots) | Probable meaning |
|---|---:|---|
| `80x80-red-yellow-anim.gif` | 18 | SAC vs RCSS/SSA (yellow = Shan State Army?) |
| `80x80-red-grey-anim.gif` | 16 | SAC vs unknown (grey = ambiguous) |
| `Fighter-jet-red-icon.svg` | 14 | SAC airbase (should be in SKIP_MARKS) |
| `Location dot green-yellow.svg` | 7 | Hybrid NUG/EAO? |
| `Location dot green-blue.svg` | 6 | Hybrid resistance/EAO? |
| `80x80-red-purple-anim.gif` | 4 | SAC vs UWSA? |
| `Location dot lime.svg` | 4 | NUG/PDF variant? |
| `Location dot deeppink.svg` | 2 | Unknown |
| 4 directional arc icons | 4 | Advance direction arrows (should be in SKIP_MARKS) |

Total UNMAPPED records in temporal dataset: **75** across all 24 snapshots (0.93% of 8,035 rows). These carry `confidence="low"` and are filtered out in step 9's `VALID_STATUSES` check, which only passes `government`, `resistance`, `ethnic_armed_group`, and `contested`.

---

## C) Cell-to-Actor Attribution

### C.1 Spatial unit

**PRIOGRID 0.5° × 0.5° cells (~55 km sides).** This is UCDP's native spatial unit — both the feature matrix (step 8) and the label vector (step 9) use it. Cell IDs (`priogrid_gid`) are integers computed via a verified closed-form formula:

```python
row = int((lat + 90) / 0.5) + 1
col = int((lon + 180) / 0.5) + 1
gid = (row - 1) * 720 + col
```

This formula was validated against two known UCDP cell IDs before use (cell 158235, cell 161111).

### C.2 Where cell assignments come from

Each Wikipedia-labeled town/township is a **point**, not a polygon. There are no "areas controlled by X" shapes, no polygon overlays, and no spatial join with administrative boundaries. A cell gets its actor assignment from **all Wikipedia-labeled towns that happen to fall inside that 0.5° square**.

### C.3 The attribution rule (step-by-step)

**Step 9 (`step9_build_labels.py`) implements this in full:**

1. Load the 8,035-row temporal ground truth (352 unique locations × 24 snapshots).
2. For each location, compute its PRIOGRID cell ID using the formula above.
3. **Filter out non-territorial labels**: drop rows where `control_status` is `"unknown"` or `"communal"`. Keep only `{"government", "resistance", "ethnic_armed_group", "contested"}`. This removes 275 unknown-labeled observations across the temporal panel.
4. **Group by `(priogrid_gid, year_month)`**. For each group, apply three aggregation functions:
   - `majority_vote(series)`: return `value_counts().index[0]` — the most frequent label. No explicit tie-break rule; ties resolve by pandas insertion order.
   - `vote_confidence(series)`: `"high"` if only one unique label or ≥75% agree; `"medium"` if ≥50%; `"low"` if <50%.
   - `vote_count(series)`: number of towns that contributed to this cell-snapshot.
5. **Build the complete monthly panel** for November 2023 – March 2026 (all labeled cells × all months in the UCDP feature matrix from Nov 2023 onward).
6. **Forward-fill (LOCF)** within each cell: for months where no Wikipedia editor updated the module, carry the last known label forward. Drop any rows that still have NaN after filling (months before a cell's first Wikipedia snapshot).
7. **Add `is_direct_snapshot` flag**: True if this row corresponds to a month where Wikipedia was actually edited; False if LOCF-filled.

### C.4 Time dimension

**Monthly, per snapshot.** Each cell gets one label per calendar month. Attribution is dynamic — a cell can change status when Wikipedia editors update the module.

### C.5 Towns per cell distribution

| Towns in cell | Cell-month count |
|---:|---:|
| 1 | 2,260 |
| 2 | 1,202 |
| 3 | 495 |
| 4 | 244 |
| 5 | 24 |
| 6 | 64 |
| 8 | 24 |

Most cells (52%) contain only a single labeled town, so for those cells the "majority vote" is trivially the single label with no possibility of disagreement.

### C.6 Tie-breaking and contested cells

There is **no explicit conflict-aware tie-breaking rule.** `majority_vote` returns the first-ranked label by count; if two labels tie, pandas `value_counts` returns them in insertion order. Only 37 cell-months have `label_confidence = "low"` (a tie or near-tie); of those, 19 have `government` as the assigned label, 17 have `resistance`, and 1 has `contested`. None of the 37 were inspected manually.

Cells whose label status changed across months (dynamic, not contested within one snapshot): 73 of 185 labeled cells (39.5%) changed their assigned status at least once.

### C.7 Coverage

| | Count |
|---|---|
| Total UCDP-active cells (at least one post-coup event) | 199 |
| Cells with ≥1 Wikipedia label in any snapshot | **185 (93.0%)** |
| Unlabeled conflict cells | 14 (7.0%) |
| Total labeled cell-months in label panel | 5,238 |
| Direct snapshot rows | 4,313 (82.3%) |
| LOCF forward-filled rows | 925 (17.7%) |
| Label confidence: high | 4,465 (85.2%) |
| Label confidence: medium | 736 (14.1%) |
| Label confidence: low | 37 (0.7%) |

**Control status distribution across all labeled cell-months:**

| Control status | Cell-months | % |
|---|---:|---:|
| government | 3,323 | 63.4% |
| contested | 891 | 17.0% |
| ethnic_armed_group | 613 | 11.7% |
| resistance | 411 | 7.8% |

![Wikipedia points projected to PRIOGRID cells, March 2026](figs/fig3_wiki_points_to_cells.png)

---

## Caveats and Known Issues

### What the code handles well

- **Full revision history capture**: Step 5 paginates through all Wikipedia revisions, not just the current one. The 24-month temporal panel captures all territorial control changes since Operation 1027.
- **Complete coordinate coverage**: All 369 Lua entries have explicit `lat=`/`long=` fields. Coordinate completeness is 100% in all output files.
- **Known non-location marks explicitly skipped**: The `SKIP_MARKS` set prevents airports, mountain passes, and decorative overlays from polluting the dataset.
- **LOCF clearly flagged**: The `is_direct_snapshot` column distinguishes real observations from imputed ones, so users can filter to direct evidence only.
- **Rate limiting**: Step 5 sleeps 1.5 seconds between revision content fetches to avoid Wikipedia rate limits.

### What the code does not handle

1. **Admin1/Admin2 hierarchy not extracted.** The Lua module organizes entries by state and district in its comment lines (102 such comments identified), but the regex parser ignores all comment text. There is no `admin1` or `admin2` column in any output file. Downstream geographic analysis requires a separate GADM spatial join.

2. **Blue dot ambiguity.** `Location dot blue.svg` is defined as "Kachin Independence Army (KIO/KIA)" but is used by Wikipedia editors for all three northern EAOs (KIO/KIA, MNDAA, PSLF/TNLA). Locations belonging to MNDAA and PSLF/TNLA in the Shan/Sagaing border regions are systematically mislabeled as KIO/KIA.

3. **9 UNMAPPED historical marks.** Icons introduced and later retired by Wikipedia editors were never added to MARK_MAPPING. 75 historical observations carry `"UNMAPPED: <mark>"` actor labels and are dropped from the label vector (valid statuses filter). The most likely meanings of the dropped marks are SAC vs RCSS (18 instances) and SAC vs ambiguous (16 instances).

4. **No infobox, belligerent, or article text parsing.** The pipeline cannot answer "which groups are allied with which" or "what territory does X claim" from article text. The only actor information is the icon color — a single categorical variable chosen by whichever Wikipedia volunteer last edited that town's entry.

5. **Wikipedia editor lag and subjectivity.** The module reflects volunteer editors' judgments, sourced from news reports. The SVG map was updated on 2026-05-07 with "April changes" that are not reflected in the data (latest captured revision: 2026-03-30). The editor methodology is "update when news reports confirm a change," meaning transient or contested territory may be systematically underrepresented.

6. **3 edit gaps in the temporal panel.** Months with no Wikipedia edit (Feb–Mar 2025, Jun 2025, Nov–Dec 2025) are filled by LOCF. The label panel assigns the same status for 2–3 consecutive months during these gaps regardless of what may have actually happened on the ground.

7. **No majority vote tie-breaking.** In the 37 low-confidence cell-months, the winner is determined by pandas insertion order — effectively arbitrary. A principled tie-breaking rule (e.g., prefer "contested" over any single-actor claim) is not implemented.

8. **`ethnic_armed_group` absent from first two snapshots.** The Nov 2023 and Dec 2023 snapshots show zero EAO-labeled locations. This appears to reflect the state of the map at that time — editors had not yet resolved newly contested areas to clear EAO control — rather than a genuine absence of EAO activity. Any model trained on these early snapshots will underestimate EAO presence in 2023.

9. **`marksize` encoding not used.** The Lua module encodes town population in `marksize` (larger dot = larger city). This population signal is read but discarded. It could serve as a proxy for strategic importance.

10. **`link=` field discarded.** Every entry has a `link=` field pointing to the corresponding Wikipedia article (e.g., `"Hinthada"`). This could enable automated enrichment (fetching population, coordinates from Wikidata, admin hierarchy) but is not used.
