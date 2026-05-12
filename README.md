# territorial_control_dsdm_master_project
Using conflict event data, develop and evaluate methods to estimate and map territorial control over time and space. Illustrate output through examples on specific countries


src/data_collection/
│
│  ── GROUND TRUTH PIPELINE (Wikipedia → y labels) ──────────────────
│
├── step1_fetch_wikipedia_module.py
│   Fetches the raw Lua source of Wikipedia's Myanmar Civil War module
│   via the Wikipedia API and saves it to data/raw/validation/myanmar_module_raw.lua
│   WHY: The Lua module is the only structured, machine-readable source
│        of expert-coded territorial control labels for Myanmar.
│
├── step2_parse_lua_module.py
│   Parses the Lua source with regex to extract 346 location entries
│   (lat, lon, icon → control status) and saves myanmar_wikipedia_parsed_step2.csv
│   WHY: Converts raw Wikipedia code into a structured table we can use.
│
├── step3_add_timestamps.py
│   Fetches the revision timestamp of the module, stamps all rows with
│   the snapshot date, and saves myanmar_wikipedia_control_labels.csv
│   WHY: Gives our ground truth a date so we can align it with UCDP event months.
│
├── step4_validation_map.py
│   Plots all 346 labeled points on a Myanmar admin boundary map and
│   saves myanmar_wikipedia_ground_truth_map.png to reports/figures/
│   WHY: Geographic sanity check — confirms all points fall inside Myanmar.
│
├── step5_temporal_snapshots.py
│   Fetches 276 historical revisions of the Wikipedia module and extracts
│   24 monthly snapshots → saves myanmar_wikipedia_temporal_groundtruth.csv
│   WHY: Gives us 8,035 labeled observations across time, not just one snapshot.
│
├── step6_temporal_visualisation.py
│   Produces three outputs from the temporal data: an animated GIF,
│   a small-multiples grid, and a trend line chart → saves to reports/figures/
│   WHY: Validates temporal patterns and provides World Bank presentation material.
│
│  ── FEATURE PIPELINE (UCDP events → X features) ───────────────────
│
├── build_actor_taxonomy.py
│   Creates actor_taxonomy_myanmar.csv — a crosswalk table mapping every
│   UCDP actor abbreviation (KIO, ULA, PSLF…) to 3 levels of grouping
│   WHY: UCDP uses abbreviations; we need to know which side each actor is on.
│
├── merge_ucdp_sources.py
│   Merges GED 25.1 (1989-2024) + Candidate 2025 + Candidate Jan-Mar 2026
│   into one clean file: GEDEvent_myanmar_merged.csv (Myanmar only, 9,336 rows)
│   WHY: Creates a single canonical source file all teammates use identically.
│
├── step7_load_ucdp_myanmar.py
│   Loads the merged UCDP file, filters to post-coup period (Feb 2021+),
│   joins the actor taxonomy, saves myanmar_ucdp_events.csv (6,446 rows)
│   WHY: Produces a clean, labelled event-level dataset ready for aggregation.
│
├── step8_priogrid_features.py
│   Aggregates 6,446 events to 199 PRIOGRID cells × 62 months with 43 features
│   (event counts, fatalities, actor presence, lags, rolling averages)
│   → saves myanmar_priogrid_features.csv — this is the X matrix
│   WHY: Models work on cell-month level, not individual events.
│
│  ── LABEL PIPELINE (Wikipedia points → y labels) ──────────────────
│
├── step9_build_labels.py
│   Projects 352 Wikipedia point labels onto PRIOGRID cells using the
│   PRIOGRID formula, applies majority vote per cell, forward-fills gaps,
│   → saves myanmar_priogrid_labels.csv — this is the y vector
│   WHY: Converts point-level ground truth to the same spatial unit as X.
│
└── step10_build_training_dataset.py
    Joins X (features) and y (labels) on (priogrid_gid, year_month),
    adds a temporal train/validate split at Oct 2025,
    → saves myanmar_training_dataset.csv and myanmar_prediction_dataset.csv
    WHY: Creates the final model-ready files teammates load directly.