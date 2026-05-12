# ============================================================
# STEP 8: Aggregate UCDP events → PRIOGRID monthly feature matrix
# Input:  data/processed/myanmar_ucdp_events.csv
# Output: data/processed/myanmar_priogrid_features.csv
# This is the X matrix that feeds directly into the model
# ============================================================

# Right now we have a list of 5,737 individual conflict events (one row per violent incident).
# The model doesn't work at the event level — it works at the cell-month level.
# Step 8 asks: "For each of the 192 PRIOGRID cells, for each of the 50 months since the coup, what happened?"
# The answer becomes a row in the feature matrix — the X that feeds the model.

# The feature engineering step aggregates 5,737 georeferenced conflict events into a 192-cell × 50-month panel
# using UCDP's native PRIOGRID cell assignments at the 0.5° resolution recommended by Mihai Croicu
# based on conflict data's effective spatial resolution of ~50km. Each cell-month observation contains 30+ features
# including actor-specific event counts, fatality measures, one-sided violence indicators,
# and temporal lag features at 1, 3, and 6 month horizons —
# because territorial control is path-dependent,
# and who controlled an area last month is the strongest predictor of this month

import pathlib  # pathlib: file path handling
from itertools import product  # product: create all combinations of cells × months

import numpy as np  # numpy: numerical operations and NaN handling
import pandas as pd  # pandas: data manipulation and groupby aggregation

# ── 1. PATHS ──────────────────────────────────────────────────

EVENTS_PATH = pathlib.Path("data/processed/myanmar_ucdp_events.csv")  # Step 7 output
OUT_DIR = pathlib.Path("data/processed")  # Output folder
OUT_PATH = OUT_DIR / "myanmar_priogrid_features.csv"  # Feature matrix output

# ── 2. LOAD EVENTS ────────────────────────────────────────────

print("=" * 60)
print("STEP 8: Building PRIOGRID monthly feature matrix")
print("=" * 60)

df = pd.read_csv(EVENTS_PATH, encoding="utf-8")  # Load our Step 7 output
df["date_start"] = pd.to_datetime(df["date_start"])  # Parse date column to datetime

print(f"\n✅ Loaded {len(df):,} post-coup Myanmar events")

# ── 3. FILTER BY SPATIAL PRECISION ───────────────────────────
#
# where_prec codes how precisely the event was located:
#   1 = exact GPS coordinates
#   2 = small area  (<25 km²)
#   3 = large area  (25–100 km²)
#   4 = very large area (>100 km²)
#   5 = nearest named town
#   6 = first-order admin unit (ADM1 = state/region level)
#   7 = country level only (event assigned to capital)
#
# At 0.5° / ~55km grid resolution, events coded to level 6 or 7
# are assigned to the ADM1 centroid or capital and may fall in
# the wrong cell. We keep ≤5 for reliable grid assignment.

before = len(df)  # Count before filtering
df = df[df["where_prec"] <= 5].copy()  # Keep only events with reliable coordinates
removed = before - len(df)  # How many we dropped
print("\n[A] Spatial precision filter (where_prec ≤ 5):")
print(f"   Removed {removed:,} low-precision events ({100 * removed / before:.1f}%)")
print(f"   Kept    {len(df):,} events for feature engineering")

# ── 4. ADD ACTOR PRESENCE FLAGS ───────────────────────────────
#
# For each event, mark which major actors are involved.
# This creates binary (0/1) columns that when summed give
# "how many events in this cell-month involved actor X"

# Actor codes to track — these match the level3 strings in our taxonomy
TRACK_ACTORS = {  # Dict: column_name → level3 code
    "events_gov": "military_sac",  # SAC/Tatmadaw government forces
    "events_nug": "nug_pdf",  # NUG / People's Defence Force
    "events_kio": "kio_kia",  # Kachin Independence Organisation/Army
    "events_ula": "ula_aa",  # United League of Arakan / Arakan Army
    "events_knu": "knu_knla",  # Karen National Union
    "events_mndaa": "mndaa",  # Myanmar National Democratic Alliance Army
    "events_pslf": "pslf_tnla",  # Palaung State Liberation Front / TNLA
    "events_cnf": "cnf",  # Chin National Front
    "events_rcss": "rcss_ssas",  # Shan State Army South
    "events_uwsa": "uwsa",  # United Wa State Army
}

for (
    col_name,
    level3_code,
) in TRACK_ACTORS.items():  # Loop through each actor we want to track
    # An event "involves" this actor if they appear on EITHER side
    df[col_name] = (  # Create binary column: 1 if actor present, 0 if not
        (df["side_a_level3"] == level3_code)  # Actor appears as side_a (initiator)
        | (df["side_b_level3"] == level3_code)  # OR actor appears as side_b (target)
    ).astype(int)  # Convert boolean to 0/1 integer

# ── 5. ADD CIVILIAN TARGETING FLAG ───────────────────────────
#
# One-sided violence (type 3) = deliberate targeting of civilians
# This is a key signal for territorial control AND humanitarian risk

df["is_onesided"] = (df["type_of_violence"] == 3).astype(
    int
)  # 1 if one-sided violence, 0 otherwise
df["is_statebased"] = (df["type_of_violence"] == 1).astype(
    int
)  # 1 if state-based conflict, 0 otherwise
df["gov_vs_civilians"] = (  # 1 if SAC deliberately targeted civilians
    (df["side_a_level3"] == "military_sac")  # SAC was the perpetrator
    & (df["type_of_violence"] == 3)  # AND it was one-sided violence
).astype(int)

# ── 6. AGGREGATE TO CELL-MONTH ────────────────────────────────
#
# groupby is the core operation: collect all events in the same
# (priogrid_gid, year_month) bucket and compute summary statistics

print("\n[B] Aggregating events to PRIOGRID cell × month...")

# Build the aggregation dictionary: column → aggregation function
# This tells pandas WHAT to compute for each column in each bucket
AGG_DICT = {
    # ── Event counts ─────────────────────────────────
    "id": "count",  # count: total number of events (id is always present)
    "is_statebased": "sum",  # sum: number of state-based conflict events
    "is_onesided": "sum",  # sum: number of one-sided violence events
    "gov_vs_civilians": "sum",  # sum: SAC attacks on civilians specifically
    # ── Fatality measures ────────────────────────────
    "best": "sum",  # sum: total deaths (best estimate)
    "deaths_a": "sum",  # sum: combatant deaths on side A
    "deaths_b": "sum",  # sum: combatant deaths on side B
    "deaths_civilians": "sum",  # sum: civilian deaths
    # ── Actor presence counts ────────────────────────
    **{
        col: "sum" for col in TRACK_ACTORS.keys()
    },  # sum each actor flag: count of events per actor
    # ── Conflict diversity ───────────────────────────
    "dyad_name": "nunique",  # nunique: number of distinct conflict dyads active this month
    "adm_1": "first",  # first: state/region name (all events in cell should share this)
}

# Run the aggregation
cell_month = (
    df.groupby(
        ["priogrid_gid", "year_month"]
    )  # groupby: split into buckets by cell AND month
    .agg(AGG_DICT)  # agg: apply the aggregation functions
    .reset_index()  # reset_index: turn the group keys back into columns
)

# Rename the raw "id" count to something meaningful
cell_month = cell_month.rename(
    columns={  # rename: give columns descriptive names
        "id": "total_events",  # Total events in this cell-month
        "best": "total_fatalities",  # Total fatalities
        "dyad_name": "n_unique_dyads",  # Number of distinct conflict dyads
    }
)

print(f"   Raw cell-months with events: {len(cell_month):,}")

# ── 7. BUILD COMPLETE PANEL (fill zero months) ────────────────
#
# Problem: our aggregation only has rows WHERE events occurred.
# For the model, we also need rows where NOTHING happened —
# zero conflict is still information (the cell was quiet).
#
# We create a complete grid: every cell × every month combination,
# then merge with our aggregated data, filling missing with zeros.

print("\n[C] Building complete panel (all cells × all months)...")

# Get all cells that had at least one post-coup event
active_cells = df["priogrid_gid"].unique()  # All cells with any event
# Get all months from Feb 2021 to Mar 2026
all_months = sorted(df["year_month"].unique())  # All months in the data

# Create every possible (cell, month) pair
all_combos = pd.DataFrame(  # Build a DataFrame of all combinations
    list(
        product(active_cells, all_months)
    ),  # product: cartesian product of cells × months
    columns=["priogrid_gid", "year_month"],  # Name the two columns
)
print(
    f"   Complete panel size: {len(all_combos):,} rows ({len(active_cells)} cells × {len(all_months)} months)"
)

# Merge: left join keeps ALL cell-month combinations, even those with no events
panel = all_combos.merge(cell_month, on=["priogrid_gid", "year_month"], how="left")

# Fill numeric NaN with 0 (= no events that month, not missing data)
numeric_cols = panel.select_dtypes(
    include=[np.number]
).columns  # Get all numeric columns
panel[numeric_cols] = panel[numeric_cols].fillna(0)  # Fill NaN with 0

# Fill adm_1 (state name) using forward fill within each cell group
panel = panel.sort_values(
    ["priogrid_gid", "year_month"]
)  # Sort chronologically within each cell
panel["adm_1"] = panel.groupby("priogrid_gid")[
    "adm_1"
].transform(  # transform: apply within each cell group
    lambda x: x.ffill().bfill()  # ffill=forward fill, bfill=backward fill gaps
)

print(f"   Zero-event cell-months added: {len(panel) - len(cell_month):,}")

# ── 8. ADD TEMPORAL LAG FEATURES ─────────────────────────────
#
# Lag features = "what happened in previous months in this cell"
# These are crucial because territorial control is persistent —
# who controlled an area last month is a strong predictor of this month.
#
# lag1 = 1 month ago, lag3 = 3 months ago, lag6 = 6 months ago

print("\n[D] Adding lag features...")

panel = panel.sort_values(["priogrid_gid", "year_month"]).reset_index(
    drop=True
)  # Sort for correct lag computation

for lag in [1, 3, 6]:  # Create lags at 1, 3, and 6 months
    for col in [
        "total_events",
        "total_fatalities",
        "events_gov",
        "events_nug",  # Lag these key columns
        "gov_vs_civilians",
        "is_onesided",
    ]:
        lag_col_name = f"{col}_lag{lag}"  # Name: e.g. "total_events_lag1"
        panel[lag_col_name] = (
            panel.groupby("priogrid_gid")[
                col
            ]  # Group by cell so lags don't cross cell boundaries
            .shift(lag)  # shift: move values forward by `lag` months
            .fillna(0)  # Fill first `lag` months with 0 (no prior data)
        )

# ── 9. ADD ROLLING WINDOW FEATURES ───────────────────────────
#
# Rolling features = "what was the average over the past N months"
# These smooth out single-month spikes and capture sustained conflict intensity

for window in [3, 6]:  # 3-month and 6-month rolling windows
    for col in ["total_events", "total_fatalities"]:
        roll_col_name = f"{col}_roll{window}m"  # Name: e.g. "total_events_roll3m"
        panel[roll_col_name] = panel.groupby("priogrid_gid")[
            col
        ].transform(  # Group by cell  # transform: apply function within each group
            lambda x: x.rolling(
                window, min_periods=1
            ).mean()  # rolling: sliding window mean; min_periods=1 handles early months
        )

# ── 10. ADD DOMINANCE FEATURES ────────────────────────────────
#
# Dominance = what fraction of events in this cell-month were
# initiated by the government? A high fraction = SAC is actively
# operating here; a low fraction = non-government actors dominating

panel["gov_event_share"] = np.where(  # np.where: conditional expression (like Excel IF)
    panel["total_events"] > 0,  # If there were any events
    panel["events_gov"] / panel["total_events"],  # Compute government share of events
    0.0,  # Otherwise 0 (no events → no share)
)

panel["nug_event_share"] = np.where(
    panel["total_events"] > 0, panel["events_nug"] / panel["total_events"], 0.0
)

# ── 11. CONVERT YEAR_MONTH TO DATETIME FOR EASY FILTERING ─────

panel["year_month_dt"] = pd.to_datetime(  # Convert "2021-02" string to proper datetime
    panel["year_month"] + "-01"  # Append "-01" to make it a valid date (first of month)
)

# ── 12. PRINT SUMMARY ─────────────────────────────────────────

print("\n── Feature Matrix Summary ──")
print(f"   Rows (cell-months):   {len(panel):,}")
print(f"   Columns (features):   {len(panel.columns)}")
print(f"   PRIOGRID cells:       {panel['priogrid_gid'].nunique()}")
print(f"   Months covered:       {panel['year_month'].nunique()}")
print(
    f"   Date range:           {panel['year_month'].min()} → {panel['year_month'].max()}"
)

print("\n── Feature columns created ──")
feature_cols = [
    c
    for c in panel.columns
    if c  # Feature columns = everything except ID and metadata
    not in ["priogrid_gid", "year_month", "year_month_dt", "adm_1"]
]
for i, col in enumerate(feature_cols):  # Print all feature names
    print(f"   {i + 1:>2}. {col}")

print("\n── Conflict intensity distribution ──")
print(
    f"   Cell-months with zero events:  {(panel['total_events'] == 0).sum():,} ({100 * (panel['total_events'] == 0).mean():.1f}%)"
)
print(
    f"   Cell-months with 1–5 events:   {((panel['total_events'] > 0) & (panel['total_events'] <= 5)).sum():,}"
)
print(
    f"   Cell-months with 6–20 events:  {((panel['total_events'] > 5) & (panel['total_events'] <= 20)).sum():,}"
)
print(f"   Cell-months with >20 events:   {(panel['total_events'] > 20).sum():,}")

print("\n── Top 10 most active cells ──")
top_cells = (
    panel.groupby("priogrid_gid")["total_events"]  # Group by cell
    .sum()  # Sum all events across all months
    .sort_values(ascending=False)  # Sort highest first
    .head(10)
)  # Keep top 10
for cell_id, total in top_cells.items():  # Print each
    # Get the state name for this cell
    state = panel[panel["priogrid_gid"] == cell_id]["adm_1"].iloc[0]
    print(f"   Cell {cell_id:>8}  {total:>4} events  ({state})")

# ── 13. SAVE ──────────────────────────────────────────────────

panel.to_csv(OUT_PATH, index=False, encoding="utf-8")  # Save feature matrix

print(f"\n✅ Feature matrix saved: {OUT_PATH}")
print(f"   Size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")

print("\n" + "=" * 60)
print("STEP 8 COMPLETE.")
print("Next: Step 9 = join ground truth labels → build train/validation set")
print("=" * 60)
