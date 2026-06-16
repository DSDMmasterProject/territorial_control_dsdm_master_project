# ── filter_priogrid_catalog.py ────────────────────────────────────────────────
# data from here: https://grid.prio.org/#/download
# Filters raw PRIOGRID catalog CSV files to Myanmar cells only.
# Input:  data/raw/priogrid_static.csv   (from grid.prio.org Static Variables)
#         data/raw/priogrid_yearly.csv   (from grid.prio.org Yearly Variables)
# Output: data/processed/priogrid_static_myanmar.csv
#         data/processed/priogrid_yearly_myanmar.csv
# Run from project root: uv run python src/data_collection/filter_priogrid_catalog.py

from pathlib import Path  # file path handling

import pandas as pd  # dataframe operations

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)  # project root (two levels up from this file)
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # raw data folder
DATA_PROC = PROJECT_ROOT / "data" / "processed"  # processed data folder

# ── 1. Load the existing UCDP feature matrix to get Myanmar cell IDs ──────────
# We use this to filter the global PRIOGRID files to only Myanmar cells.
features = pd.read_csv(DATA_PROC / "myanmar_priogrid_features.csv")
myanmar_gids = set(
    features["priogrid_gid"].unique()
)  # set of Myanmar priogrid_gid values
print(f"Myanmar cells in UCDP feature matrix: {len(myanmar_gids)}")

# ── 2. Load and filter the static table ───────────────────────────────────────
# The static file uses 'gid' as the identifier (may differ from 'priogrid_gid').
# We try both naming conventions to handle different download formats.
static_raw = pd.read_csv(DATA_RAW / "priogrid_static.csv")

# Detect which column name is used for the cell identifier
id_col_static = "priogrid_gid" if "priogrid_gid" in static_raw.columns else "gid"
print(f"\nStatic file identifier column: '{id_col_static}'")
print(f"Static file columns: {list(static_raw.columns)}")

# Rename to priogrid_gid for consistency
if id_col_static == "gid":
    static_raw = static_raw.rename(columns={"gid": "priogrid_gid"})

# Filter to Myanmar cells
myanmar_static = static_raw[static_raw["priogrid_gid"].isin(myanmar_gids)].copy()
print(f"Myanmar rows in static file: {len(myanmar_static)}")

# ── 3. Select the static features we want ────────────────────────────────────
# These are ALL confirmed to exist in the downloaded file.
# Column name 'mountains_mean' is used in the actual file (not 'mountain_mean').
STATIC_COLS_WANTED = [
    "priogrid_gid",
    "mountains_mean",  # % mountainous terrain — terrain defensibility for EAOs
    "ttime_mean",  # travel time to nearest city (minutes) — state reach proxy
    "forest_gc",  # % forest cover (GlobCover 2009) — EAO operational cover
    "agri_gc",  # % agricultural land — lowland gov territory indicator
    "petroleum_s",  # petroleum deposit present, no known discovery year (binary)
    "gem_s",  # gem deposit present (rubies/jade — critical for KIA finance)
    "diamsec_s",  # secondary diamond deposit, no known discovery year (binary)
    "diamprim_s",  # primary diamond deposit, no known discovery year (binary)
]

# Only keep columns that actually exist in the file — safe against version differences
available = [c for c in STATIC_COLS_WANTED if c in myanmar_static.columns]
missing = [c for c in STATIC_COLS_WANTED if c not in myanmar_static.columns]
if missing:
    print(f"⚠️  Static columns not found: {missing}")
print(f"✅ Static columns selected: {[c for c in available if c != 'priogrid_gid']}")

myanmar_static = myanmar_static[available].reset_index(drop=True)
myanmar_static[available[1:]] = myanmar_static[available[1:]].fillna(0.0)  # missing = 0

print("\nStatic feature statistics:")
print(myanmar_static[available[1:]].describe().round(3))

# ── 4. Load and filter the yearly table ──────────────────────────────────────
# The yearly file may use 'gid' instead of 'priogrid_gid'.
# PRIOGRID v.2.0 covers 1946-2014 — we take the most recent year available.
yearly_raw = pd.read_csv(DATA_RAW / "priogrid_yearly.csv")

# Detect identifier column
id_col_yearly = "priogrid_gid" if "priogrid_gid" in yearly_raw.columns else "gid"
print(f"\nYearly file identifier column: '{id_col_yearly}'")
print(f"Yearly file columns: {list(yearly_raw.columns)}")

if id_col_yearly == "gid":
    yearly_raw = yearly_raw.rename(columns={"gid": "priogrid_gid"})

# Filter to Myanmar cells
myanmar_yearly = yearly_raw[yearly_raw["priogrid_gid"].isin(myanmar_gids)].copy()
print(f"Myanmar rows in yearly file before year filter: {len(myanmar_yearly)}")

# ── 5. Select yearly features — only those confirmed present ─────────────────
# We want the latest available year for each cell (static proxy for 2021-2026).
YEARLY_COLS_WANTED = [
    "priogrid_gid",
    "year",
    "bdist1",  # km to nearest land-contiguous international border
    "capdist",  # km to national capital (Naypyidaw)
    "drug_y",  # binary: drug cultivation (opium poppy) — Mihai's key feature
    "petroleum_y",  # binary: petroleum with known discovery year
    "gem_y",  # binary: gem deposit with known discovery year
    "nlights_mean",  # mean nighttime lights (available 1992-2013 only)
    "pop_gpw_sum",  # population count (GPW, available for 1990/1995/2000/2005)
]

available_yearly = [c for c in YEARLY_COLS_WANTED if c in myanmar_yearly.columns]
missing_yearly = [c for c in YEARLY_COLS_WANTED if c not in myanmar_yearly.columns]

if missing_yearly:
    print(f"\n⚠️  Yearly columns not found: {missing_yearly}")
    print(
        "   → These variables need to be selected when re-downloading the yearly file."
    )
    print("   → For now, proceeding with available yearly columns only.")

if len(available_yearly) <= 2:  # only priogrid_gid + year, no real features
    print("\n⚠️  No useful yearly features found — saving empty yearly file.")
    print("   → Re-download priogrid_yearly.csv with: bdist1, capdist, drug_y selected")
    myanmar_yearly_out = pd.DataFrame({"priogrid_gid": list(myanmar_gids)})
else:
    myanmar_yearly = myanmar_yearly[available_yearly].copy()
    myanmar_yearly["year"] = pd.to_numeric(myanmar_yearly["year"], errors="coerce")
    # Take the latest year per cell — used as static proxy for 2021-2026
    myanmar_yearly_out = (
        myanmar_yearly.sort_values("year")
        .groupby("priogrid_gid")
        .last()  # latest available year per cell
        .reset_index()
    )
    myanmar_yearly_out[available_yearly[2:]] = myanmar_yearly_out[
        available_yearly[2:]
    ].fillna(0.0)
    print("\nYearly features (latest year per cell):")
    print(
        myanmar_yearly_out[
            [c for c in available_yearly if c not in ["priogrid_gid", "year"]]
        ]
        .describe()
        .round(3)
    )

# ── 6. Save ───────────────────────────────────────────────────────────────────
static_out = DATA_PROC / "priogrid_static_myanmar.csv"
yearly_out = DATA_PROC / "priogrid_yearly_myanmar.csv"

myanmar_static.to_csv(static_out, index=False)  # save filtered static features
myanmar_yearly_out.to_csv(yearly_out, index=False)  # save filtered yearly features

print(f"\n✅ Static features saved  → {static_out}")
print(f"✅ Yearly features saved  → {yearly_out}")
print(f"   Static: {len(myanmar_static)} cells × {len(available) - 1} features")
print(
    f"   Yearly: {len(myanmar_yearly_out)} cells × {len(available_yearly) - 2} features"
)
