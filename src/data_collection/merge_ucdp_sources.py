# ============================================================
# merge_ucdp_sources.py
# ONE-TIME script: merge all UCDP raw files into one clean file
# Run this once after downloading the raw UCDP files.
# All team members run this to get identical data.
#
# Input files (must be in data/raw/ucdp/):
#   GEDEvent_v25_1.csv           → 1989–2024 (annual GED)
#   GEDEvent_v25_01_25_12.csv    → Jan–Dec 2025 (Candidate)
#   GEDEvent_v26_01_26_03.csv    → Jan–Mar 2026 (Candidate)
#
# Output:
#   data/raw/ucdp/GEDEvent_myanmar_merged.csv   ← USE THIS in all scripts
# ============================================================

import pathlib  # pathlib: file path handling

import pandas as pd  # pandas: load and merge CSV files

# ── 1. DEFINE ALL SOURCE FILES ────────────────────────────────

UCDP_DIR = pathlib.Path("data/raw/ucdp")  # Folder containing all raw UCDP files

SOURCES = [  # List of all files to merge, in chronological order
    {
        "path": UCDP_DIR / "GEDEvent_v25_1.csv",  # Annual GED — gold standard quality
        "label": "GED_25.1",  # Human-readable source label
        "coverage": "1989-01-01 to 2024-12-31",  # What time period this covers
    },
    {
        "path": UCDP_DIR / "GEDEvent_v25_01_25_12.csv",  # Full year 2025 Candidate
        "label": "Candidate_25.01.25.12",
        "coverage": "2025-01-01 to 2025-12-31",
    },
    {
        "path": UCDP_DIR / "GEDEvent_v26_01_26_03.csv",  # Q1 2026 Candidate
        "label": "Candidate_26.01.26.03",
        "coverage": "2026-01-01 to 2026-03-31",
    },
]

OUTPUT_PATH = UCDP_DIR / "GEDEvent_myanmar_merged.csv"  # Output: one clean merged file

# ── 2. CHECK ALL FILES EXIST BEFORE LOADING ───────────────────

print("=" * 60)
print("UCDP Source Merger")
print("=" * 60)

missing = [
    s for s in SOURCES if not s["path"].exists()
]  # Find any files that are missing
if missing:  # If any are missing, stop and tell the user
    print("\n❌ Missing files — download these first:")
    for s in missing:
        print(f"   {s['path']}  ({s['coverage']})")
    raise FileNotFoundError("Missing UCDP source files. See instructions above.")

print("\n✅ All source files found:")
for s in SOURCES:  # Print each file with its size
    size_mb = (
        s["path"].stat().st_size / 1e6
    )  # stat().st_size gives file size in bytes; /1e6 = MB
    print(f"   {s['path'].name:<40} {size_mb:>7.1f} MB  ({s['coverage']})")

# ── 3. LOAD AND FILTER EACH FILE TO MYANMAR ───────────────────

print("\n[A] Loading and filtering each file to Myanmar...")

# We only load the columns we actually need for analysis
# This avoids loading all 49 GED columns and keeps the merged file small
KEEP_COLS = [
    "id",
    "year",
    "date_start",
    "date_end",
    "type_of_violence",
    "conflict_name",
    "dyad_name",
    "side_a",
    "side_b",
    "latitude",
    "longitude",
    "priogrid_gid",
    "country",
    "country_id",
    "adm_1",
    "adm_2",
    "best",
    "deaths_a",
    "deaths_b",
    "deaths_civilians",
    "deaths_unknown",
    "where_prec",
    "date_prec",
    "code_status",  # code_status: "clear" vs "preliminary" — useful for weighting Candidate events
]

chunks = []  # List to collect each file's Myanmar subset

for source in SOURCES:  # Loop through each source file
    print(f"\n   Loading {source['label']}...")

    available_cols = pd.read_csv(  # Read 0 rows just to see what columns exist
        source["path"], nrows=0
    ).columns.tolist()

    cols_to_load = [
        c for c in KEEP_COLS if c in available_cols
    ]  # Only request columns that actually exist
    missing_cols = [
        c for c in KEEP_COLS if c not in available_cols
    ]  # Note any columns missing from this file
    if missing_cols:
        print(f"   ⚠️  Columns not in this file (will be filled NA): {missing_cols}")

    df = pd.read_csv(  # Load the file, only selected columns
        source["path"],
        usecols=cols_to_load,
        encoding="utf-8",
        low_memory=False,  # Prevents mixed-type warnings on large files
    )

    mmr = df[
        df["country"] == "Myanmar (Burma)"
    ].copy()  # Filter to Myanmar only immediately

    mmr["data_source"] = source[
        "label"
    ]  # Add a column recording where this row came from

    for col in KEEP_COLS:  # Ensure all expected columns exist
        if col not in mmr.columns:  # If a column is missing from this file
            mmr[col] = pd.NA  # Fill with NA so all files have the same shape

    print(
        f"   {source['label']:<30} {len(df):>8,} global events → {len(mmr):>5,} Myanmar events"
    )
    chunks.append(mmr)  # Add Myanmar subset to our collection

# ── 4. CONCATENATE ALL CHUNKS ─────────────────────────────────

print("\n[B] Merging all chunks...")

merged = pd.concat(chunks, ignore_index=True)  # Stack all Myanmar subsets vertically
print(f"   Total before deduplication: {len(merged):,} events")

# ── 5. DEDUPLICATE BY EVENT ID ────────────────────────────────
#
# GED events have permanent numeric IDs. If a Candidate event was
# later reviewed and added to the annual GED, it would appear in
# both files with the same ID. We keep the GED version (higher
# quality) by sorting so GED rows come first, then deduplicating.
#
# The three files we're using are non-overlapping in time, so
# deduplication should change very little — but it's good practice.

merged = merged.sort_values(  # Sort so GED_25.1 rows come before Candidates
    "data_source",  # Alphabetical: Candidate_25... < Candidate_26... < GED_25.1
    ascending=False,  # Descending: GED_25.1 first (G > C alphabetically)
)

merged = merged.drop_duplicates(  # Remove any events with the same ID
    subset="id",  # Use event ID as the deduplication key
    keep="first",  # keep="first": keep GED version (it's sorted first)
)

merged = merged.sort_values("date_start").reset_index(
    drop=True
)  # Sort chronologically for clean output

print(f"   Total after deduplication:  {len(merged):,} events")
print(
    f"   Duplicates removed:         {len(pd.concat(chunks, ignore_index=True)) - len(merged):,}"
)

# ── 6. PRINT SUMMARY ─────────────────────────────────────────

print("\n── Merged Dataset Summary ──")
print(f"   Date range:    {merged['date_start'].min()} → {merged['date_start'].max()}")
print(f"   Total events:  {len(merged):,}")
print("\n   Events by source:")
print(
    merged["data_source"].value_counts().to_string()
)  # How many events came from each file
print("\n   Events by type of violence:")
tov = {
    1: "State-based",
    2: "Non-state",
    3: "One-sided",
}  # Human-readable violence type labels
for k, label in tov.items():
    n = (merged["type_of_violence"] == k).sum()
    print(f"   {label:<15} {n:>5,}")

# ── 7. SAVE ───────────────────────────────────────────────────

merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")  # Save merged file

size_mb = OUTPUT_PATH.stat().st_size / 1e6  # File size in MB
print(f"\n✅ Saved: {OUTPUT_PATH}")
print(f"   File size: {size_mb:.1f} MB")
print(f"   Rows: {len(merged):,}   Columns: {len(merged.columns)}")

print("\n" + "=" * 60)
print("MERGE COMPLETE.")
print("All team members: run this script once after downloading")
print("the 3 raw UCDP files. Then use GEDEvent_myanmar_merged.csv")
print("in all downstream scripts.")
print("=" * 60)
