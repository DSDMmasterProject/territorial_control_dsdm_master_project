# ============================================================
# load_ucdp_events.py
# Load and label Myanmar UCDP event data
# Input:   data/raw/ucdp/GEDEvent_myanmar_merged.csv
#          (created by merge_ucdp_sources.py — run that first)
#          data/raw/validation/actor_taxonomy_myanmar.csv
#          (created by build_actor_taxonomy.py — run that first)
# Output:  data/processed/myanmar_ucdp_events.csv
#          data/processed/myanmar_ucdp_events_full_history.csv
# ============================================================

# Reads the merged UCDP file, filters to Myanmar post-coup events (Feb 2021 onwards),
# parses dates, adds a year-month key, and joins the actor taxonomy for both sides
# of each conflict. The result is a clean, analysis-ready event dataset with actor
# labels at three levels of granularity.


import pathlib  # pathlib: file path handling

import pandas as pd  # pandas: data loading and manipulation

# ── 1. PATHS ──────────────────────────────────────────────────

EVENTS_PATH = pathlib.Path(
    "data/raw/ucdp/GEDEvent_myanmar_merged.csv"
)  # Single merged source file
TAX_PATH = pathlib.Path(
    "data/raw/validation/actor_taxonomy_myanmar.csv"
)  # Actor crosswalk
OUT_DIR = pathlib.Path("data/processed")  # Output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Create if missing
OUT_PATH = OUT_DIR / "myanmar_ucdp_events.csv"  # Post-coup output
FULL_PATH = OUT_DIR / "myanmar_ucdp_events_full_history.csv"  # Full history output

# ── 2. LOAD THE MERGED FILE ───────────────────────────────────

print("=" * 60)
print("Loading UCDP Myanmar event data")
print("=" * 60)

if not EVENTS_PATH.exists():  # Check merged file exists before loading
    raise FileNotFoundError(  # Tell the user exactly what to do if missing
        f"Merged file not found: {EVENTS_PATH}\n"
        "Run src/data_collection/merge_ucdp_sources.py first."
    )

print("\n[A] Loading merged Myanmar UCDP file...")
print(f"   File: {EVENTS_PATH}")
print(f"   Size: {EVENTS_PATH.stat().st_size / 1e6:.1f} MB")

df = pd.read_csv(
    EVENTS_PATH, encoding="utf-8", low_memory=False
)  # Load the merged file

print(f"\n   Total Myanmar events loaded: {len(df):,}")
print("\n   Events by source:")
print(
    df["data_source"].value_counts().to_string()
)  # Show how many rows from each source file

# ── 3. PARSE DATES AND ADD YEAR-MONTH ─────────────────────────

print("\n[B] Parsing dates and adding year_month column...")

df["date_start"] = pd.to_datetime(
    df["date_start"], errors="coerce"
)  # Parse to datetime; coerce = bad dates become NaT
df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")  # Same for end date

df["year_month"] = (
    df["date_start"].dt.to_period("M").astype(str)
)  # Extract "YYYY-MM" string from each date

print(
    f"   Date range:          {df['date_start'].min().date()} → {df['date_start'].max().date()}"
)
print(f"   Year-months covered: {df['year_month'].nunique()}")

# ── 4. FILTER TO POST-COUP PERIOD ────────────────────────────

df_post2021 = df[df["date_start"] >= "2021-02-01"].copy()  # Feb 1 2021 = coup date
print(f"\n   Post-coup events (Feb 2021 onwards): {len(df_post2021):,}")

# ── 5. JOIN ACTOR TAXONOMY ────────────────────────────────────

print("\n[C] Joining actor taxonomy...")

tax = pd.read_csv(TAX_PATH, encoding="utf-8")  # Load our actor crosswalk

# Join taxonomy for side_a actors
df_post2021 = df_post2021.merge(
    tax[
        ["ucdp_name", "level1", "level2", "level3", "display_name"]
    ].rename(  # Rename columns to avoid collision with side_b
        columns={
            "ucdp_name": "side_a",
            "level1": "side_a_level1",
            "level2": "side_a_level2",
            "level3": "side_a_level3",
            "display_name": "side_a_display",
        }
    ),
    on="side_a",  # Join on actor name
    how="left",  # Keep all events even if actor not in taxonomy
)

# Join taxonomy for side_b actors
df_post2021 = df_post2021.merge(
    tax[["ucdp_name", "level1", "level2", "level3", "display_name"]].rename(
        columns={
            "ucdp_name": "side_b",
            "level1": "side_b_level1",
            "level2": "side_b_level2",
            "level3": "side_b_level3",
            "display_name": "side_b_display",
        }
    ),
    on="side_b",
    how="left",
)

# Report any actors that didn't match the taxonomy
unmatched_a = df_post2021[df_post2021["side_a_level1"].isna()]["side_a"].unique()
unmatched_b = df_post2021[df_post2021["side_b_level1"].isna()]["side_b"].unique()
if len(unmatched_a):
    print(f"   ⚠️  Unmatched side_a actors: {unmatched_a}")
if len(unmatched_b):
    print(f"   ⚠️  Unmatched side_b actors: {unmatched_b}")
if not len(unmatched_a) and not len(unmatched_b):
    print("   ✅ All actors matched in taxonomy")

# ── 6. SUMMARY STATISTICS ─────────────────────────────────────

print("\n── Post-2021 Myanmar Events Summary ──")
print(f"   Total events:     {len(df_post2021):,}")
print(f"   PRIOGRID cells:   {df_post2021['priogrid_gid'].nunique()}")
print(f"   Year-months:      {df_post2021['year_month'].nunique()}")
print(f"   Total fatalities: {df_post2021['best'].sum():,.0f}")

print("\n   Type of violence:")
tov = {1: "State-based", 2: "Non-state", 3: "One-sided"}
for k, v in tov.items():
    n = (df_post2021["type_of_violence"] == k).sum()
    print(f"   {v:<15} {n:>6,} events")

print("\n   Top 10 conflict dyads:")
print(df_post2021["dyad_name"].value_counts().head(10).to_string())

print("\n   Side A (initiator) breakdown:")
print(df_post2021["side_a"].value_counts().to_string())

# ── 7. SAVE ───────────────────────────────────────────────────

df_post2021.to_csv(OUT_PATH, index=False, encoding="utf-8")  # Save post-coup dataset
print(f"\n✅ Saved: {OUT_PATH}  ({len(df_post2021):,} rows)")

df.to_csv(FULL_PATH, index=False, encoding="utf-8")  # Save full history dataset
print(f"✅ Saved: {FULL_PATH}  ({len(df):,} rows, full history)")

print("\n" + "=" * 60)
print("COMPLETE.")
print("Next: build_priogrid_features.py = aggregate events to PRIOGRID cells per month")
print("=" * 60)
