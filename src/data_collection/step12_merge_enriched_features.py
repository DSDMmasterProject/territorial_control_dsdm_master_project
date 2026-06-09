# ── step12_merge_enriched_features.py ────────────────────────────────────────
# Merges PRIOGRID catalog features (terrain + resources) into the UCDP feature matrix.
# Static features broadcast to all months (same value per cell, every month).
# Yearly features: latest available year used as static proxy for 2021-2026.
# Output: data/processed/myanmar_priogrid_features_enriched.csv
# Run from project root: uv run python src/data_collection/step12_merge_enriched_features.py

from pathlib import Path  # file path handling

import pandas as pd  # dataframe operations

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_PROC = PROJECT_ROOT / "data" / "processed"  # processed data folder

# ── 1. Load base UCDP feature matrix ─────────────────────────────────────────
features = pd.read_csv(DATA_PROC / "myanmar_priogrid_features.csv")
print(
    f"Base feature matrix: {features.shape}  ({features['priogrid_gid'].nunique()} cells)"
)

# ── 2. Load PRIOGRID static features (terrain + resources) ───────────────────
static = pd.read_csv(DATA_PROC / "priogrid_static_myanmar.csv")
print(
    f"Static features:     {static.shape}  ({static['priogrid_gid'].nunique()} cells)"
)

# ── 3. Load PRIOGRID yearly features (latest year per cell) ──────────────────
yearly = pd.read_csv(DATA_PROC / "priogrid_yearly_myanmar.csv")
print(
    f"Yearly features:     {yearly.shape}  ({yearly['priogrid_gid'].nunique()} cells)"
)

# ── 4. Identify which new feature columns are available ──────────────────────
# We skip non-feature columns (gid, year) when merging.
base_cols = set(features.columns)
static_feature_cols = [c for c in static.columns if c != "priogrid_gid"]
yearly_feature_cols = [c for c in yearly.columns if c not in ["priogrid_gid", "year"]]

print(f"\nStatic features to merge: {static_feature_cols}")
print(f"Yearly features to merge: {yearly_feature_cols}")

# ── 5. Merge static features — broadcasts to all months automatically ─────────
# Merging on priogrid_gid only → each cell gets its static value repeated
# for every monthly row, which is exactly what we want.
enriched = features.merge(static, on="priogrid_gid", how="left")

# ── 6. Merge yearly features — latest year used as static proxy ───────────────
# Drop the 'year' column from yearly before merging (we don't want it in output).
if yearly_feature_cols:  # only merge if there are actual yearly features
    yearly_for_merge = yearly.drop(columns=["year"], errors="ignore")
    enriched = enriched.merge(yearly_for_merge, on="priogrid_gid", how="left")

# ── 7. Fill any remaining NaN with 0 ─────────────────────────────────────────
# Cells outside PRIOGRID coverage get 0 — treated as "no resource/no drug crop".
new_cols = [c for c in enriched.columns if c not in base_cols]
enriched[new_cols] = enriched[new_cols].fillna(0.0)

print(f"\nEnriched matrix: {enriched.shape}")
print(f"New columns added ({len(new_cols)}): {new_cols}")

# ── 8. Print statistics for the new features ─────────────────────────────────
print("\nNew feature statistics (all months, 199 cells × 29 months):")
print(enriched[new_cols].describe().round(3))

# ── 9. Verify coverage — every Myanmar cell should have static values ─────────
n_missing = enriched[new_cols].isna().sum().sum()  # count any remaining NaN
print(
    f"\nRemaining NaN after fill: {n_missing}  ({'✅ clean' if n_missing == 0 else '⚠️  check merge'})"
)

# ── 10. Save ──────────────────────────────────────────────────────────────────
out = DATA_PROC / "myanmar_priogrid_features_enriched.csv"
enriched.to_csv(out, index=False)
print(f"\n✅ Enriched feature matrix saved → {out}")
print(
    f"   {features.shape[1]} original + {len(new_cols)} new = {enriched.shape[1]} total columns"
)
