#!/usr/bin/env python3
"""
export_figure_e.py — Figure E: per-cell recall on val transition events.

Data source: same 3 predictions.csv files as reporting.ipynb (production run,
train cutoff 2025-09, 10 features).  Transition-key logic is identical to
reporting.ipynb so the 31 events match Table 4 of the paper.

Output: src/reporting/figures/E_transition_recall_map.{pdf,png}

Run from project root:
    python src/visualizations/export_figure_e.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from pathlib import Path
from shapely.geometry import box

ROOT        = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "src" / "reporting" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
import pub_style
pub_style.apply()

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_PATHS = {
    "focal_mask"        : ROOT / "models" / "focal_mask"                 / "predictions.csv",
    "no_change"         : ROOT / "models" / "baseline_no_change"         / "predictions.csv",
    "random_proportion" : ROOT / "models" / "baseline_random_proportion" / "predictions.csv",
}
MODEL_ORDER = ["focal_mask", "no_change", "random_proportion"]
MODEL_LABELS = {
    "focal_mask"        : "U-net",
    "no_change"         : "No-Change\n(persistence)",
    "random_proportion" : "Random\n(proportional)",
}

# ── PRIOGRID helpers ──────────────────────────────────────────────────────────
def gid_to_latlon(gid):
    row = (gid - 1) // 720
    col = (gid - 1) % 720
    return row * 0.5 - 90.0 + 0.25, col * 0.5 - 180.0 + 0.25

def gid_to_box(gid):
    lat, lon = gid_to_latlon(gid)
    return box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25)

# ── Load predictions ──────────────────────────────────────────────────────────
print("Loading predictions …")
dfs = {}
for key in MODEL_ORDER:
    df = pd.read_csv(PRED_PATHS[key])
    df["year_month"] = df["year_month"].astype(str)
    lats, lons = zip(*df["priogrid_gid"].map(gid_to_latlon))
    df["lat"] = lats; df["lon"] = lons
    dfs[key] = df

val_dfs = {k: df[~df["is_train_month"]].copy() for k, df in dfs.items()}

# ── Transition keys (identical to reporting.ipynb) ───────────────────────────
def build_transition_keys(df_full):
    df = df_full[["priogrid_gid", "year_month", "true_class", "is_train_month"]].copy()
    df = df.sort_values(["priogrid_gid", "year_month"]).reset_index(drop=True)
    df["prev_true"] = df.groupby("priogrid_gid")["true_class"].shift(1)
    val = df[~df["is_train_month"]].copy()
    val["prev_true"] = val["prev_true"].fillna(val["true_class"])
    val["is_transition"] = val["true_class"] != val["prev_true"]
    return frozenset(zip(
        val.loc[val["is_transition"], "priogrid_gid"],
        val.loc[val["is_transition"], "year_month"],
    ))

transition_keys = build_transition_keys(dfs["focal_mask"])
print(f"  Transition events in val set: {len(transition_keys)}  "
      f"(unique cells: {len({g for g, _ in transition_keys})})")

# ── Myanmar boundaries ────────────────────────────────────────────────────────
gadm = ROOT / "data" / "raw" / "gadm" / "gadm41_MMR.gpkg"
mmr0 = gpd.read_file(gadm, layer="ADM_ADM_0")
mmr1 = gpd.read_file(gadm, layer="ADM_ADM_1")

# ── PRIOGRID cell geometries ──────────────────────────────────────────────────
unique_gids = (dfs["focal_mask"][["priogrid_gid", "lat", "lon"]]
               .drop_duplicates("priogrid_gid"))
cells_gdf = gpd.GeoDataFrame(
    unique_gids.reset_index(drop=True),
    geometry=unique_gids["priogrid_gid"].map(gid_to_box).values,
    crs="EPSG:4326",
)
# Clip to Myanmar boundary so cells outside the country (e.g. gid=150307,
# Andaman Sea) are not rendered — data and metrics are unchanged.
cells_gdf = gpd.clip(cells_gdf, mmr0)

# ── Per-model per-cell recall on transition events ────────────────────────────
recall_rows = []
for key in MODEL_ORDER:
    vdf = val_dfs[key].copy()
    vdf["_tk"] = list(zip(vdf["priogrid_gid"], vdf["year_month"]))
    trans = vdf[vdf["_tk"].isin(transition_keys)].copy()
    if len(trans) == 0:
        continue
    trans["hit"]  = trans["pred_class"] == trans["true_class"]
    cell_recall   = (trans.groupby("priogrid_gid")["hit"].mean()
                     .reset_index().rename(columns={"hit": "recall"}))
    overall       = trans["hit"].mean()
    recall_rows.append({
        "key": key, "label": MODEL_LABELS[key],
        "overall_recall": overall, "cell_recall": cell_recall,
    })
    print(f"  {MODEL_LABELS[key].replace(chr(10), ' '):30s}  "
          f"recall={overall:.3f}  (n={len(trans)})")

# ── Figure E — publication style ─────────────────────────────────────────────
print("\nRendering figure E …")

n = len(recall_rows)
# Each Myanmar panel: xlim 14° wide, ylim 21° tall → aspect ratio 1.5 (tall).
# At 4.5in per panel, height ≈ 6.75in; add title headroom → 7in total.
fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 7),
                         facecolor="white", edgecolor="white")
if n == 1:
    axes = [axes]
axes = np.array(axes).flatten()

cmap  = mcm.get_cmap("RdYlGn")
norm  = mcolors.Normalize(vmin=0, vmax=1)

for i, row in enumerate(recall_rows):
    ax = axes[i]
    ax.set_facecolor("white")

    merged = cells_gdf.merge(row["cell_recall"], on="priogrid_gid", how="left")
    no_trans = merged["recall"].isna()

    # Grey fill for stable cells (no transition event)
    merged[no_trans].plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.3)

    # Choropleth for cells with ≥1 transition event
    if not merged[~no_trans].empty:
        merged[~no_trans].plot(
            ax=ax, column="recall", cmap="RdYlGn", vmin=0, vmax=1,
            edgecolor="white", linewidth=0.5,
            legend=False,   # handled manually below
        )

    mmr0.boundary.plot(ax=ax, color="black",   linewidth=1.2)
    mmr1.boundary.plot(ax=ax, color="#555555", linewidth=0.5)

    label_clean = row["label"].replace("\n", " ")
    ax.set_title(f"{label_clean}\nrecall = {row['overall_recall']:.3f}",
                 fontsize=10, fontweight="bold", pad=6)
    ax.axis("off")
    ax.set_xlim(88, 102)
    ax.set_ylim(8, 29)

# Shared colorbar below all panels
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=axes, orientation="horizontal",
    fraction=0.025, pad=0.04, shrink=0.6, aspect=30,
)
cbar.set_label("Transition Recall  (grey = no transition event in val set)",
               fontsize=9)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout(pad=0.5)

# ── Save ──────────────────────────────────────────────────────────────────────
stem = FIGURES_DIR / "E_transition_recall_map"
pub_style.save(fig, stem)
plt.close(fig)
print("Done.")
