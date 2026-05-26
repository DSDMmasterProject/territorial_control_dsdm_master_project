# ============================================================
# eda_myanmar.py
# Exploratory Data Analysis for Myanmar territorial control project
# Covers: UCDP events, Wikipedia labels, training dataset
# Saves all figures to reports/figures/eda/
# ============================================================

import pathlib  # pathlib: file paths
import warnings

import geopandas as gpd  # geopandas: geographic plotting
import matplotlib.pyplot as plt  # matplotlib: plotting
import numpy as np  # numpy: numerical operations
import pandas as pd  # pandas: data manipulation

warnings.filterwarnings("ignore")  # suppress minor warnings for clean output

# ── PATHS ─────────────────────────────────────────────────────

EDA_DIR = pathlib.Path("reports/figures/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = pathlib.Path("data/processed/myanmar_ucdp_events.csv")
FEATURES = pathlib.Path("data/processed/myanmar_priogrid_features.csv")
LABELS = pathlib.Path("data/processed/myanmar_priogrid_labels.csv")
TRAINING = pathlib.Path("data/processed/myanmar_training_dataset.csv")
WIKI = pathlib.Path("data/raw/validation/myanmar_wikipedia_temporal_groundtruth.csv")
GADM = pathlib.Path("data/raw/gadm/gadm41_MMR.gpkg")

# ── COLOUR SCHEMES ────────────────────────────────────────────

STATUS_COLOURS = {  # Consistent colours across all plots
    "government": "#CC3333",
    "resistance": "#33AA44",
    "ethnic_armed_group": "#2266CC",
    "contested": "#FF9900",
}

print("=" * 60)
print("Myanmar Territorial Control — EDA")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# PART 1: UCDP EVENT DATA
# ═══════════════════════════════════════════════════════════════

print("\n[PART 1] UCDP Event Data")

events = pd.read_csv(EVENTS, encoding="utf-8")  # Load post-coup event data
events["date_start"] = pd.to_datetime(events["date_start"])  # Parse dates

print(f"   Events loaded: {len(events):,}")
print(
    f"   Date range: {events['date_start'].min().date()} → {events['date_start'].max().date()}"
)
print(f"   PRIOGRID cells: {events['priogrid_gid'].nunique()}")
print(f"   Total fatalities: {events['best'].sum():,.0f}")

# ── Figure 1A: Events per month over time ─────────────────────

monthly = (
    events.groupby("year_month")
    .agg(n_events=("id", "count"), fatalities=("best", "sum"))
    .reset_index()
)
monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"] + "-01")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#F8F6F0")
fig.suptitle(
    "UCDP Events — Myanmar Post-Coup Timeline (Feb 2021 – Mar 2026)",
    fontsize=13,
    fontweight="bold",
)

# Events per month
ax = axes[0]
ax.set_facecolor("#F8F6F0")
ax.bar(
    monthly["year_month_dt"],
    monthly["n_events"],  # Bar chart of monthly event counts
    color="#CC3333",
    alpha=0.7,
    width=20,
)  # width=20 days per bar
ax.set_ylabel("Events per month", fontsize=10)
ax.set_title("Conflict Event Frequency", fontsize=10)
ax.axvline(
    pd.Timestamp("2023-10-27"),
    color="black",  # Operation 1027 reference line
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Op. 1027 (Oct 2023)",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Fatalities per month
ax2 = axes[1]
ax2.set_facecolor("#F8F6F0")
ax2.bar(
    monthly["year_month_dt"],
    monthly["fatalities"],
    color="#882222",
    alpha=0.7,
    width=20,
)
ax2.set_ylabel("Fatalities per month", fontsize=10)
ax2.set_title("Monthly Fatalities (Best Estimate)", fontsize=10)
ax2.axvline(
    pd.Timestamp("2023-10-27"), color="black", linestyle="--", linewidth=1.5, alpha=0.7
)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EDA_DIR / "1a_ucdp_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 1A saved: 1a_ucdp_timeline.png")

# ── Figure 1B: Dyad breakdown ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#F8F6F0")
fig.suptitle("UCDP Events — Who is Fighting Whom?", fontsize=13, fontweight="bold")

# Top 15 dyads by event count
dyad_counts = events["dyad_name"].value_counts().head(15)
ax = axes[0]
ax.set_facecolor("#F8F6F0")
bars = ax.barh(
    range(len(dyad_counts)),  # Horizontal bar chart
    dyad_counts.values,
    color="#CC5555",
    alpha=0.8,
)
ax.set_yticks(range(len(dyad_counts)))
ax.set_yticklabels(
    [
        d.replace("Government of Myanmar (Burma) - ", "Gov. - ").replace(
            "Government of Myanmar (Burma)", "Gov. Myanmar"
        )
        for d in dyad_counts.index
    ],
    fontsize=8,
)  # Shorten long names
ax.set_xlabel("Number of events")
ax.set_title("Top 15 Conflict Dyads by Event Count", fontsize=10)
ax.invert_yaxis()  # Highest bar at top
ax.grid(True, alpha=0.3, axis="x")

# Violence type breakdown
tov_labels = {
    1: "State-based\n(armed actors)",
    2: "Non-state\n(non-govt actors)",
    3: "One-sided\n(vs civilians)",
}
tov_counts = events["type_of_violence"].value_counts().sort_index()
ax2 = axes[1]
ax2.set_facecolor("#F8F6F0")
colours_tov = ["#CC3333", "#3366CC", "#FF6600"]  # Different colour per type
wedges, texts, autotexts = ax2.pie(  # Pie chart of violence types
    tov_counts.values,
    labels=[tov_labels[k] for k in tov_counts.index],
    colors=colours_tov,
    autopct="%1.1f%%",  # autopct: show percentage labels
    startangle=90,
    textprops={"fontsize": 9},
)
ax2.set_title("Violence Type Distribution", fontsize=10)

plt.tight_layout()
fig.savefig(EDA_DIR / "1b_ucdp_actors.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 1B saved: 1b_ucdp_actors.png")

# ── Figure 1C: Geographic distribution of events ─────────────

fig, ax = plt.subplots(figsize=(9, 12), facecolor="#F0EEE8")
ax.set_facecolor("#C8D8E8")

if GADM.exists():
    mmr = gpd.read_file(GADM, layer="ADM_ADM_0").to_crs("EPSG:4326")
    mmr.plot(ax=ax, color="#E8E0D0", edgecolor="#999988", linewidth=0.5)

# Compute cell centres for plotting
feat = pd.read_csv(FEATURES, usecols=["priogrid_gid", "total_events"])
cell_totals = feat.groupby("priogrid_gid")["total_events"].sum().reset_index()
cell_totals["row"] = (cell_totals["priogrid_gid"] - 1) // 720
cell_totals["col"] = (cell_totals["priogrid_gid"] - 1) % 720
cell_totals["lat"] = cell_totals["row"] * 0.5 - 90 + 0.25  # Cell centre latitude
cell_totals["lon"] = cell_totals["col"] * 0.5 - 180 + 0.25  # Cell centre longitude

# Size = total events (log scale for visibility)
sizes = np.log1p(cell_totals["total_events"]) * 15  # log1p = log(1 + x) handles zeros
sc = ax.scatter(
    cell_totals["lon"],
    cell_totals["lat"],
    s=sizes,
    c=cell_totals["total_events"],
    cmap="YlOrRd",  # Yellow-Orange-Red colour scale
    alpha=0.7,
    zorder=5,
    marker="s",
)  # s = square marker
plt.colorbar(sc, ax=ax, label="Total events (2021-2026)", shrink=0.5)
ax.set_xlim(92, 102)
ax.set_ylim(9, 29)
ax.set_title(
    "UCDP Conflict Intensity by PRIOGRID Cell\n(2021–2026, all event types)",
    fontsize=11,
    fontweight="bold",
)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, alpha=0.3, color="white")
plt.tight_layout()
fig.savefig(EDA_DIR / "1c_ucdp_geographic.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 1C saved: 1c_ucdp_geographic.png")

# ── Print key UCDP statistics ─────────────────────────────────

print("\n   ── UCDP Key Statistics ──")
print(f"   Months with no events:  {(monthly['n_events'] == 0).sum()}")
print(
    f"   Month with most events: {monthly.loc[monthly['n_events'].idxmax(), 'year_month']} ({monthly['n_events'].max()} events)"
)
print(
    f"   Month with most deaths: {monthly.loc[monthly['fatalities'].idxmax(), 'year_month']} ({monthly['fatalities'].max():.0f} deaths)"
)
print(f"   Most active dyad:       {events['dyad_name'].value_counts().index[0]}")
print(
    f"   % events by government: {(events['side_a'] == 'Government of Myanmar (Burma)').mean() * 100:.1f}%"
)

# ═══════════════════════════════════════════════════════════════
# PART 2: WIKIPEDIA GROUND TRUTH DATA
# ═══════════════════════════════════════════════════════════════

print("\n[PART 2] Wikipedia Ground Truth Data")

wiki = pd.read_csv(WIKI, encoding="utf-8")  # Load Wikipedia temporal labels
wiki["date"] = pd.to_datetime(wiki["date"])

print(f"   Observations: {len(wiki):,}")
print(f"   Unique locations: {wiki[['lat', 'lon']].drop_duplicates().shape[0]}")
print(f"   Snapshots: {wiki['date'].nunique()}")

# ── Figure 2A: Control status composition over time ───────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#F8F6F0")
fig.suptitle(
    "Wikipedia Ground Truth — Territorial Control Over Time",
    fontsize=13,
    fontweight="bold",
)

# Stacked area chart: proportion over time
pivot = (
    wiki[wiki["control_status"].isin(STATUS_COLOURS.keys())]
    .groupby(["date", "control_status"])
    .size()
    .unstack(fill_value=0)
)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100  # Convert to percentages

ax = axes[0]
ax.set_facecolor("#F8F6F0")
bottom = np.zeros(len(pivot_pct))  # Track cumulative bottom for stacking
for status in ["government", "contested", "ethnic_armed_group", "resistance"]:
    if status in pivot_pct.columns:
        ax.fill_between(
            pivot_pct.index,
            bottom,
            bottom + pivot_pct[status].values,
            color=STATUS_COLOURS[status],
            alpha=0.8,
            label=status.replace("_", " ").title(),
        )
        bottom += pivot_pct[status].values
ax.set_ylabel("% of labeled locations", fontsize=10)
ax.set_xlabel("Snapshot date", fontsize=10)
ax.set_ylim(0, 100)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("Control Status Composition Over Time (stacked %)", fontsize=10)
ax.grid(True, alpha=0.3)

# Count over time (absolute)
ax2 = axes[1]
ax2.set_facecolor("#F8F6F0")
for status, colour in STATUS_COLOURS.items():
    if status in pivot.columns:
        ax2.plot(
            pivot.index,
            pivot[status],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
            label=status.replace("_", " ").title(),
        )
ax2.set_ylabel("Number of labeled locations", fontsize=10)
ax2.set_xlabel("Snapshot date", fontsize=10)
ax2.legend(fontsize=9)
ax2.set_title("Absolute Count per Control Status Over Time", fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EDA_DIR / "2a_wikipedia_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 2A saved: 2a_wikipedia_temporal.png")

# ── Figure 2B: Wikipedia coverage on map ─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor="#F0EEE8")
fig.suptitle(
    "Wikipedia Ground Truth — Geographic Coverage (Latest Snapshot: Mar 2026)",
    fontsize=13,
    fontweight="bold",
)

latest = wiki[wiki["date"] == wiki["date"].max()]  # Most recent snapshot only

for idx, (high_only, title) in enumerate(
    [(False, "All labels"), (True, "High-confidence labels only")]
):
    ax = axes[idx]
    ax.set_facecolor("#C8D8E8")
    if GADM.exists():
        mmr.plot(ax=ax, color="#E8E0D0", edgecolor="#999988", linewidth=0.5)

    subset = latest if not high_only else latest[latest["confidence"] == "high"]
    for status, colour in STATUS_COLOURS.items():
        sub = subset[subset["control_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["lon"],
            sub["lat"],
            c=colour,
            s=25,
            alpha=0.85,
            label=f"{status} (n={len(sub)})",
            zorder=5,
        )

    ax.set_xlim(92, 102)
    ax.set_ylim(9, 29)
    ax.set_title(f"{title} — {len(subset)} points", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3, color="white")

plt.tight_layout()
fig.savefig(EDA_DIR / "2b_wikipedia_geographic.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 2B saved: 2b_wikipedia_geographic.png")

# ── Print Wikipedia statistics ────────────────────────────────

print("\n   ── Wikipedia Key Statistics ──")
print("   Label distribution (all snapshots):")
dist = wiki["control_status"].value_counts()
for s, n in dist.items():
    print(f"   {s:<22} {n:>6,} ({100 * n / len(wiki):.1f}%)")
print("   Confidence breakdown:")
print(f"   {wiki['confidence'].value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════
# PART 3: TRAINING DATASET — FEATURES + LABELS
# ═══════════════════════════════════════════════════════════════

print("\n[PART 3] Training Dataset — Features vs Labels")

train = pd.read_csv(TRAINING, encoding="utf-8")  # Load training dataset

FEATURE_COLS = [
    c
    for c in train.columns
    if c
    not in [
        "priogrid_gid",
        "year_month",
        "year_month_dt",
        "adm_1",
        "control_status",
        "label_confidence",
        "n_towns_in_cell",
        "high_conf_towns",
        "is_direct_snapshot",
        "split",
    ]
]

print(f"   Training rows: {len(train):,}")
print(f"   Features: {len(FEATURE_COLS)}")
print("   Class distribution:")
print(f"   {train['control_status'].value_counts().to_string()}")

# ── Figure 3A: Feature distributions by class ─────────────────

# Select the most interpretable features to show
KEY_FEATURES = [
    "total_events",
    "total_fatalities",
    "events_gov",
    "events_nug",
    "events_ula",
    "events_kio",
    "gov_vs_civilians",
    "gov_event_share",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 10), facecolor="#F8F6F0")
fig.suptitle(
    "Feature Distributions by Control Status Class\n(How UCDP features differ across control classes)",
    fontsize=13,
    fontweight="bold",
)

axes_flat = axes.flatten()
for i, feat in enumerate(KEY_FEATURES):
    ax = axes_flat[i]
    ax.set_facecolor("#F8F6F0")
    for status, colour in STATUS_COLOURS.items():
        subset = train[train["control_status"] == status][feat]
        # Use violin plot for richer distribution view
        parts = ax.violinplot(
            [subset.values],  # violinplot: shows full distribution shape
            positions=[list(STATUS_COLOURS.keys()).index(status)],
            widths=0.7,
            showmedians=True,
        )  # showmedians: draw median line
        for pc in parts["bodies"]:
            pc.set_facecolor(colour)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmins"].set_color("black")
        parts["cmaxes"].set_color("black")
        parts["cbars"].set_color("black")

    ax.set_xticks(range(len(STATUS_COLOURS)))
    ax.set_xticklabels(
        [s[:4] for s in STATUS_COLOURS.keys()],  # Abbreviated labels
        fontsize=8,
    )
    ax.set_title(feat.replace("_", " "), fontsize=9, fontweight="bold")
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    # Log scale for heavily skewed distributions
    if feat in ["total_events", "total_fatalities", "events_gov"]:
        ax.set_yscale("log")  # Log scale: compresses extreme values
        ax.set_ylabel("Value (log scale)", fontsize=8)

plt.tight_layout()
fig.savefig(EDA_DIR / "3a_feature_by_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 3A saved: 3a_feature_by_class.png")

# ── Figure 3B: Correlation heatmap ───────────────────────────

fig, ax = plt.subplots(figsize=(16, 14), facecolor="#F8F6F0")
corr = train[FEATURE_COLS].corr()  # Pearson correlation matrix

# Mask upper triangle (mirror of lower — no need to show twice)
mask = np.triu(
    np.ones_like(corr, dtype=bool), k=1
)  # k=1: exclude diagonal too? No, k=0 excludes diagonal

im = ax.imshow(
    corr.values,
    cmap="RdBu_r",  # Red-Blue diverging colour map
    vmin=-1,
    vmax=1,
    aspect="auto",
)
plt.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=7)
ax.set_yticklabels(FEATURE_COLS, fontsize=7)
ax.set_title(
    "Feature Correlation Matrix\n(Red=positive, Blue=negative, White=uncorrelated)",
    fontsize=12,
    fontweight="bold",
)

plt.tight_layout()
fig.savefig(EDA_DIR / "3b_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 3B saved: 3b_correlation_matrix.png")

# ── Figure 3C: Class imbalance + temporal distribution ────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#F8F6F0")
fig.suptitle(
    "Training Dataset — Class Balance and Temporal Coverage",
    fontsize=13,
    fontweight="bold",
)

# Class imbalance bar chart
ax = axes[0]
ax.set_facecolor("#F8F6F0")
vc = train["control_status"].value_counts()
bars = ax.bar(
    range(len(vc)), vc.values, color=[STATUS_COLOURS[s] for s in vc.index], alpha=0.8
)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels([s.replace("_", "\n") for s in vc.index], fontsize=9)
ax.set_ylabel("Number of cell-months")
ax.set_title("Class Distribution (all 4,229 rows)", fontsize=10)
for bar, val in zip(bars, vc.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        f"{val:,}\n({100 * val / len(train):.1f}%)",
        ha="center",
        fontsize=8,
    )
ax.grid(True, alpha=0.3, axis="y")

# Class distribution: TRAIN vs VALIDATE
ax2 = axes[1]
ax2.set_facecolor("#F8F6F0")
train_dist = (
    train[train["split"] == "train"]["control_status"].value_counts(normalize=True)
    * 100
)
val_dist = (
    train[train["split"] == "validate"]["control_status"].value_counts(normalize=True)
    * 100
)
x = np.arange(len(STATUS_COLOURS))
width = 0.35
for i, status in enumerate(STATUS_COLOURS):
    t_val = train_dist.get(status, 0)
    v_val = val_dist.get(status, 0)
    ax2.bar(
        i - width / 2,
        t_val,
        width,
        color=STATUS_COLOURS[status],
        alpha=0.9,
        label=status if i == 0 else "",
    )
    ax2.bar(i + width / 2, v_val, width, color=STATUS_COLOURS[status], alpha=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace("_", "\n") for s in STATUS_COLOURS], fontsize=8)
ax2.set_ylabel("% of split rows")
ax2.set_title("Class % in Train (solid) vs Validate (faded)", fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# Monthly event count coloured by dominant class
ax3 = axes[2]
ax3.set_facecolor("#F8F6F0")
monthly_class = (
    train.groupby(["year_month", "control_status"]).size().unstack(fill_value=0)
)
monthly_class_pct = monthly_class.div(monthly_class.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(monthly_class_pct))
x_vals = range(len(monthly_class_pct))
for status in ["government", "contested", "ethnic_armed_group", "resistance"]:
    if status in monthly_class_pct.columns:
        ax3.bar(
            x_vals,
            monthly_class_pct[status].values,
            bottom=bottom,
            color=STATUS_COLOURS[status],
            alpha=0.8,
            width=0.8,
            label=status.replace("_", " ").title(),
        )
        bottom += monthly_class_pct[status].values
ax3.set_xlabel("Month (Nov 2023 → Mar 2026)")
ax3.set_ylabel("% of labeled cells per month")
ax3.set_title(
    "Class Distribution Per Month\n(shows how labels change over time)", fontsize=10
)
ax3.set_xticks([])  # Too many tick labels — hide them
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(EDA_DIR / "3c_class_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 3C saved: 3c_class_balance.png")

# ── Figure 3D: Training dataset on map (most recent month) ───

fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor="#F0EEE8")
fig.suptitle(
    "Training Dataset — Spatial Label Distribution\nLeft: Mar 2026 (most recent)  |  Right: Nov 2023 (earliest)",
    fontsize=13,
    fontweight="bold",
)


def convert_gid(df):
    """Convert priogrid_gid to cell centre lat/lon."""
    df = df.copy()
    df["row"] = (df["priogrid_gid"] - 1) // 720
    df["col"] = (df["priogrid_gid"] - 1) % 720
    df["cell_lat"] = df["row"] * 0.5 - 90 + 0.25
    df["cell_lon"] = df["col"] * 0.5 - 180 + 0.25
    return df


for idx, month in enumerate(["2026-03", "2023-11"]):
    ax = axes[idx]
    ax.set_facecolor("#C8D8E8")
    if GADM.exists():
        mmr.plot(ax=ax, color="#E8E0D0", edgecolor="#999988", linewidth=0.5)

    snap = convert_gid(train[train["year_month"] == month])
    for status, colour in STATUS_COLOURS.items():
        sub = snap[snap["control_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["cell_lon"],
            sub["cell_lat"],
            c=colour,
            s=80,
            alpha=0.85,
            marker="s",
            label=f"{status} ({len(sub)})",
            zorder=5,
        )

    ax.set_xlim(92, 102)
    ax.set_ylim(9, 29)
    ax.set_title(
        f"Month: {month}  |  {len(snap)} labeled cells", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, alpha=0.3, color="white")

plt.tight_layout()
fig.savefig(EDA_DIR / "3d_training_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Figure 3D saved: 3d_training_maps.png")

# ── Print final key statistics ────────────────────────────────

print("\n   ── Training Dataset Key Statistics ──")
print(
    f"   Sparsity (% zero-event rows in features): "
    f"{100 * (pd.read_csv(FEATURES)['total_events'] == 0).mean():.1f}%"
)
print("   Most correlated feature pair (check for multicollinearity):")
corr_vals = corr.abs().unstack()  # Flatten correlation matrix
corr_vals = corr_vals[corr_vals < 1.0].sort_values(
    ascending=False
)  # Exclude self-correlations
print(f"   {corr_vals.index[0]} = {corr_vals.iloc[0]:.3f}")
print("   Least correlated with any other feature:")
mean_abs_corr = corr.abs().mean()
print(f"   {mean_abs_corr.idxmin()} = {mean_abs_corr.min():.3f}")

# ── Summary ──────────────────────────────────────────────────

print(f"""
{"=" * 60}
EDA COMPLETE
{"=" * 60}
Figures saved to: {EDA_DIR}/

  1a_ucdp_timeline.png      ← event frequency + fatalities over time
  1b_ucdp_actors.png        ← who fights whom, violence type breakdown
  1c_ucdp_geographic.png    ← conflict intensity map by PRIOGRID cell
  2a_wikipedia_temporal.png ← control status composition over 24 snapshots
  2b_wikipedia_geographic.png ← labeled points on Myanmar map
  3a_feature_by_class.png   ← violin plots: how features differ by class
  3b_correlation_matrix.png ← multicollinearity check across 43 features
  3c_class_balance.png      ← class imbalance + temporal distribution
  3d_training_maps.png      ← labeled cells on map (Mar 2026 vs Nov 2023)

KEY TAKEAWAYS FOR MODELLING:
  • Baseline to beat:  60.8% (always predict government)
  • Class imbalance:   60.8% government — consider class weights or SMOTE
  • Sparsity:          ~77.8% of cell-months have zero events
  • Strongest signal:  gov_event_share, events_nug, lag features
  • Multicollinearity: lag features are highly correlated with each other
                       → consider PCA or feature selection before modelling
{"=" * 60}
""")
