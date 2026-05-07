# ============================================================
# STEP 6: Temporal visualisation — animated map + trend chart
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date: 07.05.2026
# ============================================================

# The script produces three complementary outputs from the same 8,035 rows.
# First it computes monthly proportions of each control status
# how many percent of tracked locations were government-held, contested, etc. in each of the 24 snapshots
# and plots them as overlapping lines on a trend chart,
# with a shaded area under the contested line to dramatise its growth and a reference marker for Operation 1027.
# Second it arranges all 24 snapshots in a 4×6 grid of small maps for the written report,
# each panel labelled with its date and contested count.
# Third it uses FuncAnimation to cycle through the same 24 frames as a GIF,
# drawing Myanmar fresh each frame with a progress bar along the top edge and a live control status summary box.

# Imports:

import pathlib  # pathlib: clean file path handling

import geopandas as gpd  # geopandas: geographic DataFrames for plotting on maps
import matplotlib.animation as animation  # animation: FuncAnimation for building GIF frames
import matplotlib.patches as mpatches  # mpatches: custom legend colour patches
import matplotlib.pyplot as plt  # matplotlib: core plotting library
import pandas as pd  # pandas: load and manipulate our temporal CSV

# ── 1. PATHS ─────────────────────────────────────────────────

CSV_PATH = pathlib.Path(
    "data/raw/validation/myanmar_wikipedia_temporal_groundtruth.csv"
)  # Our 8,035-row temporal dataset
GADM_PATH = pathlib.Path("data/raw/gadm/gadm41_MMR.gpkg")  # Myanmar admin boundary file
OUT_DIR = pathlib.Path("reports/figures")  # Output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Create if missing

GIF_PATH = (
    OUT_DIR / "myanmar_territorial_control_animated.gif"
)  # Output path for the animated GIF
TREND_PATH = (
    OUT_DIR / "myanmar_control_trend_timeseries.png"
)  # Output path for the trend chart
GRID_PATH = (
    OUT_DIR / "myanmar_control_smallmultiples.png"
)  # Output path for small multiples grid

# ── 2. COLOUR AND DISPLAY SETTINGS ───────────────────────────

COLOUR_MAP = {  # Control status → hex colour (same scheme as Step 4)
    "government": "#CC3333",  # Red = SAC/Military
    "resistance": "#33AA44",  # Green = resistance/NUG
    "ethnic_armed_group": "#2266CC",  # Blue = ethnic armed organisations
    "contested": "#FF9900",  # Orange = contested zones
    "unknown": "#999999",  # Grey = ambiguous
}

DISPLAY_NAMES = {  # Shorter display names for legends and chart labels
    "government": "Gov / SAC",
    "ethnic_armed_group": "Ethnic EAO",
    "contested": "Contested",
    "resistance": "Resistance",
    "unknown": "Unknown",
}

STATUS_ORDER = [  # Consistent order for stacking and legends
    "government",
    "ethnic_armed_group",
    "contested",
    "resistance",
    "unknown",
]

# ── 3. LOAD DATA ─────────────────────────────────────────────

print("=" * 60)
print("STEP 6: Temporal Visualisation")
print("=" * 60)

df = pd.read_csv(CSV_PATH, encoding="utf-8")  # Load all 8,035 temporal observations
df["date"] = pd.to_datetime(
    df["date"]
)  # Convert date strings to datetime objects for sorting
dates = sorted(df["date"].unique())  # Get sorted list of all unique snapshot dates
print(f"✅ Loaded {len(df):,} rows across {len(dates)} snapshots")  # Confirm load

myanmar = gpd.read_file(
    GADM_PATH, layer="ADM_ADM_0"
)  # Load Myanmar country outline (level 0 only)
myanmar = myanmar.to_crs("EPSG:4326")  # Reproject to standard WGS84 lat/lon
print("✅ Myanmar boundary loaded")  # Confirm

# ── 4. PRE-COMPUTE PROPORTIONS FOR TREND CHART ───────────────

# For each snapshot date, compute what % of locations each status has
# This gives us the data for the line chart
trend_rows = []  # List to collect one row per date per status

for date in dates:  # Loop through each of the 24 snapshot dates
    snap = df[df["date"] == date]  # Filter to just this snapshot's rows
    total = len(snap)  # Total locations in this snapshot
    for status in STATUS_ORDER:  # Loop through each control status
        count = (
            snap["control_status"] == status
        ).sum()  # Count locations with this status
        pct = 100 * count / total  # Convert to percentage
        trend_rows.append(
            {  # Add a row to our trend data
                "date": date,  # Snapshot date
                "status": status,  # Control status
                "count": count,  # Raw count
                "pct": pct,  # Percentage of total
            }
        )

trend_df = pd.DataFrame(trend_rows)  # Convert list of dicts to DataFrame
print(f"✅ Trend data computed: {len(trend_df)} rows")  # Confirm

# ── 5. FIGURE A: TIME-SERIES TREND CHART ─────────────────────

print("\n[A] Generating trend time-series chart...")  # Announce this figure

fig_trend, ax_trend = plt.subplots(  # Create figure and single axes
    figsize=(14, 6),  # Wide format (landscape) for time series
    facecolor="#F8F6F0",  # Warm off-white background
)
ax_trend.set_facecolor("#F8F6F0")  # Match plot area background to figure

for status in STATUS_ORDER:  # Plot one line per control status
    sub = trend_df[trend_df["status"] == status]  # Filter trend data to this status
    ax_trend.plot(  # Draw the line
        sub["date"],  # X axis: dates
        sub["pct"],  # Y axis: percentage of locations
        color=COLOUR_MAP[status],  # Line colour for this status
        linewidth=2.5,  # Line thickness
        marker="o",  # Circle markers at each data point
        markersize=5,  # Marker size
        label=DISPLAY_NAMES[status],  # Legend label
        zorder=3,  # Draw lines above grid
    )

# Shade the area under the "contested" line to make its growth visually striking
contested_sub = trend_df[
    trend_df["status"] == "contested"
]  # Filter to contested data only
ax_trend.fill_between(  # Fill area between line and x-axis
    contested_sub["date"],  # X values (dates)
    contested_sub["pct"],  # Y values (percentages)
    alpha=0.15,  # Very transparent fill
    color=COLOUR_MAP["contested"],  # Use contested orange colour
    zorder=2,  # Behind lines but above background
)

# Add a vertical reference line for Operation 1027 (= the major offensive that started in Oct 2023)
op1027_date = pd.Timestamp("2023-10-27")  # Operation 1027 started on this date
ax_trend.axvline(  # axvline: draw a vertical line at a specific x value
    x=op1027_date,  # X position = Operation 1027 start date
    color="#880000",  # Dark red colour
    linewidth=1.5,  # Line thickness
    linestyle="--",  # Dashed line style
    alpha=0.7,  # Slightly transparent
    zorder=4,  # On top of everything
)

ax_trend.text(  # Add a text label next to the reference line
    pd.Timestamp("2023-12-05"),  # Shift label rightward into the data range
    trend_df["pct"].max() * 0.80,  # Position at 80% of the max y value
    "← Op. 1027\n  Oct 2023",  # Arrow points left toward the line
    fontsize=8,
    color="#880000",  # Match line colour
    va="top",  # Align text from top
)

ax_trend.set_title(  # Chart title
    "Myanmar Territorial Control — Monthly Evolution (Nov 2023 – Mar 2026)\n"
    "Source: Wikipedia Module revision history  |  n=352 tracked locations",
    fontsize=12,
    fontweight="bold",
    pad=12,
    color="#222222",
)
ax_trend.set_xlabel("Date", fontsize=10, color="#444444")  # X axis label
ax_trend.set_ylabel(
    "% of tracked locations", fontsize=10, color="#444444"
)  # Y axis label
ax_trend.legend(loc="upper right", fontsize=9, framealpha=0.9)  # Legend in top right
ax_trend.grid(True, alpha=0.3, color="#CCCCCC", linewidth=0.5)  # Subtle grid lines
ax_trend.tick_params(labelsize=9)  # Axis tick font size

plt.tight_layout()  # Trim whitespace
fig_trend.savefig(
    TREND_PATH, dpi=200, bbox_inches="tight", facecolor=fig_trend.get_facecolor()
)  # Save
plt.close(fig_trend)  # Free memory
print(f"   ✅ Saved: {TREND_PATH}")  # Confirm

# ── 6. FIGURE B: SMALL MULTIPLES GRID ────────────────────────

print("\n[B] Generating small multiples grid (24 panels)...")  # Announce

N_COLS = 6  # 6 columns in the grid
N_ROWS = 4  # 4 rows → 6×4 = 24 panels = one per snapshot

fig_grid, axes = plt.subplots(  # Create figure with a grid of subplots
    N_ROWS,
    N_COLS,  # 4 rows × 6 columns
    figsize=(20, 16),  # Large figure for 24 panels
    facecolor="#F0EEE8",  # Parchment background
)

axes_flat = (
    axes.flatten()
)  # flatten: convert 2D array of axes to 1D list for easy indexing

for i, date in enumerate(dates):  # Loop through each snapshot date with index
    ax = axes_flat[i]  # Get the axes panel for this snapshot
    snap = df[df["date"] == date]  # Filter data to this snapshot

    myanmar.plot(  # Draw Myanmar outline as background
        ax=ax, color="#E8E0D0", edgecolor="#AAAAAA", linewidth=0.3
    )

    for status in STATUS_ORDER:  # Plot each control status as a layer
        sub = snap[snap["control_status"] == status]  # Filter to this status
        if sub.empty:  # Skip if no locations with this status
            continue
        ax.scatter(  # Scatter plot (faster than geopandas for small panels)
            sub["lon"],
            sub["lat"],  # X=longitude, Y=latitude
            c=COLOUR_MAP[status],  # Colour for this status
            s=6,  # Small point size (panels are small)
            alpha=0.8,  # Slight transparency
            linewidths=0,  # No marker border
            zorder=5,  # On top
        )

    ax.set_xlim(92, 101)  # Fix longitude range to Myanmar's extent
    ax.set_ylim(9.5, 28.5)  # Fix latitude range to Myanmar's extent
    ax.set_aspect("equal")  # equal: don't distort the map shape
    ax.set_facecolor("#C8D8E8")  # Ocean blue background
    ax.set_xticks([])  # Remove x-axis tick marks (too small to read)
    ax.set_yticks([])  # Remove y-axis tick marks

    date_str = pd.Timestamp(date).strftime("%b %Y")  # Format date as "Nov 2023"
    contested_count = (
        snap["control_status"] == "contested"
    ).sum()  # Count contested locations
    ax.set_title(  # Panel title = date + contested count
        f"{date_str}\n({contested_count} contested)", fontsize=7, pad=2, color="#222222"
    )

for j in range(
    len(dates), len(axes_flat)
):  # Turn off any unused panels (if dates < 24)
    axes_flat[j].set_visible(False)  # Hide empty panel

# Shared legend below the grid
legend_patches = [  # Build legend colour patches
    mpatches.Patch(color=COLOUR_MAP[s], label=DISPLAY_NAMES[s]) for s in STATUS_ORDER
]
fig_grid.legend(  # Add shared legend to the whole figure (not one panel)
    handles=legend_patches,
    loc="lower center",  # Centre below all panels
    ncol=5,  # All 5 statuses in one row
    fontsize=10,
    title="Control Status",
    title_fontsize=11,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 0.01),  # Fine-tune position
)

fig_grid.suptitle(  # suptitle: title for the whole figure (above all panels)
    "Myanmar Territorial Control — 24 Monthly Snapshots (Nov 2023 – Mar 2026)\n"
    "Source: Wikipedia Module:Myanmar_Civil_War_detailed_map revision history",
    fontsize=13,
    fontweight="bold",
    y=0.99,
    color="#222222",
)

plt.subplots_adjust(hspace=0.35, wspace=0.05)  # Fine-tune spacing between panels
fig_grid.savefig(
    GRID_PATH, dpi=150, bbox_inches="tight", facecolor=fig_grid.get_facecolor()
)  # Save at 150 DPI (24 panels = large file)
plt.close(fig_grid)  # Free memory
print(f"   ✅ Saved: {GRID_PATH}")  # Confirm

# ── 7. FIGURE C: ANIMATED GIF ─────────────────────────────────

print("\n[C] Generating animated GIF (24 frames)...")  # Announce — this takes longest
print("    Building frames...")

fig_anim, ax_anim = plt.subplots(  # Create figure for the animation
    figsize=(8, 11),  # Tall portrait format for Myanmar's shape
    facecolor="#F0EEE8",  # Parchment background
)


def draw_frame(i):  # Function called once per frame; i = frame index (0–23)
    ax_anim.clear()  # clear: wipe the axes before drawing new frame
    ax_anim.set_facecolor("#C8D8E8")  # Reset ocean background (cleared above)

    date = dates[i]  # Get the date for this frame
    snap = df[df["date"] == date]  # Filter data to this snapshot

    myanmar.plot(  # Draw Myanmar outline as background
        ax=ax_anim, color="#E8E0D0", edgecolor="#888877", linewidth=0.4, alpha=0.9
    )

    for status in STATUS_ORDER:  # Plot each control status layer
        sub = snap[snap["control_status"] == status]  # Filter to this status
        if sub.empty:  # Skip empty groups
            continue
        ax_anim.scatter(  # Draw points for this status
            sub["lon"],
            sub["lat"],  # Coordinates
            c=COLOUR_MAP[status],  # Colour
            s=30,  # Point size (larger than small multiples)
            alpha=0.85,  # Transparency
            linewidths=0.3,  # Thin marker border
            edgecolors="white",  # White border for separation
            zorder=5,
            label=DISPLAY_NAMES[status],  # Legend label
        )

    # Summary bar in the corner: show count per status for this frame
    gov_pct = (
        100 * (snap["control_status"] == "government").sum() / len(snap)
    )  # % government
    cont_pct = (
        100 * (snap["control_status"] == "contested").sum() / len(snap)
    )  # % contested
    res_pct = (
        100 * (snap["control_status"] == "resistance").sum() / len(snap)
    )  # % resistance
    eao_pct = (
        100 * (snap["control_status"] == "ethnic_armed_group").sum() / len(snap)
    )  # % EAO

    summary = (  # Build a text summary string
        f"Gov/SAC:   {gov_pct:4.1f}%\n"
        f"Contested: {cont_pct:4.1f}%\n"
        f"Resistance:{res_pct:4.1f}%\n"
        f"EAO:       {eao_pct:4.1f}%"
    )
    ax_anim.text(  # Draw the summary text box
        0.02,
        0.28,
        summary,  # Position: lower-left area
        transform=ax_anim.transAxes,  # Use axes coordinates (0–1)
        fontsize=8,
        family="monospace",  # Monospace font for clean column alignment
        va="top",
        bbox=dict(  # Text box styling
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="#AAAAAA", alpha=0.9
        ),
    )

    date_label = pd.Timestamp(date).strftime("%B %Y")  # Format date as "November 2023"
    ax_anim.set_title(  # Frame title = current date
        f"Myanmar Territorial Control\n{date_label}",
        fontsize=13,
        fontweight="bold",
        color="#222222",
        pad=10,
    )

    ax_anim.set_xlim(91.5, 101.5)  # Fix longitude range
    ax_anim.set_ylim(9.0, 28.5)  # Fix latitude range
    ax_anim.set_xlabel("Longitude (°E)", fontsize=8, color="#666666")  # Axis labels
    ax_anim.set_ylabel("Latitude (°N)", fontsize=8, color="#666666")
    ax_anim.tick_params(labelsize=7)  # Small tick labels

    ax_anim.legend(  # Per-frame legend
        loc="lower left",
        fontsize=7,
        title="Control Status",
        title_fontsize=8,
        framealpha=0.9,
        markerscale=1.2,
    )

    frame_num = f"{i + 1}/{len(dates)}"  # Frame counter "1/24"
    ax_anim.text(  # Draw frame counter in top-right corner
        0.98,
        0.98,
        frame_num,
        transform=ax_anim.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        color="#666666",
    )

    # Progress bar across the top to show temporal position
    ax_anim.axhline(  # Draw a horizontal line at the very top of the plot
        y=28.3,  # Near the top latitude edge
        xmin=0,  # Start of line (0 = left edge)
        xmax=(i + 1) / len(dates),  # End = fraction of time elapsed
        color="#FF9900",  # Orange progress bar
        linewidth=4,  # Thick line
        zorder=10,  # On top of everything
        solid_capstyle="butt",  # Flat line ends
    )


ani = animation.FuncAnimation(  # Create the animation object
    fig_anim,  # The figure to animate
    draw_frame,  # The function called for each frame
    frames=len(dates),  # Total number of frames = number of snapshots
    interval=1200,  # interval=1200ms: each frame shows for 1.2 seconds
    repeat=True,  # Loop the animation when it reaches the end
)

print("    Saving GIF (this may take ~30 seconds)...")  # Warn about save time
ani.save(  # Save the animation to disk
    GIF_PATH,  # Output file path
    writer="pillow",  # Use pillow library to write GIF format
    fps=0.8,  # fps=0.8: ~1.2 seconds per frame (slow enough to read)
    dpi=100,  # dpi=100: lower than static plots to keep file size reasonable
)
plt.close(fig_anim)  # Free memory
print(f"   ✅ Saved: {GIF_PATH}")  # Confirm

# ── 8. PRINT FINAL SUMMARY ────────────────────────────────────

print("\n── Output Files ──")
print(f"   Trend chart:     {TREND_PATH}")  # List all saved files
print(f"   Small multiples: {GRID_PATH}")
print(f"   Animated GIF:    {GIF_PATH}")

# Print the key insight numbers for the World Bank presentation
first_snap = df[df["date"] == dates[0]]  # First snapshot (Nov 2023)
last_snap = df[df["date"] == dates[-1]]  # Last snapshot (Mar 2026)

gov_change = (last_snap["control_status"] == "government").mean() - (
    first_snap["control_status"] == "government"
).mean()
cont_change = (last_snap["control_status"] == "contested").mean() - (
    first_snap["control_status"] == "contested"
).mean()

print("\n── Key Findings for World Bank Presentation ──")
print(
    f"   Government control: {(first_snap['control_status'] == 'government').mean() * 100:.1f}% → {(last_snap['control_status'] == 'government').mean() * 100:.1f}%  (Δ {gov_change * 100:+.1f}pp)"
)
print(
    f"   Contested zones:    {(first_snap['control_status'] == 'contested').mean() * 100:.1f}% → {(last_snap['control_status'] == 'contested').mean() * 100:.1f}%  (Δ {cont_change * 100:+.1f}pp)"
)

print("\n" + "=" * 60)
print("STEP 6 COMPLETE.")
print("Open outputs:")
print(f"  open {TREND_PATH}")
print(f"  open {GRID_PATH}")
print(f"  open {GIF_PATH}")
print("=" * 60)
