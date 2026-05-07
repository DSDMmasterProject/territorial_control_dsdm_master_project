# ============================================================
# STEP 4: Map validation plot — visualise ground truth labels
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date: 07.05.2026
# ============================================================

# The script loads our 346 labelled locations and converts them to a GeoDataFrame
# a special table that understands geographic coordinates.
# It then searches the GADM folder for Myanmar boundary files,
# loading whichever it finds to draw the country outline as a pale landmass on a blue ocean background.
# Each of the five control statuses is plotted as a separate coloured layer in a deliberate order (unknown first, government last)
# so the most important points always appear on top; point size encodes confidence so a trained eye can instantly see
# which labels are solid versus uncertain. A legend, title, and caveat note about the 5-week date lag are added,
# then the whole figure is saved at 200 DPI — sharp enough for a World Bank presentation.

# ── IMPORTS ────────────────────────────────────────────────

import pathlib  # pathlib: clean file path handling

import geopandas as gpd  # geopandas: extends pandas to handle geographic data (shapefiles, GeoJSON)
import matplotlib.patches as mpatches  # mpatches: used to create custom legend items (coloured patches)
import matplotlib.pyplot as plt  # matplotlib: the core Python plotting library
import pandas as pd  # pandas: load and manipulate our CSV data

# ── 1. DEFINE PATHS ──────────────────────────────────────────

CSV_PATH = pathlib.Path(
    "data/raw/validation/myanmar_wikipedia_control_labels.csv"
)  # Our final Step 3 CSV
GADM_DIR = pathlib.Path("data/raw/gadm")  # Folder containing admin boundary files
OUTPUT_DIR = pathlib.Path("reports/figures")  # Where we save the map
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create output folder if missing
OUTPUT_PATH = (
    OUTPUT_DIR / "myanmar_wikipedia_ground_truth_map.png"
)  # Full path for the saved map image

# ── 2. DEFINE COLOUR SCHEME ───────────────────────────────────

# Each control status gets a distinct, intuitive colour
# These colours follow the Wikipedia map convention loosely
COLOUR_MAP = {  # Dictionary mapping control status → plot colour
    "government": "#CC3333",  # Red = Myanmar military (SAC/junta)
    "resistance": "#33AA44",  # Green = resistance / NUG-aligned forces
    "ethnic_armed_group": "#2266CC",  # Blue = ethnic armed organisations (KIO, AA, etc.)
    "contested": "#FF9900",  # Orange = contested areas (two actors fighting)
    "unknown": "#999999",  # Grey = genuinely ambiguous, low confidence
}

LABEL_MAP = {  # Nicer display names for the legend
    "government": "Government / SAC (219)",
    "ethnic_armed_group": "Ethnic Armed Group (41)",
    "contested": "Contested (40)",
    "resistance": "Resistance / NUG (34)",
    "unknown": "Unknown / Ambiguous (12)",
}

# ── 3. LOAD OUR GROUND TRUTH CSV ─────────────────────────────

print("=" * 60)  # Visual separator
print("STEP 4: Generating map validation plot")  # Announce what we're doing
print("=" * 60)  # Closing separator

df = pd.read_csv(
    CSV_PATH, encoding="utf-8"
)  # Load our final labelled CSV into a DataFrame
print(f"✅ Loaded {len(df)} ground truth points")  # Confirm row count

# ── 4. CONVERT TO GEODATAFRAME ────────────────────────────────

# A GeoDataFrame (= a pandas DataFrame that also understands geographic coordinates and projections)
# is needed to properly plot points on a map alongside shapefiles (= geographic boundary files)
gdf_points = gpd.GeoDataFrame(  # Create a GeoDataFrame from our regular DataFrame
    df,  # Our pandas DataFrame with lat/lon columns
    geometry=gpd.points_from_xy(
        df["lon"], df["lat"]
    ),  # points_from_xy: creates Point geometry from lon,lat columns (lon first!)
    crs="EPSG:4326",  # CRS (= Coordinate Reference System): EPSG:4326 is standard WGS84 lat/lon
)

# ── 5. LOAD MYANMAR ADMIN BOUNDARIES ─────────────────────────

# We look for Myanmar boundary files in the GADM directory
# GADM (= Global Administrative Areas database) provides free admin boundary shapefiles
# Level 0 = country outline, Level 1 = states/regions, Level 2 = districts

myanmar_boundary = None  # Will hold the boundary GeoDataFrame (or None if not found)

# Search for any shapefile or GeoJSON containing Myanmar in its name
possible_files = (
    list(GADM_DIR.rglob("*.shp"))
    + list(GADM_DIR.rglob("*.geojson"))
    + list(GADM_DIR.rglob("*.gpkg"))
)  # rglob: recursive search for files

myanmar_files = [  # Filter to files likely to be Myanmar admin boundaries
    f
    for f in possible_files  # For each file found
    if any(
        kw in f.name.lower() for kw in ["mmr", "myanmar", "gadm41_mmr"]
    )  # Keep if name contains Myanmar keywords
]

print(
    f"\n📁 GADM files found: {len(myanmar_files)}"
)  # Report how many Myanmar GADM files exist

if myanmar_files:  # If we found at least one boundary file
    myanmar_files_sorted = sorted(
        myanmar_files
    )  # Sort alphabetically (level 0 will come before level 1)
    print(f"   Using: {myanmar_files_sorted[0]}")  # Show which file we'll use
    try:  # try: loading a shapefile can fail if it's corrupted
        myanmar_boundary = gpd.read_file(
            myanmar_files_sorted[0], layer="ADM_ADM_0"
        )  # layer="ADM_ADM_0": explicitly load only the country outline (level 0), suppressing the multi-layer warning
        myanmar_boundary = myanmar_boundary.to_crs(
            "EPSG:4326"
        )  # to_crs: reproject to WGS84 if it's in a different system
        print(
            f"   ✅ Boundary loaded: {len(myanmar_boundary)} feature(s)"
        )  # Confirm successful load
    except Exception as e:  # Catch any file loading error
        print(f"   ⚠️ Could not load boundary file: {e}")  # Warn but don't crash
        myanmar_boundary = None  # Reset to None so we skip boundary plotting

if myanmar_boundary is None:  # If no boundary file was found or loaded
    print(
        "   ℹ️  No GADM boundary found — plotting points only (no outline)"
    )  # Inform user

# ── 6. CREATE THE FIGURE ─────────────────────────────────────

fig, ax = (
    plt.subplots(  # Create a matplotlib Figure (the whole image) and Axes (the plot area)
        figsize=(
            14,
            16,
        ),  # figsize: width=14 inches, height=16 inches (tall for Myanmar's shape)
        facecolor="#F0EEE8",  # facecolor: light parchment background colour for the figure border
    )
)

ax.set_facecolor(
    "#C8D8E8"
)  # Set the plot area background to a light blue (= ocean colour)

# ── 7. PLOT THE ADMIN BOUNDARIES ─────────────────────────────

if myanmar_boundary is not None:  # Only plot boundaries if we loaded them
    myanmar_boundary.plot(  # Plot the boundary polygons
        ax=ax,  # ax: the axes to draw on
        color="#E8E0D0",  # Fill colour for Myanmar landmass (warm grey = land)
        edgecolor="#888877",  # Colour of the boundary lines between regions
        linewidth=0.5,  # linewidth: thickness of boundary lines (thin = cleaner)
        alpha=0.8,  # alpha: opacity (0=transparent, 1=fully opaque)
    )
else:  # If no boundary file, draw a simple rectangle
    ax.set_xlim(92, 102)  # Set x-axis (longitude) range to cover Myanmar
    ax.set_ylim(9, 29)  # Set y-axis (latitude) range to cover Myanmar

# ── 8. PLOT CONTROL STATUS POINTS ────────────────────────────

# Plot each control status as a separate layer so the legend works correctly
# We plot in order from least to most important (unknown first, government last) so important points show on top

PLOT_ORDER = [
    "unknown",
    "contested",
    "resistance",
    "ethnic_armed_group",
    "government",
]  # Drawing order (last = on top)

for status in PLOT_ORDER:  # Loop through each control status in our chosen order
    subset = gdf_points[
        gdf_points["control_status"] == status
    ]  # Filter to only points with this control status
    if subset.empty:  # Skip if no points have this status
        continue

    # Point size encodes confidence: high=60, medium=40, low=25
    sizes = (
        subset["confidence"]
        .map(  # map: apply a function/dictionary to each value in the column
            {
                "high": 60,
                "medium": 40,
                "low": 25,
            }  # Dictionary: confidence level → marker size
        )
        .fillna(30)
    )  # fillna: replace any NaN (missing) values with size 30

    subset.plot(  # Plot this group of points on the map
        ax=ax,  # The axes to draw on
        color=COLOUR_MAP[status],  # Colour for this control status
        markersize=sizes,  # Size of each point marker
        alpha=0.85,  # Slight transparency so overlapping points are visible
        zorder=5,  # zorder: drawing layer (higher = drawn on top of lower)
        label=LABEL_MAP[status],  # Label used in the legend
    )

# ── 9. ADD TITLE AND LABELS ───────────────────────────────────

ax.set_title(  # Add a title to the map
    "Myanmar Territorial Control — Wikipedia Ground Truth Labels\n"  # Main title line
    f"Source: Module:Myanmar_Civil_War_detailed_map  |  As of: 2026-03-30  |  n = {len(df)} locations",  # Subtitle with metadata
    fontsize=13,  # Font size for the title
    fontweight="bold",  # Bold text for emphasis
    pad=15,  # pad: space between title and the plot area
    color="#222222",  # Dark grey text colour
)

ax.set_xlabel(
    "Longitude (°E)", fontsize=10, color="#444444"
)  # Label for the x-axis (longitude)
ax.set_ylabel(
    "Latitude (°N)", fontsize=10, color="#444444"
)  # Label for the y-axis (latitude)

# ── 10. ADD LEGEND ────────────────────────────────────────────

legend_patches = [  # Create a list of coloured rectangle patches for the legend
    mpatches.Patch(  # mpatches.Patch: a coloured rectangle used as a legend entry
        color=COLOUR_MAP[status],  # The colour for this status
        label=LABEL_MAP[status],  # The text label for this status
    )
    for status in [
        "government",
        "ethnic_armed_group",
        "contested",
        "resistance",
        "unknown",
    ]  # One patch per status
]

ax.legend(  # Add the legend to the plot
    handles=legend_patches,  # Use our custom coloured patches
    loc="lower left",  # Position the legend in the lower-left corner
    fontsize=9,  # Font size for legend text
    title="Control Status",  # Legend title
    title_fontsize=10,  # Font size for the legend title
    framealpha=0.9,  # Legend background opacity
    edgecolor="#AAAAAA",  # Legend border colour
)

# ── 11. ADD CONFIDENCE NOTE ───────────────────────────────────

ax.text(  # Add a text annotation to the plot
    0.98,
    0.02,  # Position: 2% from left, 2% from bottom (in axes coordinates)
    "Point size = confidence level\n(large=high, small=low)\n\n"
    "⚠ Labels dated 2026-03-30\nSVG map updated 2026-05-07\n(~5 week lag)",  # The text to display
    transform=ax.transAxes,  # transAxes: use axes coordinates (0–1) not data coordinates
    fontsize=8,  # Small font size for the note
    verticalalignment="bottom",  # Align text from the bottom of the bounding box
    horizontalalignment="right",  # Align text from the right edge of the bounding box
    bbox=dict(  # bbox: draw a box around the text
        boxstyle="round,pad=0.4",  # Rounded corners with padding
        facecolor="white",  # White background for readability
        edgecolor="#AAAAAA",  # Light grey border
        alpha=0.85,  # Slight transparency
    ),
)

# ── 12. GRID AND FORMATTING ───────────────────────────────────

ax.grid(  # Add grid lines to the map
    True,  # True = show grid
    alpha=0.3,  # Very transparent so they don't distract from data
    color="white",  # White grid lines
    linewidth=0.5,  # Thin grid lines
)

ax.tick_params(labelsize=8)  # Set font size for axis tick labels (lat/lon numbers)
plt.tight_layout()  # tight_layout: automatically adjust margins to prevent clipping

# ── 13. SAVE THE FIGURE ───────────────────────────────────────

plt.savefig(  # Save the figure to disk
    OUTPUT_PATH,  # The file path we defined at the top
    dpi=200,  # dpi: dots per inch — 200 gives a sharp, publication-quality image
    bbox_inches="tight",  # Crop to content — removes extra whitespace around edges
    facecolor=fig.get_facecolor(),  # Preserve the parchment background colour in the saved file
)

print(f"\n✅ Map saved to: {OUTPUT_PATH}")  # Confirm the save location

plt.close()  # Close the figure to free memory (good practice in scripts)

# ── 14. PRINT COORDINATE SANITY CHECK ────────────────────────

print("\n── Coordinate Sanity Check ──")  # Label this section
print(
    f"   Lat range: {df['lat'].min():.2f}°N → {df['lat'].max():.2f}°N"
)  # Show actual latitude range
print(
    f"   Lon range: {df['lon'].min():.2f}°E → {df['lon'].max():.2f}°E"
)  # Show actual longitude range
print(
    "   Myanmar expected: lat 9–29°N, lon 92–102°E"
)  # Show expected range for Myanmar

# Check how many points fall within Myanmar's approximate bounding box
in_bounds = (
    df[  # Filter DataFrame to rows where coordinates are inside Myanmar's bounding box
        (df["lat"] >= 9)
        & (df["lat"] <= 29)  # Latitude within Myanmar's north-south extent
        & (df["lon"] >= 92)
        & (df["lon"] <= 102)  # Longitude within Myanmar's east-west extent
    ]
)

print(
    f"   Points inside Myanmar bounds: {len(in_bounds)}/{len(df)} ({100 * len(in_bounds) / len(df):.1f}%)"
)  # Report % in bounds

print("\n" + "=" * 60)  # Visual separator
print("STEP 4 COMPLETE.")  # Announce completion
print(f"Open the map: open {OUTPUT_PATH}")  # macOS command to open the image
print("=" * 60)  # Closing separator
