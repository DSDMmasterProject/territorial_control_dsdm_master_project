# ============================================================
# STEP 3: Fetch revision timestamp → add date to our dataset
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date: 07.05.2026
# ============================================================

# The script makes two API calls to Wikipedia.
# The first fetches the recent revision history of the Lua module itself —
# specifically the timestamp of the last edit — which tells us when the control labels were last updated by Wikipedia's volunteer editors.
# The second call fetches the file version history of the main SVG map from Wikimedia Commons as supplementary reference.
# It then loads our Step 2 CSV, stamps every row with that revision date, adds a source column, reorders the columns
# to exactly match our project specification, and prints a formatted summary showing control status percentages as a mini bar chart.
# Both the final CSV and the SVG version history are saved to disk.

# Last insight:
# Date gap worth noting: Our module labels are dated 2026-03-30, but the SVG map was updated 2026-05-07 (today) with "April changes."
# This means our ground truth is ~5 weeks behind the current situation.
# This is a limitation to state explicitly — ground truth from Wikipedia has a known lag.

# ── IMPORT LIBRARIES ──────────────────────────────────────────

import json  # json: built-in library to parse API responses
import pathlib  # pathlib: clean file path handling
from datetime import (  # datetime: built-in library to parse and format timestamps
    datetime,
)

import pandas as pd  # pandas: DataFrame library for loading and saving our CSV
import requests  # requests: HTTP library to call the Wikipedia API

# ── 1. DEFINE PATHS ──────────────────────────────────────────

INPUT_CSV = pathlib.Path(
    "data/raw/validation/myanmar_wikipedia_parsed_step2.csv"
)  # Our Step 2 output
OUTPUT_CSV = pathlib.Path(
    "data/raw/validation/myanmar_wikipedia_control_labels.csv"
)  # Final output path (matches the spec)
OUTPUT_DIR = OUTPUT_CSV.parent  # Parent folder of the output file
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

# ── 2. CONFIGURE API HEADERS ─────────────────────────────────

HEADERS = {
    "User-Agent": (  # Wikipedia requires a User-Agent identifying who is calling the API
        "TizianSchenk-BSEThesis/1.0 "  # Script name and version
        "(Barcelona School of Economics; "  # Institution name
        "World Bank FCV Territorial Control Project; "  # Project description
        "contact: your_email@bse.eu)"  # Replace with your actual BSE email
    )
}

# ── 3. DEFINE BOTH API CALLS ─────────────────────────────────

# API Call A: Get the revision timestamp of the Lua MODULE itself
# This tells us when the control labels were last updated by Wikipedia editors
MODULE_TIMESTAMP_URL = (
    "https://en.wikipedia.org/w/api.php"  # Wikipedia API base endpoint
    "?action=query"  # We want to query a page
    "&titles=Module:Myanmar_Civil_War_detailed_map"  # The Lua module page
    "&prop=revisions"  # We want revision information
    "&rvprop=timestamp|ids|comment"  # Fetch: timestamp, revision ID, and edit comment
    "&rvlimit=5"  # Return the 5 most recent revisions (not just the latest)
    "&format=json"  # Return as JSON
    "&formatversion=2"  # Use cleaner v2 JSON format
)

# API Call B: Get the file history of the main SVG map on Wikimedia Commons
# This is the visual map that the module powers — its history shows territorial change over time
SVG_HISTORY_URL = (
    "https://commons.wikimedia.org/w/api.php"  # Wikimedia Commons API (different from Wikipedia API)
    "?action=query"  # Query action
    "&titles=File:Myanmar_civil_war.svg"  # The main SVG map file
    "&prop=imageinfo"  # We want image file information (not page text)
    "&iiprop=timestamp|url|comment"  # Fetch: timestamp, file URL, and edit comment for each version
    "&iilimit=10"  # Return the 10 most recent file versions
    "&format=json"  # Return as JSON
    "&formatversion=2"  # Use cleaner v2 JSON format
)

# ── 4. FETCH MODULE REVISION TIMESTAMP ───────────────────────

print("=" * 60)  # Visual separator
print("STEP 3: Fetching timestamps from Wikipedia API")  # Announce what we're doing
print("=" * 60)  # Closing separator

print(
    "\n[A] Fetching Lua module revision history..."
)  # Label which API call we're making

try:  # try: attempt the request; jump to except if it fails
    resp_module = requests.get(  # Send HTTP GET request to Wikipedia API
        MODULE_TIMESTAMP_URL,  # The URL we defined above
        headers=HEADERS,  # Include polite User-Agent header
        timeout=30,  # Give up after 30 seconds of no response
    )
    resp_module.raise_for_status()  # Raise an error if server returned a bad status code (e.g., 404)
except requests.exceptions.RequestException as e:  # Catch any network-related error
    print(f"ERROR fetching module timestamp: {e}")  # Print the error message
    raise  # Re-raise: stop execution completely (we need this data)

module_data = resp_module.json()  # Parse the JSON response into a Python dictionary

# Navigate the nested JSON structure to get the revisions list
# Structure: data → "query" → "pages" → [0] → "revisions" → [list of revision objects]
module_revisions = module_data["query"]["pages"][0][
    "revisions"
]  # Extract the list of recent revisions

print(
    f"   ✅ Found {len(module_revisions)} recent revision(s)"
)  # Report how many revisions were returned

print("\n   Recent edits to the Lua module:")  # Label the revision preview
for rev in module_revisions:  # Loop through each revision object
    ts = rev.get(
        "timestamp", "unknown"
    )  # Get the timestamp field (default "unknown" if missing)
    rid = rev.get("revid", "?")  # Get the revision ID number
    cmt = rev.get(
        "comment", "(no comment)"
    )  # Get the editor's comment (why they made the change)
    print(
        f"   Rev {rid}: {ts}  →  {cmt[:80]}"
    )  # Print revision ID, timestamp, and first 80 chars of comment

# Extract the timestamp of the MOST RECENT revision (index 0 = most recent)
raw_module_timestamp = module_revisions[0][
    "timestamp"
]  # Get the timestamp string, e.g., "2025-03-14T18:22:01Z"

# Parse the timestamp string into a Python datetime object (= a structured date-and-time value)
# The "Z" at the end means UTC (= Coordinated Universal Time, the global time standard)
module_datetime = (
    datetime.fromisoformat(  # fromisoformat: convert ISO 8601 string to datetime object
        raw_module_timestamp.replace(
            "Z", "+00:00"
        )  # Replace "Z" with "+00:00" because Python needs explicit timezone offset
    )
)

module_date_str = module_datetime.strftime(
    "%Y-%m-%d"
)  # strftime: format datetime as "YYYY-MM-DD" string for our CSV

print(f"\n   📅 Module last updated: {raw_module_timestamp}")  # Show the full timestamp
print(f"   📅 Date we will use:     {module_date_str}")  # Show the simplified date

# ── 5. FETCH SVG MAP FILE HISTORY ────────────────────────────

print(
    "\n[B] Fetching SVG map file history from Wikimedia Commons..."
)  # Label this API call

try:  # try: attempt the request
    resp_svg = requests.get(  # Send HTTP GET request to Wikimedia Commons API
        SVG_HISTORY_URL,  # The URL for the SVG file history
        headers=HEADERS,  # Include User-Agent header
        timeout=30,  # 30 second timeout
    )
    resp_svg.raise_for_status()  # Raise error on bad HTTP status
except requests.exceptions.RequestException as e:  # Catch network errors
    print(
        f"   ⚠️ Could not fetch SVG history: {e}"
    )  # Warn but don't crash (this is supplementary info)
    svg_versions = []  # Set to empty list so code below still works
else:  # else: runs only if NO exception was raised
    svg_data = resp_svg.json()  # Parse JSON response
    # Navigate to the imageinfo list inside the JSON
    # Structure: data → "query" → "pages" → [0] → "imageinfo" → [list of file versions]
    svg_page = svg_data["query"]["pages"][0]  # Get the first (only) page result
    svg_versions = svg_page.get(
        "imageinfo", []
    )  # Get imageinfo list; default to empty list if missing
    print(f"   ✅ Found {len(svg_versions)} recent SVG version(s)")  # Report count

    print("\n   Recent SVG map versions:")  # Label the version preview
    for ver in svg_versions[:5]:  # Loop through just the first 5 versions
        ts = ver.get("timestamp", "unknown")  # Get version timestamp
        cmt = ver.get("comment", "(no comment)")  # Get upload comment
        url = ver.get("url", "")[:60]  # Get file URL, truncated to 60 chars for display
        print(f"   {ts}  →  {cmt[:60]}")  # Print timestamp and comment

# ── 6. LOAD OUR STEP 2 CSV AND ADD DATE COLUMN ───────────────

print("\n[C] Loading Step 2 CSV and adding date column...")  # Announce this step

df = pd.read_csv(
    INPUT_CSV, encoding="utf-8"
)  # Load the CSV we saved in Step 2 into a DataFrame

print(
    f"   Loaded {len(df)} rows from {INPUT_CSV}"
)  # Confirm row count matches what we expect

df["date"] = (
    module_date_str  # Add a new "date" column with the same timestamp for all rows
)
df["source"] = (
    "wikipedia_module"  # Add a "source" column to identify where this data came from
)

# ── 7. REORDER COLUMNS TO MATCH THE SPEC ─────────────────────

# The project spec requires these columns in this order:
# date | lat | lon | location_name | control_status | actor | source | confidence
FINAL_COLUMNS = [  # Define the exact column order we want
    "date",  # When this snapshot was recorded
    "lat",  # Latitude (decimal degrees)
    "lon",  # Longitude (decimal degrees)
    "location_name",  # Township or place name
    "control_status",  # government / resistance / contested / ethnic_armed_group / unknown
    "actor",  # Specific armed actor
    "source",  # Where this data came from
    "confidence",  # high / medium / low
]

# Check that all expected columns exist in our DataFrame
missing_cols = [
    c for c in FINAL_COLUMNS if c not in df.columns
]  # List comprehension: find columns in spec but not in df
if missing_cols:  # If any expected columns are missing
    print(f"   ⚠️ Missing columns: {missing_cols}")  # Warn the user
else:  # If all columns are present
    print("   ✅ All required columns present")  # Confirm everything is good

df_final = df[FINAL_COLUMNS]  # Reorder (and select) only the columns we want

# ── 8. PRINT FINAL SUMMARY STATISTICS ────────────────────────

print("\n── Final Dataset Summary ──")  # Label the summary section
print(f"   Total entries:      {len(df_final)}")  # Total number of labeled locations
print(f"   Date:               {module_date_str}")  # The snapshot date
print(
    f"   Coordinate range:   lat [{df_final['lat'].min():.2f}, {df_final['lat'].max():.2f}]"
)  # Latitude range
print(
    f"                       lon [{df_final['lon'].min():.2f}, {df_final['lon'].max():.2f}]"
)  # Longitude range

print("\n   Control Status Breakdown:")  # Label control status section
counts = df_final["control_status"].value_counts()  # Count entries per control status
for status, count in counts.items():  # Loop through each status and its count
    pct = 100 * count / len(df_final)  # Calculate percentage of total
    bar = "█" * int(pct / 2)  # Create a simple text bar chart (each █ = 2%)
    print(
        f"   {status:<20} {count:>4}  ({pct:4.1f}%)  {bar}"
    )  # Print status, count, percentage, and bar

print("\n   Confidence Breakdown:")  # Label confidence section
for conf, count in (
    df_final["confidence"].value_counts().items()
):  # Loop through confidence levels
    print(f"   {conf:<10} {count:>4} entries")  # Print confidence level and count

# ── 9. SAVE THE FINAL CSV ─────────────────────────────────────

df_final.to_csv(
    OUTPUT_CSV, index=False, encoding="utf-8"
)  # Save final DataFrame to CSV; index=False = no row numbers

print(f"\n✅ Final ground truth CSV saved to: {OUTPUT_CSV}")  # Confirm save location

# ── 10. SAVE SVG VERSION HISTORY AS REFERENCE ─────────────────

if svg_versions:  # Only save if we successfully fetched SVG history
    svg_history_path = (
        OUTPUT_DIR / "myanmar_svg_version_history.json"
    )  # Define path for SVG history file
    with open(svg_history_path, "w", encoding="utf-8") as f:  # Open file for writing
        json.dump(
            svg_versions, f, indent=2, ensure_ascii=False
        )  # Write JSON with indentation for readability
    print(f"✅ SVG version history saved to: {svg_history_path}")  # Confirm

print("\n" + "=" * 60)  # Visual separator
print("STEP 3 COMPLETE.")  # Announce completion
print("Next: Step 4 = map validation plot")  # Preview next step
print("=" * 60)  # Closing separator
