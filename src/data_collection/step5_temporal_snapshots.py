# ============================================================
# STEP 5: Build temporal ground truth from module revision history
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date: 07.05.2026
# ============================================================

# The script first fetches all revision IDs and timestamps of the Lua module in a paginated loop
# getting potentially hundreds of edit records cheaply without downloading their content.
# It then groups revisions by calendar month and keeps only the last edit of each month,
# giving one clean representative snapshot per month.
# For each selected month, it fetches the full Lua content of that specific revision by ID,
# runs the same parser we built in Step 2,
# and collects every location record into a master list tagged with that month's date.
# The result is a stacked CSV where each row is a location-at-a-point-in-time observation
# transforming our single 346-point snapshot into a temporal panel dataset potentially spanning 2021–2026.

# Imports:

import json  # json: parse API responses
import pathlib  # pathlib: file path handling
import re  # re: regex pattern matching (reuse our Step 2 parser logic)
import time  # time: add pauses between API calls to be polite to Wikipedia's servers
from datetime import datetime  # datetime: parse and compare timestamps

import pandas as pd  # pandas: data manipulation
import requests  # requests: HTTP library for Wikipedia API calls

# ── 1. PATHS AND CONFIG ──────────────────────────────────────

OUTPUT_DIR = pathlib.Path(
    "data/raw/validation/temporal"
)  # Separate folder for temporal snapshots
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create it if it doesn't exist

FINAL_CSV = pathlib.Path(
    "data/raw/validation/myanmar_wikipedia_temporal_groundtruth.csv"
)  # Final combined output

HEADERS = {
    "User-Agent": (
        "TizianSchenk-BSEThesis/1.0 "
        "(Barcelona School of Economics; World Bank FCV; contact: your_email@bse.eu)"
    )
}

API_DELAY = 1.5  # Seconds to wait between API calls — Wikipedia requests politeness (rate limiting)

# ── 2. REUSE OUR MARK MAPPING FROM STEP 2 ────────────────────

# This is the same dictionary we built in Step 2 — pasted here so this script is self-contained
MARK_MAPPING = {  # Maps Wikipedia icon filename → (control_status, actor, confidence)
    "Location dot red.svg": ("government", "Military Forces of Myanmar (SAC)", "high"),
    "Location dot blue.svg": (
        "ethnic_armed_group",
        "Kachin Independence Army (KIO/KIA)",
        "high",
    ),
    "Location dot green.svg": ("resistance", "Arakan Army / NUG-aligned", "high"),
    "Location dot orange.svg": (
        "ethnic_armed_group",
        "Chin ethnic armed groups (CDF/CNF)",
        "high",
    ),
    "Location dot magenta.svg": ("resistance", "Chin PDF / resistance forces", "high"),
    "Location dot cyan.svg": ("ethnic_armed_group", "Arakan Army (AA)", "high"),
    "Location dot teal.svg": ("ethnic_armed_group", "Arakan Army (AA)", "high"),
    "Location dot limegreen.svg": ("resistance", "NUG/PDF resistance forces", "high"),
    "Location dot purple.svg": (
        "ethnic_armed_group",
        "United Wa State Army (UWSA)",
        "medium",
    ),
    "Location dot yellow.svg": ("contested", "Unknown actors", "low"),
    "Location dot black.svg": ("contested", "Unknown/symbolic", "low"),
    "Location dot white.svg": ("contested", "Unknown/symbolic", "low"),
    "Location dot coral.svg": (
        "ethnic_armed_group",
        "EAO allied with SAC (unconfirmed)",
        "low",
    ),
    "Location dot ochre200.png": (
        "ethnic_armed_group",
        "Ethnic armed group (unspecified)",
        "low",
    ),
    "Location dot darkslategray.svg": ("unknown", "Actor unidentified", "low"),
    "Location dot lightgrey.svg": ("unknown", "Actor unidentified", "low"),
    "Location dot red-black.svg": ("contested", "SAC vs unknown armed group", "medium"),
    "Dot green 0d0.svg": ("resistance", "NUG/PDF resistance forces", "high"),
    "Dot yellow ff4.svg": ("contested", "Unclear/contested situation", "low"),
    "80x80-red-blue-anim.gif": ("contested", "SAC vs KIO/KIA", "high"),
    "80x80-red-lime-anim.gif": ("contested", "SAC vs NUG/PDF", "high"),
    "80x80-red-magenta-anim.gif": ("contested", "SAC vs Chin PDF", "high"),
    "80x80-red-cyan-anim.gif": ("contested", "SAC vs Arakan Army (AA)", "high"),
    "80x80-red-black-anim.gif": ("contested", "SAC vs unknown armed group", "low"),
    "80x80-red-orange-anim.gif": ("contested", "SAC vs ethnic armed group", "high"),
    "80x80-red-green-anim.gif": ("contested", "SAC vs Arakan Army", "high"),
    "80x80-blue-lime-anim.gif": ("contested", "KIO/KIA vs NUG/PDF", "high"),
    "Map-dot-grey-68a.svg": ("unknown", "Actor unidentified", "low"),
}

SKIP_MARKS = {  # Icons that are NOT control status markers — skip these
    "Myanmar Roadmap.png",
    "Fighter-jet-blue-icon.svg",
    "Fighter-jet-green-icon.svg",
    "Map-circle-blue.svg",
    "Map-circle-green.svg",
    "Map-circle-yellow.svg",
    "Map-arcNE-green.svg",
    "WhiteDot.svg",
    "Anchor pictogram green.svg",
    "Mountain pass 12x12 e.svg",
    "Mountain pass 12x12 n.svg",
    "Mountain pass 12x12 ne.svg",
    "Mountain pass 12x12 nw.svg",
    "Mountain pass 12x12 se.svg",
}

# ── 3. DEFINE THE PARSER FUNCTION (reusable) ─────────────────

# We wrap our Step 2 parsing logic into a function so we can call it for each revision
FULL_ENTRY_PATTERN = re.compile(  # Regex: match every { } block containing lat=
    r'\{([^}]*lat\s*=\s*"[^"]*"[^}]*)\}', re.DOTALL
)
LABEL_PATTERN = re.compile(  # Regex: extract place name label from a block
    r'label\s*=\s*"?\[?\[?([^\]|"]+)'
)


def parse_lua_to_records(
    lua_text, snapshot_date
):  # Function takes raw Lua text + a date string
    """Parse a Lua module snapshot and return a list of location record dicts."""
    blocks = FULL_ENTRY_PATTERN.findall(
        lua_text
    )  # Find all { } entry blocks containing lat=
    records = []  # Empty list to collect parsed records

    for block in blocks:  # Loop through each matched entry block
        lat_m = re.search(r'lat\s*=\s*"([^"]+)"', block)  # Find lat value
        lon_m = re.search(r'long\s*=\s*"([^"]+)"', block)  # Find lon value
        mark_m = re.search(r'mark\s*=\s*"([^"]+)"', block)  # Find mark/icon value

        if not (lat_m and lon_m and mark_m):  # Skip incomplete entries
            continue

        mark = mark_m.group(1).strip()  # Extract and clean mark filename
        if mark in SKIP_MARKS:  # Skip non-control icons
            continue

        try:  # Attempt coordinate conversion
            lat = float(lat_m.group(1))  # Latitude as float
            lon = float(lon_m.group(1))  # Longitude as float
        except ValueError:  # Skip if not a valid number
            continue

        label_m = LABEL_PATTERN.search(block)  # Find the place name label
        name = ""  # Default to empty
        if label_m:  # If label found
            name = label_m.group(1).strip()  # Extract and clean it
            if "|" in name:  # Handle Wikipedia pipe syntax
                name = name.split("|")[-1]  # Keep only the display name after |

        if mark not in MARK_MAPPING:  # Handle unrecognised marks
            status, actor, conf = "unknown", f"UNMAPPED:{mark}", "low"
        else:  # Known marks
            status, actor, conf = MARK_MAPPING[mark]  # Unpack from mapping

        records.append(
            {  # Add structured record to list
                "date": snapshot_date,  # The date of this revision
                "lat": lat,  # Latitude
                "lon": lon,  # Longitude
                "location_name": name,  # Place name
                "control_status": status,  # Territorial control label
                "actor": actor,  # Specific actor
                "source": "wikipedia_module",  # Data provenance
                "confidence": conf,  # Label quality
            }
        )

    return records  # Return all parsed records for this snapshot


# ── 4. FETCH ALL REVISION IDs AND TIMESTAMPS ─────────────────

print("=" * 60)
print("STEP 5: Building temporal ground truth from revision history")
print("=" * 60)

print(
    "\n[A] Fetching all revision timestamps from Wikipedia API..."
)  # Announce this sub-step

all_revisions = []  # Will hold all revision metadata dicts
rvcontinue = None  # Wikipedia paginates results; this tracks where we are

while True:  # Loop until we've fetched all pages of results
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        "&titles=Module:Myanmar_Civil_War_detailed_map"
        "&prop=revisions"
        "&rvprop=timestamp|ids|comment"  # Just timestamps and IDs (no content) — fast and lightweight
        "&rvlimit=500"  # rvlimit=500: maximum allowed per request
        "&format=json"
        "&formatversion=2"
    )

    if rvcontinue:  # If we have a pagination cursor from a previous request
        url += f"&rvcontinue={rvcontinue}"  # Append it to get the next page of results

    resp = requests.get(url, headers=HEADERS, timeout=30)  # Make the API request
    resp.raise_for_status()  # Raise error on bad HTTP status
    data = resp.json()  # Parse JSON response

    revisions = data["query"]["pages"][0][
        "revisions"
    ]  # Extract revision list from response
    all_revisions.extend(revisions)  # Add this page's revisions to our full list
    print(f"   Fetched {len(all_revisions)} revisions so far...")  # Show running total

    if "continue" in data:  # If Wikipedia says there are more pages
        rvcontinue = data["continue"][
            "rvcontinue"
        ]  # Save the cursor for the next request
        time.sleep(API_DELAY)  # Wait politely before the next request
    else:  # If no more pages
        break  # Exit the loop

print(f"\n✅ Total revisions found: {len(all_revisions)}")  # Report grand total

# ── 5. SELECT ONE REVISION PER MONTH ─────────────────────────

# We don't need every single revision — one per month is enough for validation
# Strategy: for each calendar month, take the LAST revision of that month (most up-to-date)

print(
    "\n[B] Selecting one representative revision per month..."
)  # Announce this sub-step

# Convert timestamps to datetime objects and attach to each revision
for rev in all_revisions:  # Loop through every revision
    rev["dt"] = datetime.fromisoformat(  # Parse ISO timestamp string to datetime object
        rev["timestamp"].replace("Z", "+00:00")  # Replace Z with explicit UTC offset
    )

# Sort all revisions chronologically (oldest first)
all_revisions.sort(
    key=lambda r: r["dt"]
)  # sort by datetime; lambda = anonymous function returning dt field

# Group by year-month and keep the last revision of each month
monthly = {}  # Dict mapping "YYYY-MM" → revision dict
for rev in all_revisions:  # Loop through sorted revisions
    ym = rev["dt"].strftime("%Y-%m")  # Format as "YYYY-MM" year-month key
    monthly[ym] = rev  # Overwrite — so we keep the LAST revision of each month

selected_revisions = sorted(
    monthly.values(), key=lambda r: r["dt"]
)  # Sort selected revisions chronologically

print(
    f"   Selected {len(selected_revisions)} monthly snapshots"
)  # Report how many we'll process
print(
    f"   Date range: {selected_revisions[0]['dt'].date()} → {selected_revisions[-1]['dt'].date()}"
)  # Show range

print("\n   Monthly snapshots selected:")  # Preview the selected revisions
for rev in selected_revisions:  # Loop through selected
    print(
        f"   {rev['dt'].strftime('%Y-%m-%d')}  revid={rev['revid']}  {rev.get('comment', '')[:50]}"
    )  # Print details

# ── 6. FETCH CONTENT FOR EACH SELECTED REVISION ──────────────

print(
    f"\n[C] Fetching content for {len(selected_revisions)} revisions..."
)  # Announce content fetching
print("    (This may take a few minutes — pausing between requests)")  # Warn about time

all_records = []  # Master list to hold all parsed records across all snapshots
failed_revids = []  # Track any revisions we couldn't fetch

for i, rev in enumerate(
    selected_revisions
):  # Loop through each selected revision with index
    revid = rev["revid"]  # The unique integer ID of this revision
    snap_date = rev["dt"].strftime("%Y-%m-%d")  # Date string for this snapshot

    print(
        f"   [{i + 1:>2}/{len(selected_revisions)}] {snap_date} (revid={revid})...",
        end=" ",
    )  # Progress indicator

    content_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        f"&revids={revid}"  # revids: fetch a specific revision by ID (not by title)
        "&prop=revisions"
        "&rvprop=content"  # This time we DO want the full content
        "&rvslots=main"  # Main content slot (where Lua code lives)
        "&format=json"
        "&formatversion=2"
    )

    try:  # Attempt to fetch this revision
        resp = requests.get(
            content_url, headers=HEADERS, timeout=45
        )  # Longer timeout for large content
        resp.raise_for_status()  # Error on bad status
        data = resp.json()  # Parse JSON

        page = data["query"]["pages"][0]  # Get page result
        lua_text = page["revisions"][0]["slots"]["main"][
            "content"
        ]  # Extract raw Lua source text

        records = parse_lua_to_records(
            lua_text, snap_date
        )  # Parse this revision with our function
        all_records.extend(records)  # Add all records to master list
        print(
            f"→ {len(records)} locations parsed"
        )  # Show how many locations this snapshot has

    except Exception as e:  # Catch any error (network, parsing, etc.)
        print(f"→ FAILED: {e}")  # Log the failure
        failed_revids.append(revid)  # Track which revision failed

    time.sleep(API_DELAY)  # Wait between requests (important — don't spam Wikipedia)

# ── 7. BUILD AND SAVE THE TEMPORAL DATAFRAME ─────────────────

print("\n[D] Building temporal ground truth dataset...")  # Announce final assembly

df_temporal = pd.DataFrame(all_records)  # Convert master list to DataFrame

print("\n── Temporal Dataset Summary ──")  # Label summary section
print(
    f"   Total labeled observations:  {len(df_temporal):,}"
)  # Total rows across all snapshots
print(
    f"   Unique snapshots (months):   {df_temporal['date'].nunique()}"
)  # How many distinct dates
print(
    f"   Unique locations:            {df_temporal[['lat', 'lon']].drop_duplicates().shape[0]}"
)  # Unique locations
print(
    f"   Date range:                  {df_temporal['date'].min()} → {df_temporal['date'].max()}"
)  # Time span

print("\n   Control status distribution (across all snapshots):")  # Label distribution
for status, count in (
    df_temporal["control_status"].value_counts().items()
):  # Loop through statuses
    pct = 100 * count / len(df_temporal)  # Calculate percentage
    print(
        f"   {status:<22} {count:>6,}  ({pct:.1f}%)"
    )  # Print status, count, percentage

if failed_revids:  # Only print if some revisions failed
    print(
        f"\n⚠️  Failed to fetch {len(failed_revids)} revisions: {failed_revids}"
    )  # List failed IDs

df_temporal.to_csv(FINAL_CSV, index=False, encoding="utf-8")  # Save to CSV
print(f"\n✅ Temporal ground truth saved to: {FINAL_CSV}")  # Confirm

# ── 8. SAVE REVISION METADATA ─────────────────────────────────

meta_path = pathlib.Path(
    "data/raw/validation/myanmar_module_revision_metadata.json"
)  # Path for metadata file
with open(meta_path, "w", encoding="utf-8") as f:  # Open file for writing
    json.dump(  # Write JSON
        [
            {
                "revid": r["revid"],
                "timestamp": r["timestamp"],  # Save key fields for each revision
                "comment": r.get("comment", ""),
            }
            for r in selected_revisions
        ],
        f,
        indent=2,  # indent=2: pretty-print for human readability
    )
print(f"✅ Revision metadata saved to: {meta_path}")  # Confirm

print("\n" + "=" * 60)
print("STEP 5 COMPLETE.")
print("You now have a temporal validation dataset.")
print(f"Failed revisions: {len(failed_revids)}")
print("=" * 60)
