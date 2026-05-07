# ============================================================
# STEP 2: Parse the Lua module → extract all location entries
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date:    07.05.2026
# ============================================================

# The script opens by loading the raw Lua text we saved in Step 1,
# then defines a lookup table mapping every Wikipedia icon filename to a territorial control interpretation.
# It compiles a regex pattern that hunts through the text for every { ... } block containing lat=, long=, and mark= fields,
# extracting all three values simultaneously. For each match, it checks the icon against our lookup table — skipping airport and
# road markers, flagging any unrecognised icons as "unknown" for manual review, and assigning control status, actor name,
# and confidence for everything recognised.
# A second inner loop also extracts the place name (handling Wikipedia's [[Name|DisplayName]] link syntax
# by keeping only the display part).
# Everything is assembled into a pandas DataFrame, summarised with counts per control status, and saved to a CSV.

# ── IMPORTANT LIBRARIES ─────────────────────────────────────────

import pathlib  # pathlib: built-in library for clean file path handling
import re  # re: built-in Python library for regex (= pattern matching in text)

import pandas as pd  # pandas: the standard data manipulation library (DataFrames = like Excel tables in Python)

# ── 1. DEFINE FILE PATHS ─────────────────────────────────────

INPUT_PATH = pathlib.Path(
    "data/raw/validation/myanmar_module_raw.lua"
)  # Where we saved the raw Lua text in Step 1
OUTPUT_DIR = pathlib.Path(
    "data/raw/validation"
)  # Directory where we'll save our parsed CSV
OUTPUT_DIR.mkdir(
    parents=True, exist_ok=True
)  # Create directory if it doesn't already exist

# ── 2. LOAD THE RAW LUA TEXT ─────────────────────────────────

print("=" * 60)  # Visual separator for readable terminal output
print("STEP 2: Parsing Lua module for location entries")  # Announce what we're doing
print("=" * 60)  # Closing separator line

lua_text = INPUT_PATH.read_text(
    encoding="utf-8"
)  # Load the full Lua source text from disk into a string
print(
    f"Loaded {len(lua_text):,} characters from {INPUT_PATH}\n"
)  # Confirm file was loaded and show its size

# ── 3. DEFINE THE COLOUR → CONTROL STATUS MAPPING ───────────

# This dictionary (= a lookup table: key → value) maps each Wikipedia icon filename
# to a tuple of (control_status, actor, confidence)
# We skip non-location icons (airports, roads, mountain passes) by not including them
MARK_MAPPING = {
    # ── Solid dot icons — clear single-actor control ──────────
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
    "Location dot cyan.svg": (
        "ethnic_armed_group",
        "Arakan Army (AA)",
        "high",
    ),  # AA uses cyan as their official colour
    "Location dot teal.svg": (
        "ethnic_armed_group",
        "Arakan Army (AA)",
        "high",
    ),  # Teal ≈ cyan, same actor
    "Location dot limegreen.svg": (
        "resistance",
        "NUG/PDF resistance forces",
        "high",
    ),  # Lime green = brighter NUG green
    "Location dot purple.svg": (
        "ethnic_armed_group",
        "United Wa State Army (UWSA)",
        "medium",
    ),  # UWSA controls NE Shan State
    "Location dot yellow.svg": ("contested", "Unknown actors", "low"),
    "Location dot black.svg": ("contested", "Unknown/symbolic", "low"),
    "Location dot white.svg": ("contested", "Unknown/symbolic", "low"),
    "Location dot coral.svg": (
        "ethnic_armed_group",
        "EAO allied with SAC (unconfirmed)",
        "low",
    ),  # Coral ≈ orange-red, junta-allied EAO
    "Location dot ochre200.png": (
        "ethnic_armed_group",
        "Ethnic armed group (unspecified)",
        "low",
    ),  # Ochre = various EAOs in Shan/Kayah
    "Location dot darkslategray.svg": ("unknown", "Actor unidentified", "low"),
    "Location dot lightgrey.svg": ("unknown", "Actor unidentified", "low"),
    "Location dot red-black.svg": ("contested", "SAC vs unknown armed group", "medium"),
    # ── Dot variants (same meaning, different SVG style) ──────
    "Dot green 0d0.svg": (
        "resistance",
        "NUG/PDF resistance forces",
        "high",
    ),  # Same as green dot, alternate SVG
    "Dot yellow ff4.svg": (
        "contested",
        "Unclear/contested situation",
        "low",
    ),  # Yellow = ambiguous in this map
    # ── Animated GIFs = contested between two specific actors ─
    # Filename pattern: 80x80-COLOR1-COLOR2-anim.gif
    # COLOR1 = red (SAC), COLOR2 = the opposing actor's colour
    "80x80-red-blue-anim.gif": ("contested", "SAC vs KIO/KIA", "high"),
    "80x80-red-lime-anim.gif": ("contested", "SAC vs NUG/PDF", "high"),
    "80x80-red-magenta-anim.gif": ("contested", "SAC vs Chin PDF", "high"),
    "80x80-red-cyan-anim.gif": (
        "contested",
        "SAC vs Arakan Army (AA)",
        "high",
    ),  # Cyan = AA colour
    "80x80-red-black-anim.gif": ("contested", "SAC vs unknown armed group", "low"),
    "80x80-red-orange-anim.gif": ("contested", "SAC vs ethnic armed group", "high"),
    "80x80-red-green-anim.gif": ("contested", "SAC vs Arakan Army", "high"),
    "80x80-blue-lime-anim.gif": ("contested", "KIO/KIA vs NUG/PDF", "high"),
    # ── Grey dots = ambiguous / administrative ─────────────────
    "Map-dot-grey-68a.svg": (
        "unknown",
        "Actor unidentified",
        "low",
    ),  # Grey = no clear faction
}

# Icons we want to SKIP entirely (not control-status markers)
SKIP_MARKS = {
    "Myanmar Roadmap.png",  # Base map image — not a location marker
    "Fighter-jet-blue-icon.svg",  # SAC airbase marker — infrastructure, not control zone
    "Fighter-jet-green-icon.svg",  # Resistance airbase marker — infrastructure, not control zone
    "Map-circle-blue.svg",  # Circular overlay highlight — not a location dot
    "Map-circle-green.svg",  # Circular overlay highlight — not a location dot
    "Map-circle-yellow.svg",  # Circular overlay highlight — not a location dot
    "Map-arcNE-green.svg",  # Directional route arrow — not a location dot
    "WhiteDot.svg",  # Administrative reference point — ambiguous
    "Anchor pictogram green.svg",  # Port/naval facility icon — infrastructure
    "Mountain pass 12x12 e.svg",  # Mountain pass geographic marker
    "Mountain pass 12x12 n.svg",  # Mountain pass geographic marker
    "Mountain pass 12x12 ne.svg",  # Mountain pass geographic marker
    "Mountain pass 12x12 nw.svg",  # Mountain pass geographic marker
    "Mountain pass 12x12 se.svg",  # Mountain pass geographic marker
}

# ── 4. DEFINE THE REGEX PATTERN ──────────────────────────────

# We want to match entries that look like this (all on one line):
# { lat = "17.646", long = "95.460", mark = "Location dot red.svg", marksize = "18", label = "[[Hinthada]]", ... }
#
# PATTERN EXPLAINED piece by piece:
#   \{          → literal opening curly brace { (start of a Lua table entry)
#   [^}]*?      → any characters except }, non-greedy (= stops at first match, not last)
#   lat\s*=\s*  → the word "lat" followed by optional spaces, then "=", then optional spaces
#   "([^"]+)"   → a quoted value — captures everything between the double quotes (group 1 = latitude)
#   [^}]*?      → more characters between fields
#   long\s*=\s* → same pattern for longitude
#   "([^"]+)"   → captures longitude value (group 2)
#   [^}]*?      → more characters
#   mark\s*=\s* → same pattern for mark
#   "([^"]+)"   → captures mark/icon filename (group 3)
#   [^}]*       → any remaining characters inside this entry
#   \}          → literal closing curly brace } (end of Lua table entry)
#
# re.MULTILINE allows ^ and $ to match line boundaries (not just start/end of whole string)

ENTRY_PATTERN = re.compile(  # Compile the regex pattern into a reusable object (faster than recompiling each search)
    r"\{"  # Match literal opening brace {
    r"[^}]*?"  # Match any chars except } (non-greedy) — skips to "lat"
    r'lat\s*=\s*"([^"]+)"'  # Capture latitude value between quotes (group 1)
    r"[^}]*?"  # Skip chars between lat and long
    r'long\s*=\s*"([^"]+)"'  # Capture longitude value between quotes (group 2)
    r"[^}]*?"  # Skip chars between long and mark
    r'mark\s*=\s*"([^"]+)"'  # Capture icon/mark filename between quotes (group 3)
    r"[^}]*"  # Match any remaining content inside the entry
    r"\}",  # Match literal closing brace }
    re.DOTALL,  # re.DOTALL: makes . match newlines too (handles rare multi-line entries)
)

# Separate pattern to extract the label (= place name) from a matched entry block
# Labels can be:  label = "[[Hinthada]]"   OR   label = [[Siege of Ti Bwar]]  (with or without quotes)
LABEL_PATTERN = re.compile(  # Compile the label extraction pattern
    r'label\s*=\s*"?'  # Match 'label =', then optionally a quote (the ? makes " optional)
    r"\[?\[?"  # Match 0, 1, or 2 opening square brackets [[
    r'([^\]|"]+)'  # Capture the place name: anything that is NOT ], |, or "
    r'[\]|"]?'  # Match optional closing bracket, pipe, or quote
)

# ── 5. RUN THE REGEX AND EXTRACT ALL ENTRIES ─────────────────

matches = ENTRY_PATTERN.findall(
    lua_text
)  # findall: returns a list of ALL matches in the text; each match is a tuple (lat, long, mark)

print(f"✅ Raw regex matches found: {len(matches)}")  # Show total before filtering

# ── 6. PROCESS EACH MATCH INTO A STRUCTURED RECORD ──────────

records = []  # Empty list that will hold one dictionary (= structured record) per valid location entry
skipped_marks = {}  # Dictionary to track which mark types we skipped and how many times
unknown_marks = {}  # Dictionary to track marks we didn't recognise at all

for (
    lat_str,
    lon_str,
    mark,
) in matches:  # Unpack each tuple: lat_str = latitude string, lon_str = longitude string, mark = icon filename
    mark = (
        mark.strip()
    )  # strip(): remove any accidental leading/trailing whitespace from the mark name

    if mark in SKIP_MARKS:  # If this icon is in our skip list (airport, road, etc.)
        skipped_marks[mark] = (
            skipped_marks.get(mark, 0) + 1
        )  # Count how many times we skip this mark type
        continue  # continue: skip to the next loop iteration immediately

    if (
        mark not in MARK_MAPPING
    ):  # If this icon is not in our mapping AND not in skip list
        unknown_marks[mark] = (
            unknown_marks.get(mark, 0) + 1
        )  # Count occurrences so we can review them
        control_status = (
            "unknown"  # Label it "unknown" rather than silently dropping it
        )
        actor = f"UNMAPPED_MARK: {mark}"  # Keep the mark name so we can manually investigate
        confidence = "low"  # Low confidence because we don't know what this means
    else:
        control_status, actor, confidence = MARK_MAPPING[
            mark
        ]  # Unpack the 3-tuple from our mapping dictionary

    try:  # try: attempt the float conversion; if it fails go to except
        lat = float(
            lat_str
        )  # Convert latitude string (e.g., "17.646") to a decimal number
        lon = float(
            lon_str
        )  # Convert longitude string (e.g., "95.460") to a decimal number
    except ValueError:  # Catch the error if the string isn't a valid number
        print(
            f"  ⚠️  Invalid coordinates: lat='{lat_str}' lon='{lon_str}' — skipping"
        )  # Warn but don't crash
        continue  # Skip this entry and move on

    if not (
        -90 <= lat <= 90 and 60 <= lon <= 120
    ):  # Basic sanity check: Myanmar is roughly lat 10-29, lon 92-102
        print(
            f"  ⚠️  Coordinates outside Myanmar bounds: lat={lat}, lon={lon} — flagging"
        )  # Warn about suspicious coords
        confidence = "low"  # Downgrade confidence for out-of-bounds coordinates

    records.append(
        {  # Add a new dictionary to our records list (one per valid location)
            "lat": lat,  # Latitude as float
            "lon": lon,  # Longitude as float
            "mark": mark,  # Raw icon filename (kept for transparency/debugging)
            "control_status": control_status,  # Our interpreted control status
            "actor": actor,  # The specific armed actor controlling this location
            "confidence": confidence,  # Our confidence in this label
        }
    )

# ── 7. EXTRACT LOCATION NAMES (SECOND PASS) ──────────────────

# We now do a second pass over the raw text to also capture the label= field
# We use a different approach: find each full entry block, then extract both mark AND label together

FULL_ENTRY_PATTERN = re.compile(  # Compile a pattern to capture full entry blocks
    r'\{([^}]*lat\s*=\s*"[^"]*"[^}]*)\}',  # Capture everything inside { } that contains lat =
    re.DOTALL,  # Allow matching across line breaks
)

full_matches = FULL_ENTRY_PATTERN.findall(
    lua_text
)  # Get the full content of every { } entry containing lat

location_names = []  # List to hold extracted place names in order

for block in full_matches:  # Loop through each matched { } entry content
    label_match = LABEL_PATTERN.search(
        block
    )  # Search for a label= field within this entry block
    if label_match:  # If a label was found
        raw_label = label_match.group(
            1
        ).strip()  # group(1) gets the captured text; strip() removes whitespace
        # Clean up Wikipedia link syntax: "Bago, Myanmar|Bago" → keep only the display part after |
        if (
            "|" in raw_label
        ):  # Check if the label has a pipe character (Wikipedia's display name separator)
            raw_label = raw_label.split("|")[
                -1
            ]  # Keep only the text after the last | (= the display name)
        location_names.append(raw_label)  # Add the cleaned name to our list
    else:  # If no label was found in this block
        location_names.append(
            ""
        )  # Add empty string as placeholder to keep lists aligned

# ── 8. ALIGN RECORDS WITH LOCATION NAMES ─────────────────────

# Our two passes may not align perfectly because FULL_ENTRY_PATTERN catches ALL entries
# but records only has non-skipped, valid entries. We rebuild with names here.

# Re-run the full extraction in one unified pass for clean alignment:
records_with_names = []  # New list that will combine coordinates + names in one unified pass

for block in full_matches:  # Loop over every { } block that contains a lat= field
    lat_m = re.search(
        r'lat\s*=\s*"([^"]+)"', block
    )  # Search for lat= within this block
    lon_m = re.search(
        r'long\s*=\s*"([^"]+)"', block
    )  # Search for long= within this block
    mark_m = re.search(
        r'mark\s*=\s*"([^"]+)"', block
    )  # Search for mark= within this block

    if not (lat_m and lon_m and mark_m):  # If any required field is missing
        continue  # Skip incomplete entries

    mark = mark_m.group(1).strip()  # Extract and clean the mark filename

    if mark in SKIP_MARKS:  # Skip non-control icons (airports, roads, etc.)
        continue

    try:  # Attempt coordinate conversion
        lat = float(lat_m.group(1))  # Convert latitude to float
        lon = float(lon_m.group(1))  # Convert longitude to float
    except ValueError:  # If conversion fails
        continue  # Skip this entry

    label_m = LABEL_PATTERN.search(block)  # Search for the place name label
    location_name = ""  # Default to empty if no label found
    if label_m:  # If a label was found
        location_name = label_m.group(1).strip()  # Extract and clean it
        if "|" in location_name:  # Handle "Wikipedia name|Display name" format
            location_name = location_name.split("|")[
                -1
            ]  # Keep only the display name after the pipe

    if mark not in MARK_MAPPING:  # Handle unrecognised marks
        status, actor, conf = (
            "unknown",
            f"UNMAPPED: {mark}",
            "low",
        )  # Assign unknown labels
    else:  # Handle recognised marks
        status, actor, conf = MARK_MAPPING[mark]  # Unpack from our mapping

    records_with_names.append(
        {  # Build the final record dictionary
            "lat": lat,  # Latitude as float
            "lon": lon,  # Longitude as float
            "location_name": location_name,  # Human-readable place name
            "mark": mark,  # Raw Wikipedia icon filename
            "control_status": status,  # Interpreted control status
            "actor": actor,  # Specific armed actor
            "confidence": conf,  # Label confidence level
        }
    )

# ── 9. BUILD THE PANDAS DATAFRAME ────────────────────────────

df = pd.DataFrame(
    records_with_names
)  # Convert list of dictionaries into a DataFrame (= table)

print(
    f"\n✅ Total valid location entries extracted: {len(df)}"
)  # Report how many rows we have

# ── 10. PRINT SUMMARY STATISTICS ─────────────────────────────

print("\n── Control Status Distribution ──")  # Label the summary section
print(
    df["control_status"].value_counts().to_string()
)  # Count entries per control status and print as clean text

print("\n── Top 10 Mark Types Found ──")  # Label the mark type summary
print(
    df["mark"].value_counts().head(10).to_string()
)  # Show the 10 most common icon types

if unknown_marks:  # Only print if there are unknown marks
    print("\n⚠️  Unrecognised marks (need manual review):")  # Flag these for attention
    for m, count in sorted(
        unknown_marks.items()
    ):  # Sort alphabetically for readability
        print(f"   {count:>3}x  {m}")  # Print count and mark name

if skipped_marks:  # Only print if we skipped anything
    print("\nℹ️  Skipped non-control marks:")  # Inform about intentional skips
    for m, count in sorted(skipped_marks.items()):  # Sort alphabetically
        print(f"   {count:>3}x  {m}")  # Print count and mark name

# ── 11. SHOW SAMPLE ROWS ─────────────────────────────────────

print("\n── First 10 Parsed Entries ──")  # Label the sample section
print(
    df[["lat", "lon", "location_name", "control_status", "actor", "confidence"]]
    .head(10)
    .to_string()
)  # Print first 10 rows

# ── 12. SAVE TO CSV ───────────────────────────────────────────

output_path = (
    OUTPUT_DIR / "myanmar_wikipedia_parsed_step2.csv"
)  # Define the output file path
df.to_csv(
    output_path, index=False, encoding="utf-8"
)  # Save DataFrame to CSV; index=False = don't write row numbers

print(f"\n✅ Parsed data saved to: {output_path}")  # Confirm where the file was saved

print("\n" + "=" * 60)  # Visual separator
print("STEP 2 COMPLETE.")  # Announce completion
print("Review: Are the control statuses and actors correct?")  # Prompt for verification
print("Paste output here before we proceed to Step 3.")  # Instruction for next step
print("=" * 60)  # Closing separator
