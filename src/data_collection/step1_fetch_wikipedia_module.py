# ============================================================
# STEP 1: Fetch the raw Lua module from the Wikipedia API
# Project: BSE / World Bank Territorial Control Thesis
# Author:  Tizian Schenk
# Date:    07.05.2026
# ============================================================

# The script opens by creating a folder to save our future CSV, then constructs a precise URL for Wikipedia's public API
# not scraping HTML, but asking the server directly for the raw text of the Myanmar Civil War Lua module.
# It sets a polite User-Agent header so Wikipedia's servers know who is making the request.
# It then makes the HTTP GET request with a safety timeout, and if anything goes wrong
# it tells you exactly what happened and stops cleanly. Once it gets the response,
# it navigates through the nested JSON (like opening a set of Russian dolls) to retrieve the raw Lua source code.
# It counts and displays the first 200 lines so you can inspect the structure before writing any parser,
# and saves the full source to disk as a .lua file for reference.
# Finally, it searches for lines containing "lat" to give you
# a quick preview of how many location entries exist and what they look like.

# ── IMPORTANT LIBRARIES ─────────────────────────────────────────

import json  # json: built-in library to parse JSON (= JavaScript Object Notation, a human-readable data format)
import pathlib  # pathlib: built-in library for handling file paths in an OS-independent way
import sys  # sys: built-in library to access system-level functions like printing errors and exiting

import requests  # requests: the standard library for making HTTP requests (= asking web servers for data)

# ── 1. DEFINE THE OUTPUT DIRECTORY ──────────────────────────

OUTPUT_DIR = pathlib.Path(
    "data/raw/validation"
)  # Define where we will save all output files
OUTPUT_DIR.mkdir(
    parents=True, exist_ok=True
)  # Create the directory if it doesn't exist (parents=True = create intermediate folders too)

# ── 2. DEFINE THE WIKIPEDIA API URL ─────────────────────────

# Wikipedia exposes a public API at /w/api.php — we send parameters as URL query strings
# action=query         → we want to query (= ask for) a page
# titles=Module:...    → the exact name of the Wikipedia module (Lua script) we want
# prop=revisions       → we want the revision history (= version history) of that page
# rvprop=content       → specifically, we want the raw text *content* of the latest revision
# rvslots=main         → "slots" are Wikipedia's content compartments; "main" is the primary content slot
# format=json          → return the response as JSON (not HTML)
# formatversion=2      → use the modern, cleaner version of the JSON format

API_URL = (
    "https://en.wikipedia.org/w/api.php"  # The base URL of the Wikipedia API endpoint
    "?action=query"  # Tell the API we want to query page content
    "&titles=Module:Myanmar_Civil_War_detailed_map"  # The exact page title of the Lua module
    "&prop=revisions"  # We want revision (= version) information for this page
    "&rvprop=content"  # Within revisions, we want the actual raw text content
    "&rvslots=main"  # Access the "main" content slot (where Lua code lives)
    "&format=json"  # Return the data as JSON so Python can read it easily
    "&formatversion=2"  # Use the cleaner v2 JSON format (fewer nested levels)
)

# ── 3. SET REQUEST HEADERS ───────────────────────────────────

# Wikipedia's API requires a User-Agent header (= a label identifying who/what is making the request)
# Without this, Wikipedia may block or rate-limit our requests
# Good practice: identify your project and contact email so Wikipedia admins can reach you
HEADERS = {
    "User-Agent": (
        "TizianSchenk-BSEThesis/1.0 "  # Identify the script name and version
        "Barcelona School of Economics; "  # Identify the institution
        "Master thesis - territorial control; "  # Identify the project context
        "contact: tizian.schenk@bse.eu)"  # contact email
    )
}

# ── 4. MAKE THE API REQUEST ──────────────────────────────────

print("=" * 60)  # Print a visual separator line for readability
print(
    "STEP 1: Fetching Myanmar Civil War Module from Wikipedia API"
)  # Announce what we are doing
print("=" * 60)  # Print closing separator
print(
    f"API URL: {API_URL}\n"
)  # Show the full URL we are querying (useful for debugging)

try:  # try: attempt the following code; if it fails, jump to except
    response = requests.get(  # Send an HTTP GET request (= ask the server to give us data)
        API_URL,  # The URL we defined above
        headers=HEADERS,  # Include our User-Agent so Wikipedia knows who we are
        timeout=30,  # timeout=30: give up if the server doesn't respond within 30 seconds
    )
    response.raise_for_status()  # raise_for_status: if the server returned an error code (e.g. 404 Not Found), raise an exception immediately

except (
    requests.exceptions.Timeout
):  # Catch the specific error if the server was too slow
    print(
        "ERROR: Request timed out. Check your internet connection."
    )  # Tell the user what went wrong
    sys.exit(
        1
    )  # sys.exit(1): stop the script with error code 1 (= "something went wrong")

except (
    requests.exceptions.RequestException
) as e:  # Catch any other network-related error
    print(
        f"ERROR: Could not connect to Wikipedia API: {e}"
    )  # Print the actual error message
    sys.exit(1)  # Exit the script so we don't continue with bad data

# ── 5. PARSE THE JSON RESPONSE ───────────────────────────────

data = response.json()  # Parse the HTTP response body as JSON into a Python dictionary

# The Wikipedia API returns a nested dictionary (= data inside data inside data)
# Structure: data → "query" → "pages" → [list of pages] → "revisions" → [list of versions] → "slots" → "main" → "content"

pages = data["query"][
    "pages"
]  # Navigate into the "query" → "pages" section of the response

if not pages:  # If "pages" is empty (= no results found)
    print("ERROR: No pages returned from Wikipedia API.")  # Warn the user
    sys.exit(1)  # Exit because we have nothing to work with

page = pages[
    0
]  # Take the first (and only) page result (formatversion=2 returns a list)

if "revisions" not in page:  # Check if the revisions section exists in the response
    print(
        "ERROR: No revisions found. The page may have moved or been deleted."
    )  # Inform user of likely cause
    print("Raw API response:")  # Print the raw response to help diagnose the problem
    print(
        json.dumps(data, indent=2)[:2000]
    )  # json.dumps: convert dict back to string; indent=2 = pretty-print; [:2000] = first 2000 chars only
    sys.exit(1)  # Exit if we cannot retrieve the module content

# ── 6. EXTRACT THE RAW LUA SOURCE TEXT ──────────────────────

# Lua (= a lightweight scripting language that Wikipedia uses for complex templates/modules)
# The raw source text contains the Lua code with all location data encoded as Lua tables

lua_source = page["revisions"][0]["slots"]["main"][
    "content"
]  # Navigate deep into the JSON to get the raw Lua text

print("✅ Successfully fetched module!")  # Confirm success
print(
    f"   Total characters in module: {len(lua_source):,}"
)  # Show total size (a large module = many entries)
print(
    f"   Total lines in module:      {lua_source.count(chr(10)):,}\n"
)  # Count newlines to show total line count

# ── 7. DISPLAY THE FIRST 200 LINES ──────────────────────────

lines = lua_source.split(
    "\n"
)  # split("\n"): break the full text into a list of individual lines on newline characters

print("-" * 60)  # Visual separator
print("FIRST 200 LINES OF THE LUA MODULE:")  # Label what we are printing
print("-" * 60)  # Visual separator

for i, line in enumerate(
    lines[:200]
):  # Loop through the first 200 lines; enumerate gives us both index (i) and content (line)
    print(
        f"{i + 1:>4} | {line}"
    )  # Print line number (right-aligned in 4 chars) then the line content; {i+1} because enumerate starts at 0

# ── 8. SAVE RAW MODULE TEXT TO DISK ─────────────────────────

raw_output_path = (
    OUTPUT_DIR / "myanmar_module_raw.lua"
)  # Define the file path where we will save the raw Lua source

raw_output_path.write_text(  # write_text: write a string directly to a file (creates the file if it doesn't exist)
    lua_source,  # The full Lua source text we extracted
    encoding="utf-8",  # encoding="utf-8": use UTF-8 character encoding to support Myanmar script and special characters
)

print(
    f"\n✅ Raw Lua module saved to: {raw_output_path}"
)  # Confirm where the file was saved

# ── 9. SHOW STRUCTURAL PREVIEW ───────────────────────────────

# Look for the word "lat" to find lines that contain location data
location_lines = [
    l for l in lines if "lat" in l.lower()
]  # List comprehension: filter to keep only lines containing "lat" (case-insensitive)

print(
    f"\n📍 Lines containing 'lat' (location entries): {len(location_lines)}"
)  # Show how many potential location entries we found
print("\nSample location lines (first 5):")  # Label the preview section
for sample_line in location_lines[
    :5
]:  # Loop through just the first 5 location-containing lines
    print(
        f"   → {sample_line.strip()}"
    )  # strip(): remove leading/trailing whitespace before printing

print("\n" + "=" * 60)  # Print final separator
print("STEP 1 COMPLETE. Review the 200 lines above.")  # Announce step completion
print(
    "Confirm: can you see entries with lat=, long=, mark= ?"
)  # Ask user to visually verify the structure
print("=" * 60)  # Closing separator
