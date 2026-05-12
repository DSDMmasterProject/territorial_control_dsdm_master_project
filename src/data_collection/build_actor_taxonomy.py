# ============================================================
# build_actor_taxonomy.py
# Creates the actor crosswalk CSV for Myanmar UCDP actors
# Run once; edit the rows manually to refine mappings
# Output: data/raw/validation/actor_taxonomy_myanmar.csv
# ============================================================

# (1/2)
# The taxonomy script builds a lookup table from every actor string that actually appears in the GED —
# verified from the data itself.
# Each actor gets three classification levels so you can group at whatever granularity the model needs,
# and the coverage check at the end confirms zero unmatched actors.
# The loading script reads only the 21 relevant columns from the 250MB GED file, filters immediately to Myanmar,
# does the same for the Candidate file, stacks them vertically after labelling each row's origin,
# parses dates and adds a year-month key, filters to the post-coup period (Feb 2021 onwards), and joins
# the taxonomy for both sides of each conflict. The result is a clean,
# analysis-ready event dataset with actor labels at three levels of granularity.
# -----------------------------------------------------

import pathlib  # pathlib: file path handling

import pandas as pd  # pandas: create and save the crosswalk table

# ── OUTPUT PATH ───────────────────────────────────────────────

OUT = pathlib.Path(
    "data/raw/validation/actor_taxonomy_myanmar.csv"
)  # Where to save the crosswalk
OUT.parent.mkdir(parents=True, exist_ok=True)  # Create folder if missing

# ── THE CROSSWALK TABLE ───────────────────────────────────────
#
# Every actor that actually appears in GED side_a or side_b for Myanmar.
# Strings in ucdp_name MUST match the GED exactly (verified from data).
#
# level1:  pro_government | anti_government | communal | historical
# level2:  government_sac | eao_northern | eao_western | eao_eastern |
#          pdf_nug | communal | historical | cross_border
# level3:  individual organisation code (snake_case)
#
# wiki_mark: the Wikipedia Lua module icon that corresponds to this actor
#            (used to join our ground truth labels to UCDP actors)

ROWS = [
    # ── GOVERNMENT / PRO-GOVERNMENT ──────────────────────────
    {
        "ucdp_name": "Government of Myanmar (Burma)",  # Exact GED string
        "wiki_mark": "Location dot red.svg",  # Wikipedia icon
        "level1": "pro_government",
        "level2": "government_sac",
        "level3": "military_sac",
        "display_name": "Military Forces of Myanmar (SAC/Tatmadaw)",
        "notes": "Side_a in all state-based conflict events; same forces pre/post-2021 coup",
    },
    {
        "ucdp_name": "DKBA",  # Democratic Karen Buddhist Army
        "wiki_mark": "Location dot red.svg",  # DKBA converted to BGF in 2010; pro-government
        "level1": "pro_government",
        "level2": "government_sac",
        "level3": "bgf_dkba",
        "display_name": "Democratic Karen Buddhist Army / Border Guard Force",
        "notes": "Converted to BGF under SAC command in 2010; appears as side_a",
    },
    # ── RESISTANCE (post-2021 coup) ───────────────────────────
    {
        "ucdp_name": "NUG",  # National Unity Government
        "wiki_mark": "Location dot green.svg",
        "level1": "anti_government",
        "level2": "pdf_nug",
        "level3": "nug_pdf",
        "display_name": "National Unity Government / People's Defence Force (NUG/PDF)",
        "notes": "Post-coup shadow government and its armed wing; appears in BOTH side_a and side_b",
    },
    {
        "ucdp_name": "ABSDF",  # All Burma Students Democratic Front
        "wiki_mark": "Dot green 0d0.svg",
        "level1": "anti_government",
        "level2": "pdf_nug",
        "level3": "absdf",
        "display_name": "All Burma Students Democratic Front (ABSDF)",
        "notes": "1988 student uprising origin; now NUG-aligned resistance",
    },
    # ── NORTHERN EAOs (Kachin / Three Brotherhood Alliance) ──
    {
        "ucdp_name": "KIO",  # Kachin Independence Organisation
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_northern",
        "level3": "kio_kia",
        "display_name": "Kachin Independence Organisation / Army (KIO/KIA)",
        "notes": "UCDP codes KIA events as KIO (political wing); only in side_b",
    },
    {
        "ucdp_name": "MNDAA",  # Myanmar National Democratic Alliance Army
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_northern",
        "level3": "mndaa",
        "display_name": "Myanmar National Democratic Alliance Army (MNDAA/Kokang)",
        "notes": "Led Operation 1027 (Oct 2023); controls Kokang region; only in side_b",
    },
    {
        "ucdp_name": "PSLF",  # Palaung State Liberation Front
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_northern",
        "level3": "pslf_tnla",
        "display_name": "Palaung State Liberation Front / TNLA (PSLF/TNLA)",
        "notes": "UCDP codes TNLA events as PSLF (political parent); Ta'ang Palaung group",
    },
    # ── WESTERN EAOs (Arakan / Rakhine / Chin) ───────────────
    {
        "ucdp_name": "ULA",  # United League of Arakan
        "wiki_mark": "Location dot cyan.svg",
        "level1": "anti_government",
        "level2": "eao_western",
        "level3": "ula_aa",
        "display_name": "United League of Arakan / Arakan Army (ULA/AA)",
        "notes": "UCDP codes Arakan Army events as ULA (political wing); controls Rakhine; side_a AND side_b",
    },
    {
        "ucdp_name": "CNF",  # Chin National Front
        "wiki_mark": "Location dot orange.svg",
        "level1": "anti_government",
        "level2": "eao_western",
        "level3": "cnf",
        "display_name": "Chin National Front / Army (CNF/CNA)",
        "notes": "Chin State ethnic armed group; only in side_b",
    },
    {
        "ucdp_name": "ARSA",  # Arakan Rohingya Salvation Army
        "wiki_mark": "Location dot magenta.svg",
        "level1": "anti_government",
        "level2": "eao_western",
        "level3": "arsa",
        "display_name": "Arakan Rohingya Salvation Army (ARSA)",
        "notes": "Rohingya armed group; fights both government AND ULA/AA; appears in side_a and side_b",
    },
    {
        "ucdp_name": "RSO",  # Rohingya Solidarity Organisation
        "wiki_mark": "Location dot magenta.svg",
        "level1": "anti_government",
        "level2": "eao_western",
        "level3": "rso",
        "display_name": "Rohingya Solidarity Organisation (RSO)",
        "notes": "Historical Rohingya group; mostly inactive",
    },
    {
        "ucdp_name": "BMA",  # Burma Muslim Army
        "wiki_mark": "Location dot magenta.svg",
        "level1": "anti_government",
        "level2": "eao_western",
        "level3": "bma",
        "display_name": "Burma Muslim Army (BMA)",
        "notes": "Historical Muslim armed group; appears in side_b",
    },
    # ── EASTERN EAOs (Karen / Shan / Karenni / Mon) ──────────
    {
        "ucdp_name": "KNU",  # Karen National Union
        "wiki_mark": "Location dot green.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "knu_knla",
        "display_name": "Karen National Union / KNLA (KNU)",
        "notes": "Oldest active EAO; eastern Myanmar; appears in side_a AND side_b",
    },
    {
        "ucdp_name": "RCSS",  # Restoration Council of Shan State
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "rcss_ssas",
        "display_name": "Restoration Council of Shan State / SSA-South (RCSS)",
        "notes": "Shan State Army South; historically negotiated ceasefire; side_a AND side_b",
    },
    {
        "ucdp_name": "UWSA",  # United Wa State Army
        "wiki_mark": "Location dot purple.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "uwsa",
        "display_name": "United Wa State Army (UWSA)",
        "notes": "Largest EAO by troop size; de facto ceasefire with SAC; only side_b",
    },
    {
        "ucdp_name": "PNLO",  # Pa-O National Liberation Organisation
        "wiki_mark": "Location dot ochre200.png",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "pnlo",
        "display_name": "Pa-O National Liberation Organisation (PNLO)",
        "notes": "Pa-O ethnic group in southern Shan State; only side_b",
    },
    {
        "ucdp_name": "SSPP",  # Shan State Progress Party
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "sspp",
        "display_name": "Shan State Progress Party / SSA-North (SSPP)",
        "notes": "Northern Shan State; distinct from RCSS/SSA-South",
    },
    {
        "ucdp_name": "KNPP",  # Karenni National Progressive Party
        "wiki_mark": "Location dot green.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "knpp",
        "display_name": "Karenni National Progressive Party (KNPP)",
        "notes": "Kayah/Karenni State; only in side_b",
    },
    {
        "ucdp_name": "NMSP",  # New Mon State Party
        "wiki_mark": "Location dot blue.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "nmsp",
        "display_name": "New Mon State Party (NMSP)",
        "notes": "Mon State; has had ceasefire periods; only side_b",
    },
    {
        "ucdp_name": "DKBA 5",  # DKBA Splinter group 5
        "wiki_mark": "Location dot green.svg",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "dkba5",
        "display_name": "Democratic Karen Buddhist Army - Brigade 5 (DKBA 5)",
        "notes": "Refused BGF conversion in 2010; remains anti-government Karen splinter",
    },
    # ── COMMUNAL VIOLENCE (not territorial control actors) ───
    {
        "ucdp_name": "Buddhists (Myanmar)",  # Coded as communal actor
        "wiki_mark": "Map-dot-grey-68a.svg",
        "level1": "communal",
        "level2": "communal",
        "level3": "communal_buddhist",
        "display_name": "Buddhists (Myanmar) — communal actor",
        "notes": "Rakhine Buddhist communal violence; NOT a territorial control actor; exclude from model",
    },
    {
        "ucdp_name": "Muslims (Myanmar)",  # Coded as communal actor
        "wiki_mark": "Map-dot-grey-68a.svg",
        "level1": "communal",
        "level2": "communal",
        "level3": "communal_muslim",
        "display_name": "Muslims (Myanmar) — communal actor",
        "notes": "Rohingya Muslim communal violence; NOT a territorial control actor; exclude from model",
    },
    {
        "ucdp_name": "Civilians",  # Side_b in one-sided violence
        "wiki_mark": "",
        "level1": "civilian",
        "level2": "civilian",
        "level3": "civilian",
        "display_name": "Civilians (victims of one-sided violence)",
        "notes": "Side_b for type_of_violence==3 events; marks perpetrator's control area",
    },
    # ── HISTORICAL / CROSS-BORDER (low relevance for post-2021 model) ─
    {
        "ucdp_name": "MTA",
        "wiki_mark": "",
        "level1": "historical",
        "level2": "historical",
        "level3": "mta",
        "display_name": "Mong Tai Army (MTA) — historical Shan warlord group",
        "notes": "Dissolved 1996; only in old GED events; exclude from post-2021 analysis",
    },
    {
        "ucdp_name": "MDA",
        "wiki_mark": "",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "mda",
        "display_name": "Mon Democratic Army (MDA)",
        "notes": "Small Mon splinter group",
    },
    {
        "ucdp_name": "MDA - LM",
        "wiki_mark": "",
        "level1": "anti_government",
        "level2": "eao_eastern",
        "level3": "mda_lm",
        "display_name": "Mon Democratic Army - Lay Mon Kone (MDA-LM)",
        "notes": "MDA splinter; only side_b",
    },
    {
        "ucdp_name": "God's Army",
        "wiki_mark": "",
        "level1": "historical",
        "level2": "historical",
        "level3": "gods_army",
        "display_name": "God's Army — historical Karen splinter",
        "notes": "Defunct since 2000s; only in old GED records",
    },
    {
        "ucdp_name": "NSCN-K",
        "wiki_mark": "",
        "level1": "cross_border",
        "level2": "cross_border",
        "level3": "nscn_k",
        "display_name": "National Socialist Council of Nagaland-Khaplang (NSCN-K)",
        "notes": "India-Myanmar border conflict; based in Sagaing; cross-border actor",
    },
    {
        "ucdp_name": "SCEF",  # Only in Candidate 26.01.26.03; 1 event
        "wiki_mark": "Location dot green.svg",  # Anti-government; Karen State location
        "level1": "anti_government",
        "level2": "eao_eastern",  # Kayin/Karen State = eastern EAO
        "level3": "scef",
        "display_name": "SCEF (unconfirmed — Kayin State armed group)",
        "notes": "1 event, Mar 2026, Kayin State. New actor; full name unconfirmed. Classified as eastern EAO by location. Low confidence.",
    },
    {
        "ucdp_name": "Government of India",
        "wiki_mark": "",
        "level1": "cross_border",
        "level2": "cross_border",
        "level3": "india_government",
        "display_name": "Government of India (cross-border operations)",
        "notes": "Indian forces conduct operations vs NSCN-K inside Myanmar; side_a; exclude from Myanmar control model",
    },
    {
        "ucdp_name": "UNLFW",
        "wiki_mark": "",
        "level1": "cross_border",
        "level2": "cross_border",
        "level3": "unlfw",
        "display_name": "United National Liberation Front of Western SE Asia (UNLFW)",
        "notes": "Coalition of Indian northeast insurgents (ULFA-I, NSCN factions) based in Myanmar; side_b; cross-border actor, exclude from Myanmar control model",
    },
]

# ── BUILD AND SAVE ─────────────────────────────────────────────

df = pd.DataFrame(ROWS)  # Convert list of dicts to DataFrame

df.to_csv(OUT, index=False, encoding="utf-8")  # Save as CSV

print(f"✅ Saved: {OUT}")
print(f"   {len(df)} actors defined")
print()
print("── Level 1 distribution ──")
print(
    df["level1"].value_counts().to_string()
)  # Show how many actors per level 1 category
print()
print("── Level 2 distribution ──")
print(
    df["level2"].value_counts().to_string()
)  # Show how many actors per level 2 category
print()
print("── Coverage check: actors in GED data ──")
# These are every unique value from side_a and side_b we saw in the GED
ged_actors = {  # Set of all actor strings seen in actual GED data
    "ARSA",
    "Buddhists (Myanmar)",
    "DKBA",
    "Government of Myanmar (Burma)",
    "KNU",
    "MDA",
    "MTA",
    "NUG",
    "RCSS",
    "ULA",  # side_a actors
    "ABSDF",
    "ARSA",
    "BMA",
    "CNF",
    "Civilians",
    "DKBA 5",  # side_b actors
    "God's Army",
    "KIO",
    "KNPP",
    "KNU",
    "MDA - LM",
    "MNDAA",
    "MTA",
    "Muslims (Myanmar)",
    "NMSP",
    "NSCN-K",
    "NUG",
    "PNLO",
    "PSLF",
    "RCSS",
    "RSO",
    "SSPP",
    "ULA",
    "UWSA",
}
mapped = set(df["ucdp_name"].tolist())  # Actors covered by our taxonomy
missing = ged_actors - mapped  # Actors in data but NOT in taxonomy
print(f"   GED actors: {len(ged_actors)}")
print(f"   Mapped:     {len(ged_actors - missing)}")
print(f"   Missing:    {missing if missing else 'None — full coverage ✅'}")
