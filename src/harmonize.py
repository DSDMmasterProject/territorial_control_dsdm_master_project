"""
harmonize.py — Unified actor and region harmonization for conflict mapping.

Maps raw actor names and administrative region names from ACLED, UCDP, and
Wikipedia to canonical identifiers, broad categories, and alliance labels.

Pilot country: Myanmar (MMR). Add Somalia, Nigeria, Ecuador by appending
new sections to each dict/list and registering them in the country-keyed
top-level maps (REGION_MAP_L1, REGION_MAP_L2).

Canonical broad categories : Junta | Resistance | Autonomous | Other | Contested | Civilians | Unknown
Canonical alliances         : SAC   | NUG        | NUG-noncombatant | Independent | Civilian | None

Usage
-----
    from src.harmonize import normalize_actor, normalize_region, normalize_dataframe

    info = normalize_actor("KNU/KNLA: Karen National Union/Karen National Liberation Army")
    # {"canonical_id": "KNU", "canonical_name": "Karen National Union / ...",
    #  "broad_category": "Resistance", "alliance": "NUG"}

    region = normalize_region("Magwe division")
    # "Magway"

    df_out = normalize_dataframe(df_acled, source="acled")
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ===========================================================================
# ACTOR REGISTRY — exact-match lookup table
#
# Keys are raw strings exactly as they appear in ACLED / UCDP / Wikipedia.
# To extend to a new country, append new entries at the bottom of each
# section. Do not restructure existing entries.
#
# Schema per value dict:
#   canonical_id   : SHORT_UPPERCASE code used in outputs (e.g. "KNU", "PDF")
#   canonical_name : Full readable name for display
#   broad_category : Junta | Resistance | Autonomous | Other |
#                    Contested | Civilians | Unknown
#   alliance       : SAC | NUG | NUG-noncombatant | Independent |
#                    Civilian | None
# ===========================================================================

ACTOR_REGISTRY: dict[str, dict] = {

    # -----------------------------------------------------------------------
    # MYANMAR — Junta / SAC alliance
    # -----------------------------------------------------------------------

    # State Administration Council (post-coup government)
    "Government of Myanmar (2021-) State Administration Council": {
        "canonical_id": "SAC",
        "canonical_name": "State Administration Council",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Government of Myanmar (2021-)": {
        "canonical_id": "SAC",
        "canonical_name": "State Administration Council",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "State Administration Council": {
        "canonical_id": "SAC",
        "canonical_name": "State Administration Council",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # Civilian government variants (ACLED time-period labels)
    "Government of Myanmar (2016-)": {
        "canonical_id": "GOV_MYANMAR",
        "canonical_name": "Government of Myanmar",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Government of Myanmar (2011-2016)": {
        "canonical_id": "GOV_MYANMAR",
        "canonical_name": "Government of Myanmar",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Government of Myanmar (Burma)": {
        "canonical_id": "GOV_MYANMAR",
        "canonical_name": "Government of Myanmar",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Government of Myanmar": {
        "canonical_id": "GOV_MYANMAR",
        "canonical_name": "Government of Myanmar",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # Tatmadaw — all ACLED time-period variants
    "Military Forces of Myanmar (2021-)": {
        "canonical_id": "TATMADAW",
        "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (2016-2021)": {
        "canonical_id": "TATMADAW",
        "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (2011-2016)": {
        "canonical_id": "TATMADAW",
        "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (1988-2011)": {
        "canonical_id": "TATMADAW",
        "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Tatmadaw": {
        "canonical_id": "TATMADAW",
        "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # Border Guard Force (BGF) — Tatmadaw-controlled former ceasefire groups
    "Military Forces of Myanmar (2021-) Border Guard Force": {
        "canonical_id": "BGF",
        "canonical_name": "Border Guard Force (Tatmadaw-aligned)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (2016-2021) Border Guard Force": {
        "canonical_id": "BGF",
        "canonical_name": "Border Guard Force (Tatmadaw-aligned)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (2011-2016) Border Guard Force": {
        "canonical_id": "BGF",
        "canonical_name": "Border Guard Force (Tatmadaw-aligned)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (1988-2011) Border Guard Force": {
        "canonical_id": "BGF",
        "canonical_name": "Border Guard Force (Tatmadaw-aligned)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # People's Militia Force / Pyu Saw Htee (pro-junta civilian militia)
    "Military Forces of Myanmar (2021-) People's Militia Force": {
        "canonical_id": "PYU_SAW_HTEE",
        "canonical_name": "Pyu Saw Htee / People's Militia Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Military Forces of Myanmar (2016-2021) People's Militia Force": {
        "canonical_id": "PYU_SAW_HTEE",
        "canonical_name": "Pyu Saw Htee / People's Militia Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Pyu Saw Htee": {
        "canonical_id": "PYU_SAW_HTEE",
        "canonical_name": "Pyu Saw Htee (pro-junta militia)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # Police Forces of Myanmar — all ACLED variants
    "Police Forces of Myanmar (2021-)": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2016-2021)": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2016-2021) Border Guard Police": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force / Border Guard Police",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2016-2021) Prison Guards": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force / Prison Guards",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2011-2016)": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2011-2016) Border Guard Police": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force / Border Guard Police",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (2011-2016) Border Area Immigration Scrutinization and Supervision Bureau (NaSaKa)": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force / NaSaKa",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Police Forces of Myanmar (1988-2011)": {
        "canonical_id": "POLICE_MYANMAR",
        "canonical_name": "Myanmar Police Force",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # SAC-aligned ethnic armed groups
    "PNO/PNA: Pa-Oh National Organization/Pa-Oh National Army": {
        "canonical_id": "PNA",
        "canonical_name": "Pa-Oh National Army (Pa-Oh National Organization)",
        "broad_category": "Junta",
        "alliance": "SAC",
        # NOTE: distinct from PNLO which is NUG-aligned — same ethnicity, opposing sides
    },
    "Pa-O National Army": {
        "canonical_id": "PNA",
        "canonical_name": "Pa-Oh National Army (Pa-Oh National Organization)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "ZRA: Zomi Revolutionary Army": {
        "canonical_id": "ZRA",
        "canonical_name": "Zomi Revolutionary Army",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "Zomi Revolutionary Army": {
        "canonical_id": "ZRA",
        "canonical_name": "Zomi Revolutionary Army",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "KPC: KNU/KNLA Peace Council": {
        "canonical_id": "KPC",
        "canonical_name": "KNU/KNLA Peace Council",
        "broad_category": "Junta",
        "alliance": "SAC",
        # TODO: verify post-2021 alignment; KPC was historically pro-government splinter of KNU
    },
    # Karen National Army — BGF-aligned Karen unit (distinct from KNU/KNLA)
    "Karen National Army": {
        "canonical_id": "KNA",
        "canonical_name": "Karen National Army (BGF-aligned)",
        "broad_category": "Junta",
        "alliance": "SAC",
        # TODO: verify — may refer specifically to the BGF-Karen unit under Tatmadaw
    },

    # DKBA (original / Buddhist) — SAC-aligned
    # NOTE: "DKBA (Buddhist)" is SAC-aligned; "DKBA (Benevolent)" / "DKBA 5" is NUG-aligned
    "DKBA (Buddhist): Democratic Karen Buddhist Army (1994-2010)": {
        "canonical_id": "DKBA",
        "canonical_name": "Democratic Karen Buddhist Army (original)",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "DKBA (Buddhist): Democratic Karen Buddhist Army (2016-)": {
        "canonical_id": "DKBA",
        "canonical_name": "Democratic Karen Buddhist Army",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "DKBA (Democratic Karen Buddhist Army)": {
        "canonical_id": "DKBA",
        "canonical_name": "Democratic Karen Buddhist Army",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # Political party of the military
    "USDP: Union Solidarity and Development Party": {
        "canonical_id": "USDP",
        "canonical_name": "Union Solidarity and Development Party",
        "broad_category": "Junta",
        "alliance": "SAC",
    },
    "USDP": {
        "canonical_id": "USDP",
        "canonical_name": "Union Solidarity and Development Party",
        "broad_category": "Junta",
        "alliance": "SAC",
    },

    # -----------------------------------------------------------------------
    # MYANMAR — Resistance / NUG alliance
    # -----------------------------------------------------------------------

    # National Unity Government (political leadership of the resistance)
    "NUG (National Unity Government)": {
        "canonical_id": "NUG_GOV",
        "canonical_name": "National Unity Government",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "NUG": {
        "canonical_id": "NUG_GOV",
        "canonical_name": "National Unity Government",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "National Unity Government": {
        "canonical_id": "NUG_GOV",
        "canonical_name": "National Unity Government",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # People's Defence Force — location-specific variants handled by regex pattern below
    "People's Defence Force (PDF)": {
        "canonical_id": "PDF",
        "canonical_name": "People's Defence Force",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "People's Defense Force": {
        "canonical_id": "PDF",
        "canonical_name": "People's Defence Force",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # KNU / Karen National Liberation Army
    "KNU/KNLA: Karen National Union/Karen National Liberation Army": {
        "canonical_id": "KNU",
        "canonical_name": "Karen National Union / Karen National Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KNU (Karen National Union)": {
        "canonical_id": "KNU",
        "canonical_name": "Karen National Union / Karen National Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KNU": {
        "canonical_id": "KNU",
        "canonical_name": "Karen National Union / Karen National Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Karen National Liberation Army": {
        "canonical_id": "KNU",
        "canonical_name": "Karen National Union / Karen National Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # DKBA-5 / Benevolent — NUG-aligned (distinct from original DKBA above)
    "DKBA (Benevolent): Democratic Karen Benevolent Army (2010-)": {
        "canonical_id": "DKBA5",
        "canonical_name": "Democratic Karen Benevolent Army (DKBA-5)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "DKBA 5 (Democratic Karen Buddhist Army faction)": {
        "canonical_id": "DKBA5",
        "canonical_name": "Democratic Karen Benevolent Army (DKBA-5)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "DKBA 5": {
        "canonical_id": "DKBA5",
        "canonical_name": "Democratic Karen Benevolent Army (DKBA-5)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Democratic Karen Benevolent Army (DKBA-5)": {
        "canonical_id": "DKBA5",
        "canonical_name": "Democratic Karen Benevolent Army (DKBA-5)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # KIA / Kachin Independence Army
    "KIO/KIA: Kachin Independence Organization/Kachin Independence Army": {
        "canonical_id": "KIA",
        "canonical_name": "Kachin Independence Army (Kachin Independence Organization)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KIO (Kachin Independence Organization)": {
        "canonical_id": "KIA",
        "canonical_name": "Kachin Independence Army (Kachin Independence Organization)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KIO": {
        "canonical_id": "KIA",
        "canonical_name": "Kachin Independence Army (Kachin Independence Organization)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Kachin Independence Army (KIA)": {
        "canonical_id": "KIA",
        "canonical_name": "Kachin Independence Army (Kachin Independence Organization)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Karenni / KNPP
    "KNPP/KA: Karenni National Progressive Party/Karenni Army": {
        "canonical_id": "KNPP",
        "canonical_name": "Karenni National Progressive Party / Karenni Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KNPP (Karenni National Progressive Party)": {
        "canonical_id": "KNPP",
        "canonical_name": "Karenni National Progressive Party / Karenni Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "KNPP": {
        "canonical_id": "KNPP",
        "canonical_name": "Karenni National Progressive Party / Karenni Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "4K Coalition": {
        "canonical_id": "4K_COALITION",
        "canonical_name": "4K Coalition (Karenni resistance umbrella)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Arakan Army / ULA
    "ULA/AA: United League of Arakan/Arakan Army": {
        "canonical_id": "AA",
        "canonical_name": "Arakan Army (United League of Arakan)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "ULA (United League of Arakan)": {
        "canonical_id": "AA",
        "canonical_name": "Arakan Army (United League of Arakan)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "ULA": {
        "canonical_id": "AA",
        "canonical_name": "Arakan Army (United League of Arakan)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Arakan Army": {
        "canonical_id": "AA",
        "canonical_name": "Arakan Army (United League of Arakan)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Arakan Army (AA)": {
        "canonical_id": "AA",
        "canonical_name": "Arakan Army (United League of Arakan)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # TNLA / Ta'ang National Liberation Army
    "PSLF/TNLA: Palaung State Liberation Front/Ta'ang National Liberation Army": {
        "canonical_id": "TNLA",
        "canonical_name": "Ta'ang National Liberation Army (Palaung State Liberation Front)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "PSLF (Palaung State Liberation Front)": {
        "canonical_id": "TNLA",
        "canonical_name": "Ta'ang National Liberation Army (Palaung State Liberation Front)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "PSLF": {
        "canonical_id": "TNLA",
        "canonical_name": "Ta'ang National Liberation Army (Palaung State Liberation Front)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Ta'ang National Liberation Army (TNLA)": {
        "canonical_id": "TNLA",
        "canonical_name": "Ta'ang National Liberation Army (Palaung State Liberation Front)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # MNDAA / Kokang
    "MNTJP/MNDAA: Myanmar National Truth and Justice Party/Myanmar National Democratic Alliance Army": {
        "canonical_id": "MNDAA",
        "canonical_name": "Myanmar National Democratic Alliance Army (Myanmar National Truth and Justice Party)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "MNDAA (Myanmar National Democratic Alliance Army)": {
        "canonical_id": "MNDAA",
        "canonical_name": "Myanmar National Democratic Alliance Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "MNDAA": {
        "canonical_id": "MNDAA",
        "canonical_name": "Myanmar National Democratic Alliance Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Myanmar National Democratic Alliance Army (MNDAA)": {
        "canonical_id": "MNDAA",
        "canonical_name": "Myanmar National Democratic Alliance Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Alliance coalitions
    "Brotherhood Alliance": {
        "canonical_id": "BROTHERHOOD_ALLIANCE",
        "canonical_name": "Three Brotherhood Alliance (AA + MNDAA + TNLA)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "NA-B: Northern Alliance": {
        "canonical_id": "NORTHERN_ALLIANCE",
        "canonical_name": "Northern Alliance – Burma (KIA + AA + MNDAA + TNLA)",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Northern Alliance": {
        "canonical_id": "NORTHERN_ALLIANCE",
        "canonical_name": "Northern Alliance – Burma",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Chin armed groups
    "CDF: Chinland Defense Force": {
        "canonical_id": "CDF",
        "canonical_name": "Chinland Defense Force",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Chinland Defense Force": {
        "canonical_id": "CDF",
        "canonical_name": "Chinland Defense Force",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "CNF (Chin National Front)": {
        "canonical_id": "CNF",
        "canonical_name": "Chin National Front / Chin National Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "CNF": {
        "canonical_id": "CNF",
        "canonical_name": "Chin National Front / Chin National Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Chin National Army": {
        "canonical_id": "CNA",
        "canonical_name": "Chin National Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Chin National Army (CNA)": {
        "canonical_id": "CNA",
        "canonical_name": "Chin National Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Chin Brotherhood Alliance": {
        "canonical_id": "CHIN_BROTHERHOOD",
        "canonical_name": "Chin Brotherhood Alliance",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Pa-Oh National Liberation Organisation — NUG-aligned
    # NOTE: entirely distinct from PNA/PNO which is SAC-aligned
    "PNLO (Pa-Oh National Liberation Organization)": {
        "canonical_id": "PNLO",
        "canonical_name": "Pa-Oh National Liberation Organisation",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "PNLO": {
        "canonical_id": "PNLO",
        "canonical_name": "Pa-Oh National Liberation Organisation",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Pa-O National Liberation Army (PNLO)": {
        "canonical_id": "PNLO",
        "canonical_name": "Pa-Oh National Liberation Organisation",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Pa-O National Liberation Army": {
        "canonical_id": "PNLO",
        "canonical_name": "Pa-Oh National Liberation Organisation",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # ABSDF
    "ABSDF (All Burma Students' Democratic Front)": {
        "canonical_id": "ABSDF",
        "canonical_name": "All Burma Students' Democratic Front",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "ABSDF": {
        "canonical_id": "ABSDF",
        "canonical_name": "All Burma Students' Democratic Front",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },

    # Regional resistance forces
    "AFA: Ayeyarwaddy Federal Army": {
        "canonical_id": "AFA",
        "canonical_name": "Ayeyarwaddy Federal Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "BLA: Bago Liberation Army": {
        "canonical_id": "BLA",
        "canonical_name": "Bago Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "ANF: Ayeyarwady National Force": {
        "canonical_id": "ANF",
        "canonical_name": "Ayeyarwady National Force",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "PRA: People's Revolution Army": {
        "canonical_id": "PRA",
        "canonical_name": "People's Revolution Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
    },
    "Mon Liberation Army": {
        "canonical_id": "MLA",
        "canonical_name": "Mon Liberation Army",
        "broad_category": "Resistance",
        "alliance": "NUG",
        # NOTE: distinct from NMSP/MNLA (New Mon State Party) which is coded Autonomous
    },
    "Danu anti-junta forces": {
        "canonical_id": "DANU_RESISTANCE",
        "canonical_name": "Danu anti-junta forces",
        "broad_category": "Resistance",
        "alliance": "NUG",
        # TODO: verify — Danu is an ethnic group in Shan State; group identity may vary
    },
    "APA: Ayadaw People's Alliance": {
        "canonical_id": "APA",
        "canonical_name": "Ayadaw People's Alliance",
        "broad_category": "Resistance",
        "alliance": "NUG",
        # TODO: verify NUG alignment; assumed from context as local resistance group
    },
    "KRDA: Kalay Region Defense Association": {
        "canonical_id": "KRDA",
        "canonical_name": "Kalay Region Defense Association",
        "broad_category": "Resistance",
        "alliance": "NUG",
        # TODO: verify NUG alignment; assumed from context
    },

    # -----------------------------------------------------------------------
    # MYANMAR — Autonomous / NUG-noncombatant
    # -----------------------------------------------------------------------

    # SSA-North / SSPP
    "SSPP/SSA-N: Shan State Progress Party/Shan State Army-North": {
        "canonical_id": "SSA_N",
        "canonical_name": "Shan State Army – North (Shan State Progress Party)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "SSPP (Shan State Progress Party)": {
        "canonical_id": "SSA_N",
        "canonical_name": "Shan State Army – North (Shan State Progress Party)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "SSPP": {
        "canonical_id": "SSA_N",
        "canonical_name": "Shan State Army – North (Shan State Progress Party)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "Shan State Army – North (SSA-N / SSPP)": {
        "canonical_id": "SSA_N",
        "canonical_name": "Shan State Army – North (Shan State Progress Party)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },

    # SSA-South / RCSS
    "RCSS/SSA-S: Restoration Council of Shan State/Shan State Army-South": {
        "canonical_id": "SSA_S",
        "canonical_name": "Shan State Army – South (Restoration Council of Shan State)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "RCSS (Restoration Council of Shan State)": {
        "canonical_id": "SSA_S",
        "canonical_name": "Shan State Army – South (Restoration Council of Shan State)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "RCSS": {
        "canonical_id": "SSA_S",
        "canonical_name": "Shan State Army – South (Restoration Council of Shan State)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "Shan State Army – South (SSA-S / RCSS)": {
        "canonical_id": "SSA_S",
        "canonical_name": "Shan State Army – South (Restoration Council of Shan State)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },

    # NDAA (Mong La Group)
    "NDAA (National Democratic Alliance Army)": {
        "canonical_id": "NDAA",
        "canonical_name": "National Democratic Alliance Army (Mong La Group)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "NDAA": {
        "canonical_id": "NDAA",
        "canonical_name": "National Democratic Alliance Army (Mong La Group)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "National Democratic Alliance Army (NDAA)": {
        "canonical_id": "NDAA",
        "canonical_name": "National Democratic Alliance Army (Mong La Group)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "National Democratic Alliance Army": {
        "canonical_id": "NDAA",
        "canonical_name": "National Democratic Alliance Army (Mong La Group)",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },

    # NMSP / New Mon State Party — ceasefire group; largely neutral post-2021
    "NMSP/MNLA: New Mon State Party/Mon National Liberation Army": {
        "canonical_id": "NMSP",
        "canonical_name": "New Mon State Party / Mon National Liberation Army",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
        # TODO: verify post-2021 stance; NMSP signed NCA but has not clearly sided with SAC or NUG
    },
    "NMSP (New Mon State Party)": {
        "canonical_id": "NMSP",
        "canonical_name": "New Mon State Party / Mon National Liberation Army",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },
    "NMSP": {
        "canonical_id": "NMSP",
        "canonical_name": "New Mon State Party / Mon National Liberation Army",
        "broad_category": "Autonomous",
        "alliance": "NUG-noncombatant",
    },

    # -----------------------------------------------------------------------
    # MYANMAR — Other / Independent (not clearly aligned with either side)
    # -----------------------------------------------------------------------

    "UWSP/UWSA: United Wa State Party/United Wa State Army": {
        "canonical_id": "UWSA",
        "canonical_name": "United Wa State Army (United Wa State Party)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "UWSA (United Wa State Army)": {
        "canonical_id": "UWSA",
        "canonical_name": "United Wa State Army (United Wa State Party)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "UWSA": {
        "canonical_id": "UWSA",
        "canonical_name": "United Wa State Army (United Wa State Party)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "United Wa State Army (UWSA)": {
        "canonical_id": "UWSA",
        "canonical_name": "United Wa State Army (United Wa State Party)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "United Wa State Army": {
        "canonical_id": "UWSA",
        "canonical_name": "United Wa State Army (United Wa State Party)",
        "broad_category": "Other",
        "alliance": "Independent",
    },

    # ARSA (Rohingya Islamist militant group)
    "ARSA: Arakan Rohingya Salvation Army": {
        "canonical_id": "ARSA",
        "canonical_name": "Arakan Rohingya Salvation Army",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "ARSA (Arakan Rohingya Salvation Army)": {
        "canonical_id": "ARSA",
        "canonical_name": "Arakan Rohingya Salvation Army",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "ARSA": {
        "canonical_id": "ARSA",
        "canonical_name": "Arakan Rohingya Salvation Army",
        "broad_category": "Other",
        "alliance": "Independent",
    },

    "United National Liberation Front of Western South East Asia": {
        "canonical_id": "UNLF_WSA",
        "canonical_name": "United National Liberation Front of Western South East Asia",
        "broad_category": "Other",
        "alliance": "Independent",
        # TODO: verify current operational status and alliance
    },

    # Mon National Liberation Army — distinct from Mon Liberation Army (NUG)
    "Mon National Liberation Army (MNLA)": {
        "canonical_id": "MNLA",
        "canonical_name": "Mon National Liberation Army",
        "broad_category": "Other",
        "alliance": "Independent",
        # TODO: verify distinction from NUG-aligned Mon Liberation Army
    },

    # RSO (older Rohingya group, largely historical)
    "RSO (Rohingya Solidarity Organization)": {
        "canonical_id": "RSO",
        "canonical_name": "Rohingya Solidarity Organization",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "RSO": {
        "canonical_id": "RSO",
        "canonical_name": "Rohingya Solidarity Organization",
        "broad_category": "Other",
        "alliance": "Independent",
    },

    # Foreign-based armed groups operating in/near Myanmar
    "ULFA-I: United Liberation Front of Asom-Independent": {
        "canonical_id": "ULFA_I",
        "canonical_name": "United Liberation Front of Asom – Independent (India)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "ULFA-I": {
        "canonical_id": "ULFA_I",
        "canonical_name": "United Liberation Front of Asom – Independent (India)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "Unidentified Armed Group (India)": {
        "canonical_id": "UNKNOWN_IND",
        "canonical_name": "Unidentified Armed Group (India)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "NSCN-K (National Socialist Council of Nagaland - Khaplang)": {
        "canonical_id": "NSCN_K",
        "canonical_name": "National Socialist Council of Nagaland – Khaplang",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "NSCN-K": {
        "canonical_id": "NSCN_K",
        "canonical_name": "National Socialist Council of Nagaland – Khaplang",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "PLA: People's Liberation Army of Manipur": {
        "canonical_id": "PLA_MANIPUR",
        "canonical_name": "People's Liberation Army of Manipur (India)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    # Historical / minor groups
    "God's Army": {
        "canonical_id": "GODS_ARMY",
        "canonical_name": "God's Army (Karen millenarian group, historical)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "MTA": {
        "canonical_id": "MTA",
        "canonical_name": "Mong Tai Army (Khun Sa, historical — surrendered 1996)",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "MDA (Myanmar Democratic Alliance)": {
        "canonical_id": "MDA",
        "canonical_name": "Myanmar Democratic Alliance",
        "broad_category": "Other",
        "alliance": "Independent",
        # TODO: identify exact MDA faction in UCDP records
    },
    "MDA - LM (Myanmar Democratic Alliance faction)": {
        "canonical_id": "MDA",
        "canonical_name": "Myanmar Democratic Alliance",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "MDA": {
        "canonical_id": "MDA",
        "canonical_name": "Myanmar Democratic Alliance",
        "broad_category": "Other",
        "alliance": "Independent",
    },
    "BMA": {
        "canonical_id": "BMA",
        "canonical_name": "BMA (Myanmar armed group — unidentified in UCDP)",
        "broad_category": "Unknown",
        "alliance": "None",
        # TODO: identify this UCDP actor
    },

    # -----------------------------------------------------------------------
    # MYANMAR — Civilians
    # -----------------------------------------------------------------------

    "Civilians (Myanmar)": {
        "canonical_id": "CIVILIANS",
        "canonical_name": "Civilians",
        "broad_category": "Civilians",
        "alliance": "Civilian",
    },
    "Civilians": {
        "canonical_id": "CIVILIANS",
        "canonical_name": "Civilians",
        "broad_category": "Civilians",
        "alliance": "Civilian",
    },
    "Protesters (Myanmar)": {
        "canonical_id": "PROTESTERS",
        "canonical_name": "Protesters",
        "broad_category": "Civilians",
        "alliance": "Civilian",
    },
    "Rioters (Myanmar)": {
        "canonical_id": "RIOTERS",
        "canonical_name": "Rioters",
        "broad_category": "Civilians",
        "alliance": "Civilian",
    },
    "Buddhists (Myanmar)": {
        "canonical_id": "COMMUNAL_BUDDHISTS",
        "canonical_name": "Buddhist communal actors (Myanmar)",
        "broad_category": "Civilians",
        "alliance": "None",
    },
    "Muslims (Myanmar)": {
        "canonical_id": "COMMUNAL_MUSLIMS",
        "canonical_name": "Muslim communal actors (Myanmar)",
        "broad_category": "Civilians",
        "alliance": "None",
    },

    # -----------------------------------------------------------------------
    # MYANMAR — Unknown / Unidentified
    # -----------------------------------------------------------------------

    "Unidentified Armed Group (Myanmar)": {
        "canonical_id": "UNKNOWN_MMR",
        "canonical_name": "Unidentified Armed Group (Myanmar)",
        "broad_category": "Unknown",
        "alliance": "None",
    },
    "Unidentified Armed Group (Bangladesh)": {
        "canonical_id": "UNKNOWN_BGD",
        "canonical_name": "Unidentified Armed Group (Bangladesh)",
        "broad_category": "Unknown",
        "alliance": "None",
    },
    "Unidentified Communal Militia (Myanmar)": {
        "canonical_id": "UNKNOWN_COMMUNAL",
        "canonical_name": "Unidentified Communal Militia (Myanmar)",
        "broad_category": "Unknown",
        "alliance": "None",
    },
    "Rakhine Ethnic Militia (Myanmar)": {
        "canonical_id": "RAKHINE_MILITIA",
        "canonical_name": "Rakhine Ethnic Militia",
        "broad_category": "Unknown",
        "alliance": "None",
        # TODO: verify — may refer to Arakan Liberation Army or other Rakhine groups
    },
    "Rohingya Muslim Militia (Myanmar)": {
        "canonical_id": "ROHINGYA_MILITIA",
        "canonical_name": "Rohingya Muslim Militia",
        "broad_category": "Unknown",
        "alliance": "None",
    },
    "Private Security Forces (Myanmar)": {
        "canonical_id": "PRIVATE_SECURITY",
        "canonical_name": "Private Security Forces (Myanmar)",
        "broad_category": "Unknown",
        "alliance": "None",
    },
    "TSTF: Thongwa Special Task Force": {
        "canonical_id": "TSTF",
        "canonical_name": "Thongwa Special Task Force",
        "broad_category": "Unknown",
        "alliance": "None",
        # TODO: verify — may be NUG-aligned local resistance force
    },
    "King Cobra-Khin U": {
        "canonical_id": "KING_COBRA",
        "canonical_name": "King Cobra – Khin U",
        "broad_category": "Unknown",
        "alliance": "None",
        # TODO: verify — local armed group in Sagaing
    },
    "Phaung Daing Phaung Daing Group": {
        "canonical_id": "PHAUNG_DAING",
        "canonical_name": "Phaung Daing Group",
        "broad_category": "Unknown",
        "alliance": "None",
        # TODO: verify identity and alliance
    },

    # -----------------------------------------------------------------------
    # SOMALIA — placeholder for future onboarding
    # "Somali National Army": { ... }
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # NIGERIA — placeholder for future onboarding
    # "Military Forces of Nigeria (2015-)": { ... }
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # ECUADOR — placeholder for future onboarding
    # "Military Forces of Ecuador (2023-)": { ... }
    # -----------------------------------------------------------------------
}


# ===========================================================================
# ACTOR PATTERNS — regex fallback (applied after exact-match fails)
#
# Order matters: first match wins. Keep more-specific patterns before generic.
# Each entry: (compiled_regex, canonical_dict)
# ===========================================================================

ACTOR_PATTERNS: list[tuple[re.Pattern, dict]] = [
    # All People's Defence/Defense Force location variants → PDF
    (
        re.compile(r"people'?s\s+de[fe]en[cs]e\s+force", re.IGNORECASE),
        {
            "canonical_id": "PDF",
            "canonical_name": "People's Defence Force",
            "broad_category": "Resistance",
            "alliance": "NUG",
        },
    ),
    # All Military Forces of Myanmar variants (catches sub-units not individually listed)
    (
        re.compile(r"military\s+forces?\s+of\s+myanmar", re.IGNORECASE),
        {
            "canonical_id": "TATMADAW",
            "canonical_name": "Tatmadaw (Myanmar Armed Forces)",
            "broad_category": "Junta",
            "alliance": "SAC",
        },
    ),
    # All Government of Myanmar variants
    (
        re.compile(r"government\s+of\s+myanmar", re.IGNORECASE),
        {
            "canonical_id": "GOV_MYANMAR",
            "canonical_name": "Government of Myanmar",
            "broad_category": "Junta",
            "alliance": "SAC",
        },
    ),
    # All Police Forces of Myanmar variants
    (
        re.compile(r"police\s+forces?\s+of\s+myanmar", re.IGNORECASE),
        {
            "canonical_id": "POLICE_MYANMAR",
            "canonical_name": "Myanmar Police Force",
            "broad_category": "Junta",
            "alliance": "SAC",
        },
    ),
    # Border Guard Force
    (
        re.compile(r"border\s+guard\s+force", re.IGNORECASE),
        {
            "canonical_id": "BGF",
            "canonical_name": "Border Guard Force",
            "broad_category": "Junta",
            "alliance": "SAC",
        },
    ),
]


# ===========================================================================
# FALLBACK — returned when no match found anywhere
# ===========================================================================

_UNKNOWN_ACTOR: dict = {
    "canonical_id": "UNKNOWN",
    "canonical_name": "Unknown / Unidentified",
    "broad_category": "Unknown",
    "alliance": "None",
}


# ===========================================================================
# REGION MAPS — admin level 1
#
# Keys are raw strings from ACLED / UCDP / other sources.
# Values are GADM-canonical admin1 names for Myanmar.
# Structure: {country_iso3: {raw_spelling: canonical_gadm_name}}
# ===========================================================================

_REGION_L1_MYANMAR: dict[str, str] = {
    # --- ACLED spellings ---
    "Ayeyarwady": "Ayeyarwady",
    "Bago": "Bago",
    "Bago-East": "Bago",
    "Bago-West": "Bago",
    "Chin": "Chin",
    "Kachin": "Kachin",
    "Kayah": "Kayah",
    "Kayin": "Kayin",
    "Magway": "Magway",
    "Mandalay": "Mandalay",
    "Mon": "Mon",
    "Nay Pyi Taw": "Naypyidaw",
    "Nay Pyi Taw Union Territory": "Naypyidaw",
    "Naypyidaw": "Naypyidaw",
    "Rakhine": "Rakhine",
    "Sagaing": "Sagaing",
    "Shan": "Shan",
    "Shan-East": "Shan",
    "Shan-North": "Shan",
    "Shan-South": "Shan",
    "Tanintharyi": "Tanintharyi",
    "Yangon": "Yangon",
    # --- UCDP spellings (old division/state/region suffixes) ---
    "Ayeyarwady division": "Ayeyarwady",
    "Ayeyarwady region": "Ayeyarwady",
    "Bago division": "Bago",
    "Bago region": "Bago",
    "Chin state": "Chin",
    "Kachin state": "Kachin",
    "Karen state": "Kayin",             # Colonial-era name for Kayin State
    "Kayah state": "Kayah",
    "Kayin state": "Kayin",
    "Magway region": "Magway",
    "Magwe division": "Magway",         # Old spelling still in UCDP
    "Magwe region": "Magway",
    "Mandalay division": "Mandalay",
    "Mandalay region": "Mandalay",
    "Mon state": "Mon",
    "Naypyidaw Union territory": "Naypyidaw",
    "Naypyidaw union territory": "Naypyidaw",
    "Rakhine state": "Rakhine",
    "Sagaing district": "Sagaing",
    "Sagaing division": "Sagaing",
    "Sagaing region": "Sagaing",
    "Shan state": "Shan",
    "Tanintharyi division": "Tanintharyi",
    "Tanintharyi region": "Tanintharyi",
    "Yangon division": "Yangon",
    "Yangon region": "Yangon",
    # --- Alternative/colonial spellings ---
    "Arakan state": "Rakhine",          # Colonial name
    "Irrawaddy": "Ayeyarwady",          # Colonial name
    "Irrawaddy division": "Ayeyarwady",
    "Rangoon division": "Yangon",       # Colonial name
    "Rangoon region": "Yangon",
    "Tenasserim division": "Tanintharyi",
    "Pegu division": "Bago",
    "Karenni state": "Kayah",           # Former name
}

REGION_MAP_L1: dict[str, dict[str, str]] = {
    "MMR": _REGION_L1_MYANMAR,
    # "SOM": _REGION_L1_SOMALIA,   # TODO: add when Somalia is onboarded
    # "NGA": _REGION_L1_NIGERIA,   # TODO
    # "ECU": _REGION_L1_ECUADOR,   # TODO
}


# ===========================================================================
# REGION MAPS — admin level 2
#
# High-cardinality; populate on demand from GADM level-2 boundary file.
# Existing entries fix known UCDP/ACLED variant spellings.
# ===========================================================================

_REGION_L2_MYANMAR: dict[str, str] = {
    # UCDP variant → GADM canonical
    "Dawai": "Dawei",
    "Dawai-Myeik": "Dawei",
    "Hpa-an": "Hpa-An",
    "Kengtong": "Kengtung",
    "Mawlamyine": "Mawlamyaing",
    "Mergui": "Myeik",
    # TODO: expand using data/raw/gadm/ GADM level-2 boundary file for Myanmar
}

REGION_MAP_L2: dict[str, dict[str, str]] = {
    "MMR": _REGION_L2_MYANMAR,
    # "SOM": _REGION_L2_SOMALIA,   # TODO
    # "NGA": _REGION_L2_NIGERIA,   # TODO
    # "ECU": _REGION_L2_ECUADOR,   # TODO
}


# ===========================================================================
# SOURCE → COLUMN MAP
#
# Maps the source identifier to the raw column names for actor A, actor B,
# and admin1. Extend when adding new data sources.
# ===========================================================================

_SOURCE_COLUMNS: dict[str, dict[str, Optional[str]]] = {
    "acled": {
        "actor_a": "actor1",
        "actor_b": "actor2",
        "admin1":  "admin1",
    },
    "ucdp": {
        "actor_a": "side_a",
        "actor_b": "side_b",
        "admin1":  "adm_1",
    },
    "wikipedia": {
        "actor_a": "control_actor",  # as used in myanmar_control_labels_hexgrid.csv
        "actor_b": None,
        "admin1":  None,
    },
}


# ===========================================================================
# PUBLIC API
# ===========================================================================

def normalize_actor(name: str) -> dict:
    """Return canonical actor information for a raw actor name string.

    Resolution order:
      1. Exact string match in ACTOR_REGISTRY
      2. Case-insensitive exact match in ACTOR_REGISTRY
      3. First regex match in ACTOR_PATTERNS
      4. Fallback: Unknown category

    Parameters
    ----------
    name : str
        Raw actor name as it appears in ACLED, UCDP, or Wikipedia.

    Returns
    -------
    dict with keys:
        canonical_id, canonical_name, broad_category, alliance
    Never raises — returns Unknown category if no match found.
    """
    if not name or not isinstance(name, str):
        return dict(_UNKNOWN_ACTOR)

    name = name.strip()

    # 1. Exact match
    if name in ACTOR_REGISTRY:
        return dict(ACTOR_REGISTRY[name])

    # 2. Case-insensitive exact match
    name_lower = name.lower()
    for key, value in ACTOR_REGISTRY.items():
        if key.lower() == name_lower:
            return dict(value)

    # 3. Regex pattern match
    for pattern, canonical in ACTOR_PATTERNS:
        if pattern.search(name):
            return dict(canonical)

    return dict(_UNKNOWN_ACTOR)


def normalize_region(name: str, level: int = 1, country: str = "MMR") -> str:
    """Return the GADM-canonical region name for a raw admin region string.

    Parameters
    ----------
    name : str
        Raw region name from ACLED, UCDP, or other source.
    level : int
        Administrative level: 1 (state/region) or 2 (district/township).
    country : str
        ISO-3166-1 alpha-3 country code. Default "MMR" (Myanmar).

    Returns
    -------
    str
        GADM canonical name, or the original string if not found.
    Never raises.
    """
    if not name or not isinstance(name, str):
        return name

    name = name.strip()
    region_map = (REGION_MAP_L1 if level == 1 else REGION_MAP_L2).get(country, {})

    # Exact match
    if name in region_map:
        return region_map[name]

    # Case-insensitive match
    name_lower = name.lower()
    for raw, canonical in region_map.items():
        if raw.lower() == name_lower:
            return canonical

    return name  # Return original if no mapping found


def get_broad_category(actor_name: str) -> str:
    """Return just the broad category for an actor name.

    Convenience wrapper around normalize_actor.

    Parameters
    ----------
    actor_name : str
        Raw actor name.

    Returns
    -------
    str
        One of: Junta, Resistance, Autonomous, Other, Contested, Civilians, Unknown.
    """
    return normalize_actor(actor_name)["broad_category"]


def normalize_dataframe(
    df: pd.DataFrame,
    source: str,
    country: str = "MMR",
) -> pd.DataFrame:
    """Add canonical columns to a conflict event DataFrame.

    Works non-destructively: original columns are preserved and new columns
    are appended. Returns a copy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw conflict event data from ACLED, UCDP, or Wikipedia.
    source : str
        Data source identifier. One of: 'acled', 'ucdp', 'wikipedia'.
    country : str
        ISO-3166-1 alpha-3 country code for region normalization. Default "MMR".

    Returns
    -------
    pd.DataFrame
        Copy of df with additional columns:
          - canonical_actor_a  : canonical_id for primary actor
          - broad_category_a   : broad category for primary actor
          - canonical_actor_b  : canonical_id for secondary actor (if available)
          - broad_category_b   : broad category for secondary actor (if available)
          - canonical_admin1   : GADM-canonical admin1 name (if available)

    Raises
    ------
    ValueError
        If source is not a recognised source identifier.
    """
    if source not in _SOURCE_COLUMNS:
        raise ValueError(
            f"Unknown source '{source}'. Must be one of: {list(_SOURCE_COLUMNS)}"
        )

    df = df.copy()
    cols = _SOURCE_COLUMNS[source]

    col_a = cols.get("actor_a")
    if col_a and col_a in df.columns:
        _info_a = df[col_a].apply(normalize_actor)
        df["canonical_actor_a"] = _info_a.apply(lambda x: x["canonical_id"])
        df["broad_category_a"]  = _info_a.apply(lambda x: x["broad_category"])

    col_b = cols.get("actor_b")
    if col_b and col_b in df.columns:
        _info_b = df[col_b].apply(normalize_actor)
        df["canonical_actor_b"] = _info_b.apply(lambda x: x["canonical_id"])
        df["broad_category_b"]  = _info_b.apply(lambda x: x["broad_category"])

    col_admin1 = cols.get("admin1")
    if col_admin1 and col_admin1 in df.columns:
        df["canonical_admin1"] = df[col_admin1].apply(
            lambda x: normalize_region(x, level=1, country=country)
            if pd.notna(x)
            else x
        )

    return df


# ===========================================================================
# VALIDATION — run as script
# ===========================================================================

if __name__ == "__main__":
    # Representative sample actors from each source
    _ACLED_ACTORS = [
        "Military Forces of Myanmar (2021-)",
        "Military Forces of Myanmar (2021-) Border Guard Force",
        "Military Forces of Myanmar (2021-) People's Militia Force",
        "Police Forces of Myanmar (2021-)",
        "Government of Myanmar (2021-) State Administration Council",
        "Pyu Saw Htee",
        "PNO/PNA: Pa-Oh National Organization/Pa-Oh National Army",
        "ZRA: Zomi Revolutionary Army",
        "KPC: KNU/KNLA Peace Council",
        "DKBA (Buddhist): Democratic Karen Buddhist Army (2016-)",
        "People's Defense Force - Pathein",
        "People's Defense Force - Mandalay",
        "People's Defense Force - Hkamti",
        "KNU/KNLA: Karen National Union/Karen National Liberation Army",
        "DKBA (Benevolent): Democratic Karen Benevolent Army (2010-)",
        "KIO/KIA: Kachin Independence Organization/Kachin Independence Army",
        "KNPP/KA: Karenni National Progressive Party/Karenni Army",
        "ULA/AA: United League of Arakan/Arakan Army",
        "PSLF/TNLA: Palaung State Liberation Front/Ta'ang National Liberation Army",
        "MNTJP/MNDAA: Myanmar National Truth and Justice Party/Myanmar National Democratic Alliance Army",
        "Brotherhood Alliance",
        "NA-B: Northern Alliance",
        "CDF: Chinland Defense Force",
        "AFA: Ayeyarwaddy Federal Army",
        "BLA: Bago Liberation Army",
        "ANF: Ayeyarwady National Force",
        "SSPP/SSA-N: Shan State Progress Party/Shan State Army-North",
        "RCSS/SSA-S: Restoration Council of Shan State/Shan State Army-South",
        "NMSP/MNLA: New Mon State Party/Mon National Liberation Army",
        "UWSP/UWSA: United Wa State Party/United Wa State Army",
        "ARSA: Arakan Rohingya Salvation Army",
        "ULFA-I: United Liberation Front of Asom-Independent",
        "Civilians (Myanmar)",
        "Protesters (Myanmar)",
        "Rioters (Myanmar)",
        "Unidentified Armed Group (Myanmar)",
        "Private Security Forces (Myanmar)",
        "Rakhine Ethnic Militia (Myanmar)",
    ]

    _UCDP_ACTORS = [
        "Government of Myanmar (Burma)",
        "KNU (Karen National Union)",
        "ULA (United League of Arakan)",
        "DKBA (Democratic Karen Buddhist Army)",
        "RCSS (Restoration Council of Shan State)",
        "NUG (National Unity Government)",
        "DKBA 5 (Democratic Karen Buddhist Army faction)",
        "KIO (Kachin Independence Organization)",
        "KNPP (Karenni National Progressive Party)",
        "UWSA (United Wa State Army)",
        "MNDAA (Myanmar National Democratic Alliance Army)",
        "PSLF (Palaung State Liberation Front)",
        "SSPP (Shan State Progress Party)",
        "CNF (Chin National Front)",
        "PNLO (Pa-Oh National Liberation Organization)",
        "ARSA (Arakan Rohingya Salvation Army)",
        "ABSDF (All Burma Students' Democratic Front)",
        "RSO (Rohingya Solidarity Organization)",
        "NSCN-K (National Socialist Council of Nagaland - Khaplang)",
        "God's Army",
        "MDA (Myanmar Democratic Alliance)",
        "BMA",
        "MTA",
        "Civilians",
        "Buddhists (Myanmar)",
        "Muslims (Myanmar)",
    ]

    _WIKI_ACTORS = [
        # SAC alliance
        "Tatmadaw",
        "Karen National Army",
        "Pa-O National Army",
        "Zomi Revolutionary Army",
        # NUG alliance
        "People's Defence Force (PDF)",
        "Kachin Independence Army (KIA)",
        "Karen National Liberation Army",
        "Democratic Karen Benevolent Army (DKBA-5)",
        "4K Coalition",
        "Arakan Army (AA)",
        "Ta'ang National Liberation Army (TNLA)",
        "Myanmar National Democratic Alliance Army (MNDAA)",
        "Chin National Army (CNA)",
        "Chin Brotherhood Alliance",
        "Pa-O National Liberation Army (PNLO)",
        "Mon Liberation Army",
        "Danu anti-junta forces",
        # Autonomous
        "Shan State Army – North (SSA-N / SSPP)",
        "Shan State Army – South (SSA-S / RCSS)",
        "National Democratic Alliance Army (NDAA)",
        # Other
        "United Wa State Army (UWSA)",
        "United National Liberation Front of Western South East Asia",
        "Mon National Liberation Army (MNLA)",
    ]

    # Print actor validation table
    col_w = {"source": 12, "raw": 70, "id": 25, "cat": 15, "alliance": 20}
    header = (
        f"{'Source':<{col_w['source']}}"
        f"{'Raw Name':<{col_w['raw']}}"
        f"{'Canonical ID':<{col_w['id']}}"
        f"{'Category':<{col_w['cat']}}"
        f"Alliance"
    )
    separator = "-" * (sum(col_w.values()) + 8)

    print("\n=== ACTOR MAPPING VALIDATION ===\n")
    print(header)
    print(separator)

    for source_label, actors in [
        ("ACLED", _ACLED_ACTORS),
        ("UCDP", _UCDP_ACTORS),
        ("Wikipedia", _WIKI_ACTORS),
    ]:
        for actor in actors:
            info = normalize_actor(actor)
            flag = "  [UNMAPPED]" if info["canonical_id"] == "UNKNOWN" else ""
            print(
                f"{source_label:<{col_w['source']}}"
                f"{actor[:col_w['raw'] - 2]:<{col_w['raw']}}"
                f"{info['canonical_id']:<{col_w['id']}}"
                f"{info['broad_category']:<{col_w['cat']}}"
                f"{info['alliance']}{flag}"
            )
        print()

    # Region validation
    print("\n=== REGION MAPPING VALIDATION ===\n")
    _REGION_TESTS = [
        ("ACLED", "Bago-East"),
        ("ACLED", "Bago-West"),
        ("ACLED", "Shan-North"),
        ("ACLED", "Shan-South"),
        ("ACLED", "Shan-East"),
        ("ACLED", "Nay Pyi Taw"),
        ("ACLED", "Kayah"),
        ("ACLED", "Kayin"),
        ("UCDP",  "Ayeyarwady division"),
        ("UCDP",  "Ayeyarwady region"),
        ("UCDP",  "Bago region"),
        ("UCDP",  "Karen state"),
        ("UCDP",  "Magwe division"),
        ("UCDP",  "Sagaing district"),
        ("UCDP",  "Shan state"),
        ("UCDP",  "Tanintharyi region"),
        ("UCDP",  "Naypyidaw Union territory"),
        ("UCDP",  "Kachin state"),
        ("UCDP",  "Chin state"),
        ("ALT",   "Karen state"),
        ("ALT",   "Irrawaddy"),
        ("ALT",   "Pegu division"),
        ("ALT",   "Tenasserim division"),
        ("NOVEL", "Unrecognised Region XYZ"),
    ]

    print(f"{'Source':<10} {'Raw Name':<35} {'Canonical'}")
    print("-" * 65)
    _l1_mmr = REGION_MAP_L1.get("MMR", {})
    for src, region in _REGION_TESTS:
        canonical = normalize_region(region)
        if region not in _l1_mmr and region.lower() not in {k.lower() for k in _l1_mmr}:
            flag = "  [no mapping — returned as-is]"
        else:
            flag = ""
        print(f"{src:<10} {region:<35} {canonical}{flag}")
