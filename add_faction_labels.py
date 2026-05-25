#!/usr/bin/env python3
"""
Add faction hierarchy columns to both the Wikipedia panel dataset
and the UCDP events dataset so they speak the same language.

Usage:
    python add_faction_labels.py

Input:
    extracting_ground_truth/panel_dataset.csv
    ucdp/GEDEvent_myanmar_merged.csv          (adjust path if needed)

Output:
    extracting_ground_truth/panel_dataset_labelled.csv
    ucdp/ucdp_labelled.csv

Hierarchy:
    Level 3 (wiki_label)  : direct faction name e.g. KIA, AA, PDF
    Level 2 (level2_group): Junta / Pro-democracy / Ethnic armed / Ceasefire / Other
    Level 1 (level1_side) : Pro-Junta / Anti-Junta / Other
"""

import os
import pandas as pd

# ── Crosswalk tables ──────────────────────────────────────────────────────

# Wikipedia label -> (level2_group, level1_side)
WIKI_HIERARCHY = {
    'Junta':            ('Junta',          'Pro-Junta'),
    'PDF':              ('Pro-democracy',  'Anti-Junta'),
    'KIA':              ('Ethnic armed',   'Anti-Junta'),
    'AA':               ('Ethnic armed',   'Anti-Junta'),
    'Karen':            ('Ethnic armed',   'Anti-Junta'),
    'Karenni':          ('Ethnic armed',   'Anti-Junta'),
    'TNLA':             ('Ethnic armed',   'Anti-Junta'),
    'MNDAA':            ('Ethnic armed',   'Anti-Junta'),
    'Chin Resistance':  ('Ethnic armed',   'Anti-Junta'),
    'ZRA':              ('Ethnic armed',   'Anti-Junta'),
    'Mon':              ('Ethnic armed',   'Anti-Junta'),
    'Danu':             ('Pro-democracy',  'Anti-Junta'),
    'SSPP':             ('Ceasefire',      'Anti-Junta'),
    'RCSS':             ('Ceasefire',      'Anti-Junta'),
    'NDAA':             ('Ceasefire',      'Anti-Junta'),
    'UWSA':             ('Ceasefire',      'Other'),
    'MNLA':             ('Ethnic armed',   'Anti-Junta'),
}

# UCDP actor name -> wiki_label
# None = no equivalent Wikipedia zone
UCDP_TO_WIKI = {
    'Government of Myanmar (Burma)': 'Junta',
    'NUG':                           'PDF',
    'MNDAA':                         'MNDAA',
    'ULA':                           'AA',
    'KIO':                           'KIA',
    'KNU':                           'Karen',
    'KNPP':                          'Karenni',
    'PSLF':                          'TNLA',
    'RCSS':                          'RCSS',
    'SSPP':                          'SSPP',
    'CNF':                           'Chin Resistance',
    'PNLO':                          'ZRA',
    'ABSDF':                         'PDF',       # allied with NUG/PDF
    'ARSA':                          None,        # Rohingya group, no Wikipedia zone
    'SCEF':                          None,        # tiny Shan group
    'UNLFW':                         None,        # Indian border group
    'Government of India':           None,        # not part of Myanmar conflict
    'Civilians':                     None,        # not an actor
}

def get_wiki_hierarchy(wiki_label):
    """Return (level2_group, level1_side) for a wiki label."""
    if wiki_label is None:
        return ('Other', 'Other')
    return WIKI_HIERARCHY.get(wiki_label, ('Other', 'Other'))


# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PANEL_IN        = os.path.join(BASE_DIR, 'extracting_ground_truth', 'panel_dataset.csv')
PANEL_OUT       = os.path.join(BASE_DIR, 'extracting_ground_truth', 'panel_dataset_labelled.csv')

# Adjust this path to wherever your UCDP file is
UCDP_IN         = os.path.join(BASE_DIR, 'data', 'raw', 'ucdp', 'GEDEvent_myanmar_merged.csv')
UCDP_OUT        = os.path.join(BASE_DIR, 'data', 'raw', 'ucdp', 'ucdp_labelled.csv')


# ── 1. Label Wikipedia panel dataset ─────────────────────────────────────
print("Processing Wikipedia panel dataset...")
panel = pd.read_csv(PANEL_IN)
print(f"  Loaded {len(panel):,} rows")

print(f"  Columns: {panel.columns.tolist()}")
panel['wiki_label']   = panel['faction']
panel['level2_group'] = panel['faction'].map(lambda f: get_wiki_hierarchy(f)[0])
panel['level1_side']  = panel['faction'].map(lambda f: get_wiki_hierarchy(f)[1])

panel.to_csv(PANEL_OUT, index=False)
print(f"  Saved -> {PANEL_OUT}")
print(f"\n  Level 1 distribution:")
print(panel['level1_side'].value_counts().to_string())
print(f"\n  Level 2 distribution:")
print(panel['level2_group'].value_counts().to_string())


# ── 2. Label UCDP dataset ─────────────────────────────────────────────────
print("\nProcessing UCDP dataset...")

if not os.path.exists(UCDP_IN):
    print(f"  ERROR: UCDP file not found at {UCDP_IN}")
    print("  Update the UCDP_IN path in this script and re-run.")
else:
    ucdp = pd.read_csv(UCDP_IN)
    print(f"  Loaded {len(ucdp):,} rows ({ucdp['year'].min()}-{ucdp['year'].max()})")

    # Filter to our time range
    ucdp = ucdp[ucdp['year'] >= 2023].copy()
    print(f"  Filtered to 2023+: {len(ucdp):,} rows")

    # Add labels for side_a
    ucdp['wiki_label_a']   = ucdp['side_a'].map(UCDP_TO_WIKI)
    ucdp['level2_group_a'] = ucdp['side_a'].map(lambda a: get_wiki_hierarchy(UCDP_TO_WIKI.get(a))[0])
    ucdp['level1_side_a']  = ucdp['side_a'].map(lambda a: get_wiki_hierarchy(UCDP_TO_WIKI.get(a))[1])

    # Add labels for side_b
    ucdp['wiki_label_b']   = ucdp['side_b'].map(UCDP_TO_WIKI)
    ucdp['level2_group_b'] = ucdp['side_b'].map(lambda b: get_wiki_hierarchy(UCDP_TO_WIKI.get(b))[0])
    ucdp['level1_side_b']  = ucdp['side_b'].map(lambda b: get_wiki_hierarchy(UCDP_TO_WIKI.get(b))[1])

    ucdp.to_csv(UCDP_OUT, index=False)
    print(f"  Saved -> {UCDP_OUT}")

    print(f"\n  Side A level 1 distribution:")
    print(ucdp['level1_side_a'].value_counts().to_string())
    print(f"\n  Side B level 1 distribution:")
    print(ucdp['level1_side_b'].value_counts().to_string())

    # Check for any unmapped actors
    unmapped_a = ucdp[ucdp['wiki_label_a'].isna()]['side_a'].unique()
    unmapped_b = ucdp[ucdp['wiki_label_b'].isna()]['side_b'].unique()
    all_unmapped = set(unmapped_a) | set(unmapped_b)
    if all_unmapped:
        print(f"\n  Actors with no wiki_label (expected): {sorted(all_unmapped)}")

print("\nDone.")
