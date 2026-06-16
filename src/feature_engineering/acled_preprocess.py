# ============================================================
# acled_preprocess.py
# Build ACLED territorial-control aggregates → cell × month panel
#
# Inputs:
#   data/raw/acled/acled_myanmar_*.csv     (most recent by date in filename)
#   data/processed/myanmar_priogrid_labels.csv
#
# Output:
#   data/processed/acled_wiki_comparison.parquet
#
# Run from project root:
#   uv run python src/feature_engineering/acled_preprocess.py
# ============================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── 1. PATHS ──────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parents[2]
ACLED_DIR = ROOT / 'data' / 'raw' / 'acled'
WIKI_PATH = ROOT / 'data' / 'processed' / 'myanmar_priogrid_labels.csv'
OUT_PATH  = ROOT / 'data' / 'processed' / 'acled_wiki_comparison.parquet'

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 2. RESOLVE MOST RECENT ACLED FILE ─────────────────────────────────────────

_acled_files = list(ACLED_DIR.glob('acled_myanmar_*.csv'))
if not _acled_files:
    raise FileNotFoundError(f'No acled_myanmar_*.csv files found in {ACLED_DIR}')

acled_path = max(
    _acled_files,
    key=lambda f: re.search(r'(\d{4}-\d{2}-\d{2})', f.name).group(1),
)
print(f'Using ACLED file: {acled_path.name}')

# ── 3. CONSTANTS ──────────────────────────────────────────────────────────────

COUP_DATE = pd.Timestamp('2021-02-01')

CONTROL_SIGNAL_TYPES = {
    'Non-state actor overtakes territory',
    'Government regains territory',
    'Non-violent transfer of territory',
    'Headquarters or base established',
}

# Maps raw ACLED actor side → binary gov/opo/unmapped used downstream
SIDE_MAP = {
    'Pro-junta':               'gov',
    'Anti-junta':              'opo',
    'Neutral / Self-administered': 'opo',
    'Other / Unmapped':        'unmapped',
}

# Maps Wikipedia control_status → 3-value side for agreement comparison
WIKI_SIDE_MAP = {
    'government':         'gov',
    'resistance':         'opo',
    'ethnic_armed_group': 'opo',
    'contested':          'contested',
}

# ── 4. ACTOR TAXONOMY ─────────────────────────────────────────────────────────
#
# Rules are evaluated in order; first match wins.
# Each rule maps a list of case-insensitive regexes against actor1.

ACTOR_RULES = [

    # ── ANTI-JUNTA ▸ National Unity Government (PDF, NUG) ────────────────────
    {
        'patterns': [
            r"people.s\s+def",
            r"people\s+def(?:ense|ence)\s+force",
            r"\bpdf\b",
            r"unidentified anti.coup",
            r"people.s security force",
            r"natogyi people",
            r"kani people.s def",
        ],
        'side': 'Anti-junta',
        'coalition': 'National Unity Government (PDF, NUG)',
        'actor_group': "People's Defence Force",
    },

    # ── ANTI-JUNTA ▸ Northern Alliance ───────────────────────────────────────
    {
        'patterns': [r"kio/kia", r"kachin independence"],
        'side': 'Anti-junta',
        'coalition': 'Northern Alliance',
        'actor_group': 'Kachin Independence Army (KIA)',
    },
    {
        'patterns': [r"^na-b:.*northern alliance"],
        'side': 'Anti-junta',
        'coalition': 'Northern Alliance',
        'actor_group': 'Kachin Independence Army (KIA)',
    },

    # ── ANTI-JUNTA ▸ Three Brotherhood Alliance ───────────────────────────────
    {
        'patterns': [r"ula/aa", r"united league.*arakan", r"arakan army"],
        'side': 'Anti-junta',
        'coalition': 'Three Brotherhood Alliance',
        'actor_group': 'Arakan Army (AA)',
    },
    {
        'patterns': [r"pslf/tnla", r"ta.?ang national lib", r"palaung state lib"],
        'side': 'Anti-junta',
        'coalition': 'Three Brotherhood Alliance',
        'actor_group': 'TNLA',
    },
    {
        'patterns': [r"mntjp/mndaa", r"myanmar national.*alliance army",
                     r"myanmar national truth and justice"],
        'side': 'Anti-junta',
        'coalition': 'Three Brotherhood Alliance',
        'actor_group': 'MNDAA',
    },
    {
        'patterns': [r"^brotherhood alliance$"],
        'side': 'Anti-junta',
        'coalition': 'Three Brotherhood Alliance',
        'actor_group': 'Three Brotherhood Alliance (collective)',
    },

    # ── ANTI-JUNTA ▸ 4K Coalition ────────────────────────────────────────────
    {
        'patterns': [
            r"knu/knla", r"karen national union", r"karen national lib(?:eration)? army",
            r"\bkndo\b", r"karen national def.*org",
            r"kaw thoo lei", r"kayin force",
        ],
        'side': 'Anti-junta',
        'coalition': '4K Coalition',
        'actor_group': 'Karen National Liberation Army (KNLA)',
    },
    {
        'patterns': [r"knpp/ka", r"karenni national progressive", r"\bkarenni army\b"],
        'side': 'Anti-junta',
        'coalition': '4K Coalition',
        'actor_group': 'Karenni Army',
    },
    {
        'patterns': [r"\bkndf\b", r"karenni nationalities def"],
        'side': 'Anti-junta',
        'coalition': '4K Coalition',
        'actor_group': 'KNDF',
    },
    {
        'patterns': [r"\bknplf\b", r"karenni nationalities people.s lib"],
        'side': 'Anti-junta',
        'coalition': '4K Coalition',
        'actor_group': 'KNPLF',
    },

    # ── ANTI-JUNTA ▸ Chinland Council ────────────────────────────────────────
    {
        'patterns': [
            r"cnf/cna", r"chin national front",
            r"chinland def(?:ense|ence) force", r"cdf-kkg",
            r"acdf.*asho chin",
        ],
        'side': 'Anti-junta',
        'coalition': 'Chinland Council',
        'actor_group': 'Chinland Council',
    },

    # ── ANTI-JUNTA ▸ Chin Brotherhood Alliance ───────────────────────────────
    {
        'patterns': [
            r"chin brotherhood", r"cno/cndf",
            r"chin national org.*def", r"chin national def.*force",
            r"zomi federal union", r"pdf.*zoland", r"zoland def",
        ],
        'side': 'Anti-junta',
        'coalition': 'Chin Brotherhood Alliance',
        'actor_group': 'Chin Brotherhood Alliance',
    },

    # ── ANTI-JUNTA ▸ Spring Revolution Alliance ───────────────────────────────
    {
        'patterns': [r"pnlo/pnla", r"pa-oh national lib"],
        'side': 'Anti-junta',
        'coalition': 'Spring Revolution Alliance',
        'actor_group': 'PNLA',
    },

    # ── ANTI-JUNTA ▸ Other allied EAOs ───────────────────────────────────────
    {
        'patterns': [
            r"nmsp/mnla", r"nmsp-ad", r"mnla-ad", r"mon national lib",
            r"new mon state party",
            r"\bmsrf\b", r"\bmsru\b", r"mon state rev",
            r"mon state mount", r"ramonnya mon",
        ],
        'side': 'Anti-junta',
        'coalition': 'Other allied EAOs',
        'actor_group': 'Mon National Liberation Army (MNLA)',
    },
    {
        'patterns': [r"dkba.*benev", r"democratic karen benev"],
        'side': 'Anti-junta',
        'coalition': 'Other allied EAOs',
        'actor_group': 'Other allied EAOs',
    },

    # ── ANTI-JUNTA ▸ Other anti-junta organisations ───────────────────────────
    {
        'patterns': [r"\babsdf\b", r"all burma students"],
        'side': 'Anti-junta',
        'coalition': 'Other anti-junta organisations',
        'actor_group': 'Other anti-junta',
    },
    {
        'patterns': [
            r"^pla:.*people.s lib", r"people.s liberation army",
            r"anti.fascist.*(?:armed )?force", r"anti.fascist int",
            r"\bpadf\b.*anti.fascist",
        ],
        'side': 'Anti-junta',
        'coalition': 'Other anti-junta organisations',
        'actor_group': 'Other anti-junta',
    },

    # ── ANTI-JUNTA ▸ Local resistance forces ─────────────────────────────────
    {
        'patterns': [
            r"guerr?illa",
            r"revolution(?:ary)?\s+(?:forces?|army|front|alliance)",
            r"anti.dictatorship",
            r"burma\s+(?:national\s+revolutionary|liberation\s+democratic)",
            r"pyauk\s+kyar",
            r"sit\s+kyaung",
            r"ye\s+bi\s+lu|ye\s+ogre",
            r"myingyan\s+black\s+tiger|\bmbt\b.*myingyan|myingyan.*\bmbt\b",
            r"\bcdsom\b",
            r"kanbalu.*(?:underground|\bug\b)",
            r"27\s+revolution\s+forces",
            r"\bpafd\b|army\s+to\s+fight\s+dictatorship",
            r"yaw\s+defense\s+force",
            r"black\s+eagle\s+defense",
            r"\bmdds\b|myingyan\s+district\s+drone",
            r"\bugp.okpo\b|underground.*guerr?illa.*okpo",
            r"\bmrda\b|royal\s+dragon\s+army",
            r"\bprdf\b|palaw\s+regional\s+defense",
            r"chindwin\s+attack\s+force",
            r"kyaikhto\s+revolution|\bkrf\b",
            r"generation\s+z\s+power",
            r"young\s+force.ug",
            r"civic\s+defense\s+militia.*siyin",
            r"brave\s+heart\s+army|\bbha\b.*brave",
            r"\bmnrf\b|moe\s+nyo\s+revolution",
            r"phoenix\s+df\b|phoenix\s+defense\s+force.*nattalin",
            r"\badpra\b",
            r"yangon\s+army",
            r"dark\s+shadow",
            r"brave\s+warriors\s+for\s+myanmar|\bbwm\b",
            r"chindwin\s+brothers",
            r"\bblpa\b|black\s+leopard\s+army",
            r"dawei\s+defense\s+team|\bddt\b.*dawei",
            r"people.s\s+revolution\s+army",
            r"danger\s+force\s+ldf",
            r"\bpkdf\b|people.s\s+knight\s+defense",
            r"nat\s+soe\s+myay|demon\s+underground\s+revolutionary",
            r"bo\s+lin\s+yone\s+tatphwe|bo\s+eagle\s+force",
            r"bagan\s+ogre\s+force",
            r"freedom\s+revolution\s+force|\bfrf\b|farmers\s+revolution\s+force",
            r"operation\s+flame",
            r"\bkddf\b|kyaukse\s+district\s+defense",
            r"red\s+bandana\s+column|pu\s+war\s+ni\s+sit",
            r"\bnrdf\b|natogyi\s+regional\s+defense",
            r"\busba\b|united\s+states\s+of\s+burma\s+army",
            r"freeland\s+attack\s+force|\bfla\b.*freeland",
            r"people.s\s+revolution\s+alliance.*magway|pra\s+magway",
            r"\bsstf\b|salingyi\s+special\s+task",
            r"phoenix\s+sgg",
            r"myaung\s+revolutionary\s+army|\bmra\b.*myaung",
            r"\bcgf\b|chauk\s+guerr?illa",
            r"dawei.*joint\s+forces|dawei\s+kha\s+yaing",
            r"myaing\s+villages\s+revolution|\bmvrf\b",
            r"northern\s+thandaung\s+defense|\bntdf\b",
            r"\beagle\s+force\b",
            r"golden\s+eagle\s+force",
            r"people.s\s+servant\s+revolution|\bpsr\b.*wetlet",
            r"shar\s+htoo\s+waw\s+drone",
            r"khin\s+u\s+support|\bkso\b.*khin",
            r"du\s+ya\s+ka\s+(?:sit|column)",
            r"mya\s+nan\s+dar.*(?:mandalay|sit|operation)",
            r"daw\s+na\s+column",
            r"yaw\s+revolution\s+army.*tilin|yra.*tilin",
            r"king\s+cobra.khin\s+u",
            r"thayarwady\s+galon",
            r"taw\s+gyi\s+mway\s+hauk|king\s+cobra\s+force",
            r"\bacplf\b|anti.coup\s+people.s\s+liberation",
            r"gyobingauk\s+(?:hero|guerr?illa)",
            r"\bbldf\b|burma\s+liberation\s+democratic",
            r"brother\s+defense\s+force.*yinmarbin|\bbdf\b.*yinmarbin",
            r"\bmgw\b|magway\s+guerr?illa\s+warfare",
            r"tha\s+pyay\s+nyo\s+(?:guerr?illa|pyauk)",
            r"\bystf\b|ye\s+special\s+task",
            r"aung\s+si\s+taw.*mandalay|\bast.mdy\b",
        ],
        'side': 'Anti-junta',
        'coalition': 'National Unity Government (PDF, NUG)',
        'actor_group': "People's Defence Force",
    },

    # ── PRO-JUNTA ▸ Tatmadaw and allies ──────────────────────────────────────
    {
        'patterns': [
            r"military forces of myanmar",
            r"police forces of myanmar",
            r"government of myanmar",
        ],
        'side': 'Pro-junta',
        'coalition': 'Tatmadaw and allies',
        'actor_group': 'Tatmadaw',
    },
    {
        'patterns': [r"pyu saw", r"pyusawhti"],
        'side': 'Pro-junta',
        'coalition': 'Tatmadaw and allies',
        'actor_group': 'Pyusawhti militias',
    },
    {
        'patterns': [r"thway thauk aphwe", r"blood comrades.*pro.mil"],
        'side': 'Pro-junta',
        'coalition': 'Tatmadaw and allies',
        'actor_group': 'Thway Thout',
    },

    # ── PRO-JUNTA ▸ Aligned ethnic armed organisations ────────────────────────
    {
        'patterns': [r"^kna:.*karen national army$"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'KNA',
    },
    {
        'patterns': [r"alp/ala", r"arakan liberation"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'ALA',
    },
    {
        'patterns': [r"pno/pna", r"pa-oh national org", r"pa.oh national army"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'PNA',
    },
    {
        'patterns': [r"shanni nationalities", r"\bsna\b.*shanni"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'Shanni Nationalities Army',
    },
    {
        'patterns': [r"\bzra\b", r"zomi revolutionary army"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'ZRA',
    },
    {
        'patterns': [r"\barsa\b", r"arakan rohingya salvation"],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'ARSA',
    },
    {
        'patterns': [
            r"kno/kna.*kuki", r"kuki national",
            r"sspp/ssa-n", r"shan state progress",
            r"dkba.*buddh", r"democratic karen buddh",
            r"rohingya muslim militia",
            r"\brso\b.*rohingya", r"rakhine ethnic militia",
            r"thway thauk",
        ],
        'side': 'Pro-junta',
        'coalition': 'Aligned ethnic armed organisations',
        'actor_group': 'Other pro-junta EAOs',
    },

    # ── NEUTRAL / SELF-ADMINISTERED ───────────────────────────────────────────
    {
        'patterns': [r"uwsp/uwsa", r"united wa state"],
        'side': 'Neutral / Self-administered',
        'coalition': 'Self-administered zones',
        'actor_group': 'UWSA',
    },
    {
        'patterns': [r"psc/ndaa-ess", r"ndaa.*eastern shan"],
        'side': 'Neutral / Self-administered',
        'coalition': 'Self-administered zones',
        'actor_group': 'NDAA',
    },
    {
        'patterns': [r"rcss/ssa-s", r"restoration council.*shan", r"shan state army.south"],
        'side': 'Neutral / Self-administered',
        'coalition': 'Self-administered zones',
        'actor_group': 'RCSS / Shan State Army South',
    },

    # ── OTHER / UNMAPPED ──────────────────────────────────────────────────────
    {
        'patterns': [r"civilians?\s*\(myanmar\)", r"^civilians?\s*$"],
        'side': 'Other / Unmapped',
        'coalition': 'Unmapped',
        'actor_group': 'Civilians',
    },
    {
        'patterns': [r"protesters?\s*\(myanmar\)", r"^protesters?\s*$"],
        'side': 'Other / Unmapped',
        'coalition': 'Unmapped',
        'actor_group': 'Protesters',
    },
    {
        'patterns': [r"rioters?\s*\(myanmar\)", r"^rioters?\s*$"],
        'side': 'Other / Unmapped',
        'coalition': 'Unmapped',
        'actor_group': 'Protesters',
    },
]

# Time-varying overrides applied after base mapping
TIME_RULES = [
    {
        'actor_group': 'TNLA', 'direction': 'after',
        'cutoff': pd.Timestamp('2026-01-01'),
        'override': {'side': 'Other / Unmapped', 'coalition': 'Other combatant', 'actor_group': 'TNLA'},
    },
    {
        'actor_group': 'MNDAA', 'direction': 'after',
        'cutoff': pd.Timestamp('2026-01-01'),
        'override': {'side': 'Other / Unmapped', 'coalition': 'Other combatant', 'actor_group': 'MNDAA'},
    },
    {
        'actor_group': 'KNPLF', 'direction': 'before',
        'cutoff': pd.Timestamp('2023-01-01'),
        'override': {'side': 'Other / Unmapped', 'coalition': 'Unmapped', 'actor_group': 'KNPLF'},
    },
    {
        'actor_group': 'PNLA', 'direction': 'before',
        'cutoff': pd.Timestamp('2024-01-01'),
        'override': {'side': 'Other / Unmapped', 'coalition': 'Unmapped', 'actor_group': 'PNLA'},
    },
]

# Fallback when no named rule matches: use inter1 actor type
INTER1_FALLBACK = {
    'State forces':          ('Pro-junta',        'Tatmadaw and allies',  'Tatmadaw'),
    'Rebel group':           ('Other / Unmapped', 'Other combatant',      'Unmapped'),
    'Political militia':     ('Other / Unmapped', 'Other combatant',      'Unmapped'),
    'Identity militia':      ('Other / Unmapped', 'Other combatant',      'Unmapped'),
    'Rioters':               ('Other / Unmapped', 'Unmapped',             'Protesters'),
    'Protesters':            ('Other / Unmapped', 'Unmapped',             'Protesters'),
    'Civilians':             ('Other / Unmapped', 'Unmapped',             'Civilians'),
    'External/Other forces': ('Other / Unmapped', 'Unmapped',             'Unmapped'),
}

# Pre-compile regexes once
_compiled_rules = [
    {**r, '_pats': [re.compile(p, re.IGNORECASE) for p in r['patterns']]}
    for r in ACTOR_RULES
]


# ── 5. ACTOR GROUPING FUNCTION ────────────────────────────────────────────────

def _match_actor(actor_str: str):
    s = str(actor_str)
    for rule in _compiled_rules:
        if any(p.search(s) for p in rule['_pats']):
            return rule['side'], rule['coalition'], rule['actor_group']
    return None, None, None


def apply_actor_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    unique_actors = df['actor1'].dropna().unique()
    lookup = {a: _match_actor(a) for a in unique_actors}

    results       = df['actor1'].map(lookup)
    df['side']        = results.map(lambda x: x[0] if isinstance(x, tuple) else None)
    df['coalition']   = results.map(lambda x: x[1] if isinstance(x, tuple) else None)
    df['actor_group'] = results.map(lambda x: x[2] if isinstance(x, tuple) else None)

    # inter1-type fallback for unmatched rows
    fallback_mask = df['side'].isna()
    for inter1_val, (fb_side, fb_coal, fb_ag) in INTER1_FALLBACK.items():
        m = fallback_mask & (df['inter1'] == inter1_val)
        df.loc[m, ['side', 'coalition', 'actor_group']] = [fb_side, fb_coal, fb_ag]

    # Catch any remaining nulls
    still_null = df['side'].isna()
    df.loc[still_null, ['side', 'coalition', 'actor_group']] = [
        'Other / Unmapped', 'Unmapped', 'Unmapped'
    ]

    # Time-varying overrides
    for rule in TIME_RULES:
        ag_mask   = df['actor_group'] == rule['actor_group']
        date_mask = (df['event_date'] >= rule['cutoff']) if rule['direction'] == 'after' \
                    else (df['event_date'] <  rule['cutoff'])
        ov = rule['override']
        df.loc[ag_mask & date_mask, ['side', 'coalition', 'actor_group']] = [
            ov['side'], ov['coalition'], ov['actor_group']
        ]

    return df


# ── 6. LOAD AND PREPARE ACLED ─────────────────────────────────────────────────

print('Loading ACLED data...')
df_raw = pd.read_csv(acled_path, low_memory=False)
df_raw['event_date'] = pd.to_datetime(df_raw['event_date'])

df = df_raw[df_raw['country'] == 'Myanmar'].copy()
print(f'Myanmar events: {len(df):,}  ({df["event_date"].min().date()} → {df["event_date"].max().date()})')

# Flag territorial control signal events
df['is_control_signal'] = df['sub_event_type'].isin(CONTROL_SIGNAL_TYPES)

# Apply actor taxonomy
df = apply_actor_taxonomy(df)
print(f'Actor taxonomy applied. Unmapped: {(df["actor_group"] == "Unmapped").sum():,}')


# ── 7. FILTER TO POST-COUP CONTROL SIGNALS ────────────────────────────────────

acled = df[
    (df['event_date'] >= COUP_DATE) &
    df['is_control_signal']
].copy()

acled['year_month'] = acled['event_date'].dt.to_period('M').astype(str)

# PRIOGRID cell GID from lat/lon (same formula as build_priogrid_labels.py)
acled['priogrid_gid'] = (
    np.floor((acled['latitude']  + 90)  / 0.5).astype(int) * 720 +
    np.floor((acled['longitude'] + 180) / 0.5).astype(int) + 1
)

print(f'Post-coup control signal events: {len(acled):,}  |  '
      f'{acled["priogrid_gid"].nunique()} cells  |  '
      f'{acled["year_month"].nunique()} months')


# ── 8. AGGREGATE TO CELL × MONTH ─────────────────────────────────────────────

acled['acled_side2'] = acled['side'].map(SIDE_MAP).fillna('unmapped')

cell_month = (
    acled
    .groupby(['priogrid_gid', 'year_month'], sort=False)
    .agg(
        n_gov          = ('acled_side2', lambda x: (x == 'gov').sum()),
        n_opo          = ('acled_side2', lambda x: (x == 'opo').sum()),
        n_unmapped     = ('acled_side2', lambda x: (x == 'unmapped').sum()),
        total_ctrl_evs = ('event_id_cnty', 'count'),
        admin1         = ('admin1', 'first'),
    )
    .reset_index()
)

# Classify ACLED-implied side
cell_month['acled_side'] = np.where(
    (cell_month['n_gov'] > 0) & (cell_month['n_opo'] > 0),
    'mixed',
    np.where(
        cell_month['n_gov'] > 0, 'gov',
        np.where(cell_month['n_opo'] > 0, 'opo', 'unmapped_only')
    )
)

print(f'Cell-months with ≥1 control signal: {len(cell_month):,}')


# ── 9. LOAD WIKIPEDIA LABELS AND JOIN ─────────────────────────────────────────

wiki = pd.read_csv(WIKI_PATH)
wiki['wiki_side'] = wiki['control_status'].map(WIKI_SIDE_MAP)

comparison = cell_month.merge(
    wiki[['priogrid_gid', 'year_month', 'wiki_side', 'control_status']],
    on=['priogrid_gid', 'year_month'],
    how='left',
)


# ── 10. DERIVE AGREEMENT ──────────────────────────────────────────────────────

is_mixed  = comparison['acled_side'] == 'mixed'
is_mapped = comparison['acled_side'].isin(['gov', 'opo'])
has_wiki  = comparison['wiki_side'].notna()
contested = comparison['wiki_side'] == 'contested'
match     = is_mapped & has_wiki & ~contested & (comparison['acled_side'] == comparison['wiki_side'])
mismatch  = is_mapped & has_wiki & ~contested & (comparison['acled_side'] != comparison['wiki_side'])

comparison['agreement'] = 'no_info'
comparison.loc[is_mixed,                         'agreement'] = 'acled_mixed'
comparison.loc[is_mapped & has_wiki & contested, 'agreement'] = 'wiki_contested'
comparison.loc[match,                            'agreement'] = 'agree'
comparison.loc[mismatch,                         'agreement'] = 'disagree'

print('Agreement distribution:')
for cat, n in comparison['agreement'].value_counts().items():
    print(f'  {cat:<20}  {n:>4,}  ({100 * n / len(comparison):.1f}%)')


# ── 11. SAVE ──────────────────────────────────────────────────────────────────

comparison.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e3:.1f} KB)')
print(f'Shape: {comparison.shape}')
print(f'Columns: {list(comparison.columns)}')
