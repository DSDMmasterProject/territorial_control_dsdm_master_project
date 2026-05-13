import pandas as pd
import sys
from pathlib import Path

# Add repo root to path so 'src' is importable
sys.path.insert(0, str(Path(__file__).parent))
from src.harmonize import normalize_actor, normalize_region

# ── UCDP ──────────────────────────────────────────────────────────────────────
ucdp = pd.read_csv('data/raw/ucdp/GEDEvent_v25_1.csv', low_memory=False)
mmr_ucdp = ucdp[ucdp['country'] == 'Myanmar (Burma)']
all_ucdp = pd.concat([mmr_ucdp['side_a'], mmr_ucdp['side_b']]).dropna().unique()
unmapped_ucdp = [a for a in all_ucdp if normalize_actor(a)['canonical_id'] == 'UNKNOWN']
print(f"UCDP actors: {len(all_ucdp)} unique | Unmapped: {len(unmapped_ucdp)}")
for a in sorted(unmapped_ucdp): print(f"  ✗ {a!r}")
if not unmapped_ucdp: print("  ✓ all mapped")

# ── ACLED ─────────────────────────────────────────────────────────────────────
# Auto-detect separator
for sep in ['\t', ',']:
    acled = pd.read_csv('data/raw/acled/acled_myanmar_2026-04-22.csv',
                        sep=sep, low_memory=False)
    if 'actor1' in acled.columns:
        print(f"\nACLED loaded with sep={repr(sep)}")
        break
all_acled = pd.concat([acled['actor1'], acled['actor2']]).dropna().unique()
unmapped_acled = [a for a in all_acled if normalize_actor(a)['canonical_id'] == 'UNKNOWN']
print(f"ACLED actors: {len(all_acled)} unique | Unmapped: {len(unmapped_acled)}")
if unmapped_acled:
    print("Unmapped (top 30):")
    for a in sorted(unmapped_acled)[:30]: print(f"  ✗ {a!r}")
else:
    print("  ✓ all mapped")

# ── Wikipedia ─────────────────────────────────────────────────────────────────
wiki = pd.read_csv('data/processed/myanmar_control_labels_hexgrid.csv')
all_wiki = wiki['control_actor'].dropna().unique()
unmapped_wiki = [a for a in all_wiki if normalize_actor(a)['canonical_id'] == 'UNKNOWN']
print(f"\nWikipedia actors: {len(all_wiki)} unique | Unmapped: {len(unmapped_wiki)}")
for a in sorted(unmapped_wiki): print(f"  ✗ {a!r}")
if not unmapped_wiki: print("  ✓ all mapped")

# ── Regions ───────────────────────────────────────────────────────────────────
CANONICAL = {'Kayah','Kayin','Chin','Kachin','Mon','Mandalay','Yangon',
             'Rakhine','Sagaing','Tanintharyi','Magway','Ayeyarwady','Bago','Shan','Naypyidaw'}
for source, regions in [('UCDP', mmr_ucdp['adm_1'].dropna().unique()),
                         ('ACLED', acled['admin1'].dropna().unique())]:
    unmapped = [r for r in regions
                if normalize_region(r) == r and r not in CANONICAL]
    print(f"\n{source} regions unmapped: {unmapped if unmapped else '✓ all mapped'}")