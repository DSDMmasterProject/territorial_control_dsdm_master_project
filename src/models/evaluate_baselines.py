#!/usr/bin/env python
# evaluate_baselines.py
# Compare focal_mask predictions against both baselines.
# Replicates the run_eval logic from notebook 10_focal_mask_all_features.ipynb (Cell 21).
#
# Inputs:
#   models/focal_mask/predictions.csv               (produced by focal_mask_train.py)
#   models/baseline_random_proportion/predictions.csv
#   models/baseline_no_change/predictions.csv
#
# Outputs:
#   models/baseline_comparison.csv   (comparison table)
#   stdout: formatted comparison table

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / 'models'

FOCAL_MASK_PATH      = MODELS_DIR / 'focal_mask' / 'predictions.csv'
RAND_PROP_PATH       = MODELS_DIR / 'baseline_random_proportion' / 'predictions.csv'
NO_CHANGE_PATH       = MODELS_DIR / 'baseline_no_change' / 'predictions.csv'
COMPARISON_OUT_PATH  = MODELS_DIR / 'baseline_comparison.csv'

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ['gov', 'opo', 'uncertain']
LABEL_MAP   = {'gov': 0, 'opo': 1, 'uncertain': 2}
INV_LABEL   = {0: 'gov', 1: 'opo', 2: 'uncertain'}


# ── Eval helper (mirrors notebook Cell 21) ─────────────────────────────────────
def run_eval(pred_df: pd.DataFrame) -> dict:
    val_df = pred_df[pred_df['is_train_month'] == False].copy()
    val_df['true_int'] = val_df['true_class'].map(LABEL_MAP)
    val_df['pred_int'] = val_df['pred_class'].map(LABEL_MAP)
    y_true = val_df['true_int'].values
    y_pred = val_df['pred_int'].values

    report      = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    overall_acc = (y_true == y_pred).mean()
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm     = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    val_sorted              = val_df.sort_values(['priogrid_gid', 'year_month']).copy()
    val_sorted['prev_true'] = val_sorted.groupby('priogrid_gid')['true_int'].shift(1)
    transitions             = val_sorted.dropna(subset=['prev_true']).copy()
    transitions['prev_true'] = transitions['prev_true'].astype(int)
    transitions['is_trans']  = transitions['true_int'] != transitions['prev_true']
    trans_only = transitions[transitions['is_trans']].copy()
    stable     = transitions[~transitions['is_trans']].copy()

    trans_correct  = (trans_only['true_int'] == trans_only['pred_int']).sum()
    stable_correct = (stable['true_int']     == stable['pred_int']).sum()

    return {
        'report'    : report,
        'cm'        : cm,
        'cm_norm'   : cm_norm,
        'acc'       : overall_acc,
        'n_val'     : len(val_df),
        'trans_acc' : trans_correct / max(len(trans_only), 1),
        'stable_acc': stable_correct / max(len(stable), 1),
        'n_trans'   : len(trans_only),
        'n_stable'  : len(stable),
    }


# ── Load predictions ───────────────────────────────────────────────────────────
def _load(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f'WARNING: {name} predictions not found at {path} — skipping.')
        return None
    df = pd.read_csv(path)
    df['year_month'] = df['year_month'].astype(str)
    return df


focal_df  = _load(FOCAL_MASK_PATH, 'focal_mask')
rand_df   = _load(RAND_PROP_PATH,  'baseline_random_proportion')
nochg_df  = _load(NO_CHANGE_PATH,  'baseline_no_change')

MODELS = {
    'focal_mask'             : focal_df,
    'random_proportion'      : rand_df,
    'no_change'              : nochg_df,
}
MODELS = {k: v for k, v in MODELS.items() if v is not None}

if not MODELS:
    raise RuntimeError('No predictions found. Run the model and both baselines first.')

# ── Validate that all loaded CSVs share the same (priogrid_gid, year_month) set ─
def _key_set(df: pd.DataFrame) -> frozenset:
    return frozenset(zip(df['priogrid_gid'], df['year_month']))

key_sets  = {name: _key_set(df) for name, df in MODELS.items()}
ref_name  = next(iter(key_sets))
ref_keys  = key_sets[ref_name]

mismatch = False
for name, keys in key_sets.items():
    if name == ref_name:
        continue
    only_in_ref   = ref_keys - keys
    only_in_other = keys - ref_keys
    if only_in_ref or only_in_other:
        mismatch = True
        print(f'KEY MISMATCH: "{ref_name}" vs "{name}"')
        print(f'  rows only in {ref_name}   : {len(only_in_ref):,}')
        print(f'  rows only in {name}: {len(only_in_other):,}')

if mismatch:
    raise ValueError(
        'Prediction files cover different (priogrid_gid, year_month) sets. '
        'Re-run the baselines so all three CSVs share the same cell-month index.'
    )
print(f'Key alignment check passed — all {len(MODELS)} files share '
      f'{len(ref_keys):,} (priogrid_gid, year_month) pairs.')

# ── Evaluate ───────────────────────────────────────────────────────────────────
results = {name: run_eval(df) for name, df in MODELS.items()}

# ── Comparison table ───────────────────────────────────────────────────────────
hdr = (
    f'{"Model":<28} {"N val":>6} {"Acc":>7} {"Macro F1":>9}'
    f' {"Gov F1":>8} {"Opo F1":>8} {"Unc F1":>8} {"Trans%":>8} {"Stable%":>8}'
)
print()
print(hdr)
print('-' * len(hdr))

rows = []
for name, r in results.items():
    rep = r['report']
    gov_f1  = rep['gov']['f1-score']
    opo_f1  = rep['opo']['f1-score']
    unc_f1  = rep['uncertain']['f1-score']
    macro_f1 = rep['macro avg']['f1-score']
    print(
        f'{name:<28} {r["n_val"]:>6,} {r["acc"]:>6.1%} '
        f'{macro_f1:>8.1%} '
        f'{gov_f1:>7.1%} '
        f'{opo_f1:>7.1%} '
        f'{unc_f1:>7.1%} '
        f'{r["trans_acc"]:>7.1%} '
        f'{r["stable_acc"]:>7.1%}'
    )
    rows.append({
        'model'    : name,
        'n_val'    : r['n_val'],
        'accuracy' : round(r['acc'],        4),
        'macro_f1' : round(macro_f1,        4),
        'gov_f1'   : round(gov_f1,          4),
        'opo_f1'   : round(opo_f1,          4),
        'unc_f1'   : round(unc_f1,          4),
        'trans_acc': round(r['trans_acc'],   4),
        'stable_acc': round(r['stable_acc'], 4),
        'n_trans'  : r['n_trans'],
        'n_stable' : r['n_stable'],
    })

# ── Misclassification breakdown ────────────────────────────────────────────────
print()
print('Misclassification breakdown:')
for name, r in results.items():
    cm, cm_norm = r['cm'], r['cm_norm']
    print(f'\n  [{name}]')
    for i, tc in enumerate(CLASS_NAMES):
        for j, pc in enumerate(CLASS_NAMES):
            n = cm[i, j]
            if i != j and n > 0:
                print(f'    True {tc:>9} → pred {pc:>9}: {n:4d}  ({100*cm_norm[i,j]:.1f}%)')

# ── Save table ─────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
out_df.to_csv(COMPARISON_OUT_PATH, index=False)
print(f'\nComparison table saved: {COMPARISON_OUT_PATH}')
