#!/usr/bin/env python
# focal_mask_eval.py
# Evaluate the focal-mask U-Net on the held-out validation set.
#
# Input:
#   models/focal_mask/predictions.csv    (produced by focal_mask_train.py)
#
# Outputs:
#   models/focal_mask/eval_confusion_matrix.png
#   models/focal_mask/eval_transitions.png
#   (metrics printed to stdout)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / 'models' / 'focal_mask'
PRED_PATH    = OUT_DIR / 'predictions.csv'

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['gov', 'opo', 'uncertain']
LABEL_MAP_R = {'gov': 0, 'opo': 1, 'uncertain': 2}
INV_LABEL   = {0: 'gov', 1: 'opo', 2: 'uncertain'}

# ── Load predictions ──────────────────────────────────────────────────────────
pred_csv = pd.read_csv(PRED_PATH)
val_df   = pred_csv[pred_csv['is_train_month'] == False].copy()
val_df['true_int'] = val_df['true_class'].map(LABEL_MAP_R)
val_df['pred_int'] = val_df['pred_class'].map(LABEL_MAP_R)

y_true = val_df['true_int'].values
y_pred = val_df['pred_int'].values
n_val  = len(val_df)

n_train_months = pred_csv[pred_csv['is_train_month']]['year_month'].nunique()
n_val_months   = val_df['year_month'].nunique()
n_cells        = pred_csv['priogrid_gid'].nunique()

# ── Classification report ─────────────────────────────────────────────────────
print('=' * 60)
print(f'VALIDATION SET  ({n_val_months} months, {n_val:,} cell-months)')
print('=' * 60)

report = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print(f'\n{"Class":<14} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
print('-' * 56)
for cls in CLASS_NAMES:
    r   = report[cls]
    sup = int(r['support'])
    print(f'{cls:<14} {r["precision"]:>9.1%}  {r["recall"]:>9.1%}  {r["f1-score"]:>9.1%}'
          f'  {sup:>6,} ({100 * sup / n_val:.1f}%)')
print('-' * 56)
print(f'{"macro avg":<14} {report["macro avg"]["precision"]:>9.1%}  '
      f'{report["macro avg"]["recall"]:>9.1%}  {report["macro avg"]["f1-score"]:>9.1%}  '
      f'{n_val:>6,} (100%)')
print(f'{"weighted avg":<14} {report["weighted avg"]["precision"]:>9.1%}  '
      f'{report["weighted avg"]["recall"]:>9.1%}  {report["weighted avg"]["f1-score"]:>9.1%}')

overall_acc  = (y_true == y_pred).mean()
baseline_acc = (y_true == 0).mean()
print(f'\nOverall accuracy   : {overall_acc:.1%}')
print(f'Baseline (all-gov) : {baseline_acc:.1%}')
print(f'Improvement        : +{(overall_acc - baseline_acc):.1%}')

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#F8F6F0')
fig.suptitle('Confusion Matrix — Validation Set', fontsize=13, fontweight='bold')

ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=axes[0], cmap='Blues', colorbar=True, values_format='d')
axes[0].set_title('Raw counts')
axes[0].set_facecolor('#F8F6F0')

ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
    ax=axes[1], cmap='Blues', colorbar=True, values_format='.2f')
axes[1].set_title('Row-normalised (recall on diagonal)')
axes[1].set_facecolor('#F8F6F0')

plt.tight_layout()
fig.savefig(OUT_DIR / 'eval_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Confusion matrix saved.')

print('\nMisclassification breakdown:')
for i, true_cls in enumerate(CLASS_NAMES):
    for j, pred_cls in enumerate(CLASS_NAMES):
        n   = cm[i, j]
        pct = 100 * cm_norm[i, j]
        if i != j and n > 0:
            print(f'  True {true_cls:>9} → predicted {pred_cls:>9}: {n:4d} cells  ({pct:.1f}%)')

# ── Transition analysis ───────────────────────────────────────────────────────
val_sorted = val_df.sort_values(['priogrid_gid', 'year_month']).copy()
val_sorted['prev_true_int'] = val_sorted.groupby('priogrid_gid')['true_int'].shift(1)

transitions = val_sorted.dropna(subset=['prev_true_int']).copy()
transitions['prev_true_int'] = transitions['prev_true_int'].astype(int)
transitions['is_transition']  = transitions['true_int'] != transitions['prev_true_int']

trans_only = transitions[transitions['is_transition']].copy()
stable     = transitions[~transitions['is_transition']].copy()
n_trans    = len(trans_only)
n_stable   = len(stable)
n_total    = len(transitions)

trans_correct  = (trans_only['true_int'] == trans_only['pred_int']).sum()
stable_correct = (stable['true_int']     == stable['pred_int']).sum()

print('\n' + '=' * 60)
print('TRANSITION SUMMARY (validation window)')
print('=' * 60)
print(f'  Stable cell-months    : {n_stable:,}  ({100*n_stable/n_total:.1f}%)')
print(f'  Transition cell-months: {n_trans:,}  ({100*n_trans/n_total:.1f}%)')
print(f'  Accuracy on stable    : {100*stable_correct/n_stable:.1f}%')
print(f'  Accuracy on transition: {100*trans_correct/n_trans:.1f}%')

trans_only = trans_only.copy()
trans_only['from_cls'] = trans_only['prev_true_int'].map(INV_LABEL)
trans_only['to_cls']   = trans_only['true_int'].map(INV_LABEL)
trans_only['hit']      = trans_only['true_int'] == trans_only['pred_int']

print(f'\n  {"Transition":<22} {"Count":>6}  {"Correct":>8}  {"Hit rate":>10}')
print('  ' + '-' * 50)
for from_c in CLASS_NAMES:
    for to_c in CLASS_NAMES:
        if from_c == to_c:
            continue
        subset = trans_only[
            (trans_only['from_cls'] == from_c) & (trans_only['to_cls'] == to_c)
        ]
        if len(subset) == 0:
            continue
        n_dir     = len(subset)
        n_correct = subset['hit'].sum()
        hit_rate  = n_correct / n_dir
        note = '✓' if hit_rate >= 0.5 else ('~' if hit_rate >= 0.25 else '✗')
        print(f'  {from_c:>9} → {to_c:<9}  {n_dir:>6,}  {n_correct:>8,}  {hit_rate:>9.1%}  {note}')

# ── Transition bar chart ──────────────────────────────────────────────────────
directions = [(f, t) for f in CLASS_NAMES for t in CLASS_NAMES if f != t]
counts_bar, hit_rates_bar, labels_bar = [], [], []
for from_c, to_c in directions:
    sub = trans_only[(trans_only['from_cls'] == from_c) & (trans_only['to_cls'] == to_c)]
    if len(sub) == 0:
        continue
    counts_bar.append(len(sub))
    hit_rates_bar.append(sub['hit'].mean())
    labels_bar.append(f'{from_c}\n→{to_c}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#F8F6F0')
fig.suptitle('Transition Detection — Validation Set', fontsize=13, fontweight='bold')

x          = np.arange(len(labels_bar))
bar_colors = ['#22BB44' if h >= 0.5 else ('#FF9900' if h >= 0.25 else '#CC3333')
              for h in hit_rates_bar]

ax = axes[0]
ax.set_facecolor('#F8F6F0')
bars = ax.bar(x, hit_rates_bar, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.axhline(0.5, color='grey', ls='--', lw=1, label='50% threshold')
ax.set_xticks(x); ax.set_xticklabels(labels_bar, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Hit rate')
ax.set_title('Hit Rate by Transition Direction')
for bar, h, n in zip(bars, hit_rates_bar, counts_bar):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{h:.0%}\n(n={n})', ha='center', va='bottom', fontsize=8)
patches = [
    mpatches.Patch(color='#22BB44', label='Hit rate ≥ 50% (good)'),
    mpatches.Patch(color='#FF9900', label='Hit rate 25–49% (ok)'),
    mpatches.Patch(color='#CC3333', label='Hit rate < 25% (poor)'),
]
axes[0].legend(handles=patches, fontsize=8, loc='upper right')

ax = axes[1]
ax.set_facecolor('#F8F6F0')
bars2 = ax.bar(x, counts_bar, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels_bar, fontsize=9)
ax.set_ylabel('Number of transition cell-months')
ax.set_title('Transition Volume by Direction')
for bar, n in zip(bars2, counts_bar):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(n), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / 'eval_transitions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Transition chart saved.')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  Train / val months : {n_train_months} / {n_val_months}')
print(f'  Cells in model     : {n_cells}')
print(f'  Val cell-months    : {n_val:,}')
print(f'  Overall accuracy   : {overall_acc:.1%}')
print(f'  All-gov baseline   : {baseline_acc:.1%}')
print(f'  Lift vs baseline   : +{(overall_acc - baseline_acc):.1%}')
print()
for cls in CLASS_NAMES:
    r = report[cls]
    print(f'  {cls:<12} F1={r["f1-score"]:.1%}  prec={r["precision"]:.1%}'
          f'  rec={r["recall"]:.1%}  n={int(r["support"]):,}')
print()
print(f'  Macro F1           : {report["macro avg"]["f1-score"]:.1%}')
print(f'  Weighted F1        : {report["weighted avg"]["f1-score"]:.1%}')
print(f'  Transition hit rate: {100*trans_correct/n_trans:.1f}%  ({trans_correct}/{n_trans})')
print(f'  Stable cell acc    : {100*stable_correct/n_stable:.1f}%')
print('\nCOMPLETE.')
