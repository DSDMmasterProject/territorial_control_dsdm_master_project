#!/usr/bin/env python
# baseline_random_proportion.py
# Random baseline: assign classes by sampling from the training class distribution.
# Reproduces the random baseline in notebook 10_focal_mask_all_features.ipynb (Cell 23).
#
# Cell set is restricted to the exact same (priogrid_gid, year_month) pairs that
# focal_mask writes to predictions.csv, via build_valid_gids().
#
# ── Hard metrics vs. discrimination curves ────────────────────────────────────
# This script stores two kinds of scores that serve different purposes:
#
# pred_class / pred_class_int  (sampled via np.random.choice):
#   Correct representation for hard classification metrics (F1, accuracy,
#   confusion matrix). Simulates an agent assigning one label per cell with
#   no per-cell information, using only the training class proportions.
#
# prob_gov / prob_opo / prob_uncertain  (constant = training proportions):
#   Stored for format consistency only. These constants carry no ranking signal
#   and must NOT be used to draw PR or ROC curves: a constant score vector
#   produces a degenerate curve that collapses to one point, adding sampling
#   noise with no interpretable information.
#
# For PR/ROC visualisations, represent random_proportion by its theoretical limit:
#   - PR curve (one-vs-rest per class): horizontal line at y = class prevalence
#     in the evaluation set.  AUC-PR = prevalence.
#     [Saito & Rehmsmeier (2015), PLOS ONE — "The Precision-Recall Plot Is More
#      Informative than the ROC Plot When Evaluating Binary Classifiers on
#      Imbalanced Datasets"]
#   - ROC curve: diagonal from (0,0) to (1,1).  AUC-ROC = 0.5.
#     [Davis & Goadrich (2006), ICML — "The Relationship Between
#      Precision-Recall and ROC Curves"]
# Implementation: see src/visualizations/reporting.ipynb, Section C.
#
# Inputs:
#   data/processed/10_target_variable.parquet
#   data/processed/myanmar_feature_store.csv   (for valid-GID computation only)
#
# Outputs:
#   models/baseline_random_proportion/predictions.csv

import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 20269999
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = '2025-09'

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DATA_PROC        = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR          = PROJECT_ROOT / 'models' / 'baseline_random_proportion'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH = OUT_DIR / 'predictions.csv'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.feature_selection import build_valid_gids

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_MAP   = {'gov': 0, 'opo': 1, 'uncertain': 2}
CLASS_NAMES = ['gov', 'opo', 'uncertain']

# ── Load data ──────────────────────────────────────────────────────────────────
target = pd.read_parquet(DATA_PROC / '10_target_variable.parquet')
target['year_month'] = target['year_month'].astype(str)

features_meta = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv',
                             usecols=['priogrid_gid'])

# ── Restrict to the same GID set as focal_mask ────────────────────────────────
valid_gids = build_valid_gids(
    set(features_meta['priogrid_gid'].unique()),
    set(target['priogrid_gid'].unique()),
)
n_before = len(target)
target   = target[target['priogrid_gid'].isin(valid_gids)].copy()
print(f'Target rows: {len(target):,}  (dropped {n_before - len(target)} out-of-bounds rows)'
      f'  |  months: {target["year_month"].nunique()}'
      f'  |  cells: {target["priogrid_gid"].nunique()}')

# ── Training class proportions (exactly as in notebook Cell 23) ────────────────
np.random.seed(SEED)
train_labels = target[target['year_month'] <= TRAIN_CUTOFF]['target']
class_props  = (
    train_labels.value_counts(normalize=True)
    .reindex(CLASS_NAMES)
    .fillna(0.0)
)
p_gov, p_opo, p_unc = class_props['gov'], class_props['opo'], class_props['uncertain']
print(f'Training class proportions  gov={p_gov:.1%}  opo={p_opo:.1%}  uncertain={p_unc:.1%}')

# ── Sample random predictions ──────────────────────────────────────────────────
rand_preds = np.random.choice(CLASS_NAMES, size=len(target), p=[p_gov, p_opo, p_unc])

pred_df = target[['priogrid_gid', 'year_month', 'target']].copy()
pred_df = pred_df.rename(columns={'target': 'true_class'})
pred_df['pred_class']     = rand_preds
pred_df['pred_class_int'] = pred_df['pred_class'].map(LABEL_MAP)
pred_df['prob_gov']       = p_gov
pred_df['prob_opo']       = p_opo
pred_df['prob_uncertain'] = p_unc
pred_df['confidence']     = float(max(p_gov, p_opo, p_unc))
pred_df['has_label']      = True
pred_df['is_train_month'] = pred_df['year_month'] <= TRAIN_CUTOFF

pred_df.to_csv(PREDICTIONS_PATH, index=False)
print(f'Predictions saved: {PREDICTIONS_PATH}  ({len(pred_df):,} rows)')
print(f'Val rows: {(~pred_df["is_train_month"]).sum():,}')
print('COMPLETE.')
