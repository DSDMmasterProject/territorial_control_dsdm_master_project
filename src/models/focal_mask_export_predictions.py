#!/usr/bin/env python
# focal_mask_export_predictions.py
# Load existing weights from models/focal_mask/weights.pt and export predictions.csv.
# Run this instead of retraining when the weights already exist.
#
# Inputs:
#   models/focal_mask/weights.pt
#   data/processed/10_target_variable.parquet
#   data/processed/myanmar_feature_store.csv
#
# Output:
#   models/focal_mask/predictions.csv

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DATA_PROC        = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR          = PROJECT_ROOT / 'models' / 'focal_mask'
WEIGHTS_PATH     = OUT_DIR / 'weights.pt'
PREDICTIONS_PATH = OUT_DIR / 'predictions.csv'

assert WEIGHTS_PATH.exists(), f'Weights not found: {WEIGHTS_PATH}'

# ── Config (must match focal_mask_train.py exactly) ───────────────────────────
TRAIN_CUTOFF    = '2025-06'
NEIGHBOR_RADIUS = 1
NUM_CLASSES     = 3
ENCODER_NAME    = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

INPUT_FEATURES = [
    'total_events', 'total_fatalities', 'events_gov', 'events_nug',
    'events_ula', 'events_kio', 'gov_vs_civilians', 'gov_event_share',
    'total_events_lag1', 'total_fatalities_lag1',
]
NUM_CHANNELS = len(INPUT_FEATURES)

LABEL_MAP   = {'gov': 0, 'opo': 1, 'uncertain': 2}
INV_LABEL   = {v: k for k, v in LABEL_MAP.items()}

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}')

# ── Load data ─────────────────────────────────────────────────────────────────
target = pd.read_parquet(DATA_PROC / '10_target_variable.parquet')
target['year_month'] = target['year_month'].astype(str)

features = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv')
features['year_month'] = features['year_month'].astype(str)

# ── Raster geometry ───────────────────────────────────────────────────────────
def gid_to_rowcol(gid):
    return (gid - 1) // 720, (gid - 1) % 720

all_gids   = features['priogrid_gid'].unique()
gid_coords = {gid: {'global_row': (gid - 1) // 720, 'global_col': (gid - 1) % 720}
              for gid in all_gids}
coords_df = pd.DataFrame.from_dict(gid_coords, orient='index')
coords_df.index.name = 'priogrid_gid'

min_global_row = coords_df['global_row'].min()
min_global_col = coords_df['global_col'].min()
coords_df['local_row'] = coords_df['global_row'] - min_global_row
coords_df['local_col'] = coords_df['global_col'] - min_global_col

RASTER_H = coords_df['local_row'].max() + 1
RASTER_W = coords_df['local_col'].max() + 1
PAD_H    = int(np.ceil(RASTER_H / 8) * 8)
PAD_W    = int(np.ceil(RASTER_W / 8) * 8)

gid_to_local = coords_df[['local_row', 'local_col']].to_dict('index')

labelled = target[['priogrid_gid', 'year_month', 'target']].merge(
    features, on=['priogrid_gid', 'year_month'], how='left'
)
labelled[INPUT_FEATURES] = labelled[INPUT_FEATURES].fillna(0.0)
labelled['label_int'] = labelled['target'].map(LABEL_MAP)

for gid in labelled['priogrid_gid'].unique():
    if gid in gid_to_local:
        continue
    r0, c0 = gid_to_rowcol(gid)
    lr = r0 - min_global_row
    lc = c0 - min_global_col
    if 0 <= lr < PAD_H and 0 <= lc < PAD_W:
        gid_to_local[gid] = {'local_row': lr, 'local_col': lc}

# ── Normalisation (fit on train months only) ──────────────────────────────────
train_mask_rows = labelled['year_month'] <= TRAIN_CUTOFF
feat_means = labelled.loc[train_mask_rows, INPUT_FEATURES].mean()
feat_stds  = labelled.loc[train_mask_rows, INPUT_FEATURES].std().replace(0, 1)
labelled[INPUT_FEATURES] = (labelled[INPUT_FEATURES] - feat_means) / feat_stds

# ── Build raster arrays ───────────────────────────────────────────────────────
months_with_labels = sorted(labelled['year_month'].unique())
X_rasters, month_keys = [], []

for month in months_with_labels:
    md_ = labelled[labelled['year_month'] == month]
    X_r = np.zeros((PAD_H, PAD_W, NUM_CHANNELS), dtype=np.float32)
    for _, row in md_.iterrows():
        gid = row['priogrid_gid']
        if gid not in gid_to_local:
            continue
        lr, lc         = gid_to_local[gid]['local_row'], gid_to_local[gid]['local_col']
        X_r[lr, lc, :] = row[INPUT_FEATURES].values
    X_rasters.append(np.transpose(X_r, (2, 0, 1)))
    month_keys.append(month)

X_array = np.stack(X_rasters)
print(f'Raster shape: {X_array.shape}')

# ── Load model ────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name=ENCODER_NAME, encoder_weights=None,
    in_channels=NUM_CHANNELS, classes=NUM_CLASSES,
    activation=None, decoder_channels=(128, 64, 32, 16), encoder_depth=4,
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f'Weights loaded: {WEIGHTS_PATH}')

# ── Export predictions ────────────────────────────────────────────────────────
all_pred_rows = []
with torch.no_grad():
    for i_m, month in enumerate(month_keys):
        X_t    = torch.from_numpy(X_array[i_m]).unsqueeze(0).to(DEVICE)
        logits = model(X_t)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        preds  = logits.argmax(dim=1)[0].cpu().numpy()

        month_rows = labelled[labelled['year_month'] == month]
        for _, row in month_rows.iterrows():
            gid = row['priogrid_gid']
            if gid not in gid_to_local:
                continue
            lr, lc   = gid_to_local[gid]['local_row'], gid_to_local[gid]['local_col']
            pred_int = int(preds[lr, lc])
            all_pred_rows.append({
                'priogrid_gid'   : int(gid),
                'year_month'     : month,
                'pred_class'     : INV_LABEL[pred_int],
                'pred_class_int' : pred_int,
                'prob_gov'       : float(probs[0, lr, lc]),
                'prob_opo'       : float(probs[1, lr, lc]),
                'prob_uncertain' : float(probs[2, lr, lc]),
                'confidence'     : float(probs[:, lr, lc].max()),
                'true_class'     : row['target'],
                'has_label'      : True,
                'is_train_month' : month <= TRAIN_CUTOFF,
            })

pred_df = pd.DataFrame(all_pred_rows)
pred_df.to_csv(PREDICTIONS_PATH, index=False)
print(f'Predictions saved: {PREDICTIONS_PATH}  ({len(pred_df):,} rows)')
print(f'Val rows: {(pred_df["is_train_month"] == False).sum():,}')
print('\nCOMPLETE.')
