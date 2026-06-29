#!/usr/bin/env python
# focal_mask_train.py
# Train the focal-mask U-Net for Myanmar territorial control prediction.
#
# Inputs:
#   data/processed/10_target_variable.parquet
#   data/processed/myanmar_feature_store.csv
#
# Outputs:
#   models/focal_mask/weights.pt
#   models/focal_mask/predictions.csv
#   models/focal_mask/training_curves.png
#   models/focal_mask_r1.npy   (reused by focal_mask_eval.py)

import copy
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
from pathlib import Path
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 20269999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if hasattr(torch.backends, 'mps'):
    torch.mps.manual_seed(SEED) if hasattr(torch.mps, 'manual_seed') else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f'Global seed fixed: {SEED}')

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF     = '2025-09'
NEIGHBOR_RADIUS  = 1
FOCAL_OUT_WEIGHT = 0.0
NUM_EPOCHS       = 200
BATCH_SIZE       = 8
LEARNING_RATE    = 1e-4
FREEZE_EPOCHS    = 30
UNFREEZE_LR      = 1e-5

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR   = PROJECT_ROOT / 'models'
OUT_DIR      = MODELS_DIR / 'focal_mask'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH     = OUT_DIR / 'weights.pt'
PREDICTIONS_PATH = OUT_DIR / 'predictions.csv'
FOCAL_MASK_PATH  = MODELS_DIR / f'focal_mask_r{NEIGHBOR_RADIUS}.npy'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load data ─────────────────────────────────────────────────────────────────
target = pd.read_parquet(DATA_PROC / '10_target_variable.parquet')
target['year_month'] = target['year_month'].astype(str)
print(f'Target rows: {len(target):,}  |  months: {target["year_month"].nunique()}'
      f'  |  cells: {target["priogrid_gid"].nunique()}')
print('\nClass distribution:')
print(target['target'].value_counts())

features = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv')
features['year_month'] = features['year_month'].astype(str)

# ── Raster construction ───────────────────────────────────────────────────────
INPUT_FEATURES = [
    'total_events', 'total_fatalities', 'events_gov', 'events_nug', 'events_ula',
    'events_kio', 'gov_vs_civilians', 'gov_event_share',
    'total_events_lag1', 'total_fatalities_lag1',
]
print(f'Input features ({len(INPUT_FEATURES)}): {INPUT_FEATURES}')
NUM_CHANNELS = len(INPUT_FEATURES)

LABEL_MAP   = {'gov': 0, 'opo': 1, 'uncertain': 2}
INV_LABEL   = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ['gov', 'opo', 'uncertain']


def gid_to_rowcol(gid):
    return (gid - 1) // 720, (gid - 1) % 720


all_gids   = features['priogrid_gid'].unique()
gid_coords = {}
for gid in all_gids:
    r, c = gid_to_rowcol(gid)
    gid_coords[gid] = {'global_row': r, 'global_col': c}
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

# Left join from labels: recovers zero-event cells with Wikipedia annotations
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

n_recovered = len(labelled['priogrid_gid'].unique()) - len(all_gids)
print(f'Left join: {labelled["priogrid_gid"].nunique()} unique cells'
      f'  ({n_recovered:+d} vs inner-join baseline)')

train_mask_rows   = labelled['year_month'] <= TRAIN_CUTOFF
train_months_used = sorted(labelled.loc[train_mask_rows, 'year_month'].unique())
feat_means = labelled.loc[train_mask_rows, INPUT_FEATURES].mean()
feat_stds  = labelled.loc[train_mask_rows, INPUT_FEATURES].std().replace(0, 1)
print(f'Normalisation fit on {len(train_months_used)} training months: '
      f'{train_months_used[0]} → {train_months_used[-1]}')
labelled[INPUT_FEATURES] = (labelled[INPUT_FEATURES] - feat_means) / feat_stds

months_with_labels = sorted(labelled['year_month'].unique())
X_rasters, y_rasters, masks_r, month_keys = [], [], [], []

for month in months_with_labels:
    md_ = labelled[labelled['year_month'] == month]
    X_r = np.zeros((PAD_H, PAD_W, NUM_CHANNELS), dtype=np.float32)
    y_r = np.full((PAD_H, PAD_W), fill_value=-1, dtype=np.int64)
    msk = np.zeros((PAD_H, PAD_W), dtype=np.uint8)
    for _, row in md_.iterrows():
        gid = row['priogrid_gid']
        if gid not in gid_to_local:
            continue
        lr, lc         = gid_to_local[gid]['local_row'], gid_to_local[gid]['local_col']
        X_r[lr, lc, :] = row[INPUT_FEATURES].values
        y_r[lr, lc]    = row['label_int']
        msk[lr, lc]    = 1
    X_rasters.append(np.transpose(X_r, (2, 0, 1)))
    y_rasters.append(y_r)
    masks_r.append(msk)
    month_keys.append(month)

X_array = np.stack(X_rasters)
y_array = np.stack(y_rasters)
m_array = np.stack(masks_r)

train_idx = [i for i, m in enumerate(month_keys) if m <= TRAIN_CUTOFF]
val_idx   = [i for i, m in enumerate(month_keys) if m >  TRAIN_CUTOFF]
X_train, y_train, m_train = X_array[train_idx], y_array[train_idx], m_array[train_idx]
X_val,   y_val,   m_val   = X_array[val_idx],   y_array[val_idx],   m_array[val_idx]

print(f'Raster shape: {X_array.shape}  (months x channels x H x W)')
print(f'Train months: {len(train_idx)}  ({month_keys[train_idx[0]]} → {month_keys[train_idx[-1]]})')
print(f'Val   months: {len(val_idx)}   ({month_keys[val_idx[0]]} → {month_keys[val_idx[-1]]})')

# ── Focal mask ────────────────────────────────────────────────────────────────
# Dilate the set of ever-changing cells by one queen-neighbor radius.
# CE loss is zeroed outside this focal region to prevent stable cells from
# drowning out the contested frontier signal.
label_by_gid  = labelled.groupby('priogrid_gid')['label_int'].nunique()
changing_gids = set(label_by_gid[label_by_gid > 1].index)

changing_raster = np.zeros((PAD_H, PAD_W), dtype=bool)
for gid in changing_gids:
    if gid in gid_to_local:
        lr, lc = gid_to_local[gid]['local_row'], gid_to_local[gid]['local_col']
        changing_raster[lr, lc] = True

struct        = np.ones((2 * NEIGHBOR_RADIUS + 1, 2 * NEIGHBOR_RADIUS + 1), dtype=bool)
focal_mask_np = binary_dilation(changing_raster, structure=struct)

np.save(FOCAL_MASK_PATH, focal_mask_np)
print(f'Focal mask saved: {FOCAL_MASK_PATH}'
      f'  ({focal_mask_np.sum()} focal pixels, {len(changing_gids)} changing cells)')

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}')

# ── Model and loss ────────────────────────────────────────────────────────────
NUM_CLASSES     = 3
ENCODER_NAME    = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

label_counts         = np.array([
    (labelled['label_int'] == 0).sum(),
    (labelled['label_int'] == 1).sum(),
    (labelled['label_int'] == 2).sum(),
])
class_weights_np     = len(labelled) / (NUM_CLASSES * label_counts)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

dice_loss_fn      = smp.losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1.0)
focal_mask_tensor = torch.from_numpy(focal_mask_np).bool().to(DEVICE)


def combined_loss(predictions, targets, mask):
    safe_targets = targets.clone()
    safe_targets[mask == 0] = 0
    preds_hwc   = predictions.permute(0, 2, 3, 1)
    valid_preds = preds_hwc[mask == 1]
    valid_tgts  = targets[mask == 1]
    if valid_preds.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)
    ce = F.cross_entropy(valid_preds, valid_tgts, weight=class_weights_tensor)
    dc = dice_loss_fn(predictions, safe_targets)
    return 0.5 * ce + 0.5 * dc


def focal_combined_loss(predictions, targets, label_mask):
    B           = predictions.shape[0]
    focal_broad = focal_mask_tensor.unsqueeze(0).expand(B, -1, -1)
    pixel_w     = torch.where(
        focal_broad,
        label_mask.float(),
        label_mask.float() * FOCAL_OUT_WEIGHT,
    )
    safe_targets = targets.clone()
    safe_targets[label_mask == 0] = 0
    ce_per_pixel = F.cross_entropy(
        predictions, safe_targets,
        weight=class_weights_tensor,
        reduction='none',
    )
    n_effective = pixel_w.sum() + 1e-8
    ce = (ce_per_pixel * pixel_w).sum() / n_effective
    dc = dice_loss_fn(predictions, safe_targets)
    return 0.7 * ce + 0.3 * dc


def build_model():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    return smp.Unet(
        encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
        in_channels=NUM_CHANNELS,  classes=NUM_CLASSES,
        activation=None, decoder_channels=(128, 64, 32, 16), encoder_depth=4,
    ).to(DEVICE)


def build_optimizer(model):
    params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    opt    = optim.Adam(params, lr=LEARNING_RATE, weight_decay=1e-5)
    sch    = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    return opt, sch

# ── Dataset ───────────────────────────────────────────────────────────────────
class MyanmarRasterDataset(Dataset):
    def __init__(self, X, y, mask, augment=False):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.mask    = mask.astype(np.uint8)
        self.augment = augment
        self.T       = len(X)

    def __len__(self): return self.T

    def __getitem__(self, idx):
        X, y, mask = (torch.from_numpy(self.X[idx]),
                      torch.from_numpy(self.y[idx]),
                      torch.from_numpy(self.mask[idx]))
        if self.augment:
            if torch.rand(1).item() > 0.5:
                X, y, mask = torch.flip(X, [2]), torch.flip(y, [1]), torch.flip(mask, [1])
            if torch.rand(1).item() > 0.7:
                X, y, mask = torch.flip(X, [1]), torch.flip(y, [0]), torch.flip(mask, [0])
            X = X + torch.randn_like(X) * 0.05
        return X, y, mask


def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def make_loaders():
    g        = torch.Generator(); g.manual_seed(SEED)
    train_ds = MyanmarRasterDataset(X_train, y_train, m_train, augment=True)
    val_ds   = MyanmarRasterDataset(X_val,   y_val,   m_val,   augment=False)
    tr_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, drop_last=False,
                       generator=g, worker_init_fn=seed_worker)
    va_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr_ld, va_ld


def compute_metrics(logits, targets, mask):
    preds   = torch.argmax(logits, dim=1)
    vp, vt  = preds[mask == 1], targets[mask == 1]
    if len(vt) == 0:
        return {'overall_acc': 0.0}
    acc     = (vp == vt).float().mean().item()
    cls_acc = {}
    for c, name in enumerate(CLASS_NAMES):
        cm = vt == c
        cls_acc[name] = (vp[cm] == c).float().mean().item() if cm.sum() > 0 else float('nan')
    return {'overall_acc': acc, **cls_acc}


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model, optimizer, new_lr):
    for p in model.encoder.parameters():
        p.requires_grad = True
    optimizer.add_param_group({'params': list(model.encoder.parameters()), 'lr': new_lr})

# ── Training loop ─────────────────────────────────────────────────────────────
model        = build_model()
opt, sched   = build_optimizer(model)
tr_ld, va_ld = make_loaders()

history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                            'val_acc_gov', 'val_acc_opo', 'val_acc_uncertain', 'lr']}
best_val_loss, best_wts = float('inf'), None
freeze_encoder(model)

print(f'\nTraining: {NUM_EPOCHS} epochs, TRAIN_CUTOFF={TRAIN_CUTOFF}')
for epoch in range(1, NUM_EPOCHS + 1):
    if epoch == FREEZE_EPOCHS + 1:
        print(f'  ── Phase 2: unfreezing encoder (epoch {epoch})')
        unfreeze_encoder(model, opt, UNFREEZE_LR)

    model.train()
    e_loss, e_acc, nb = 0.0, 0.0, 0
    for Xb, yb, mb in tr_ld:
        Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        opt.zero_grad()
        preds = model(Xb)
        loss  = focal_combined_loss(preds, yb, mb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        with torch.no_grad():
            m_ = compute_metrics(preds, yb, mb)
        e_loss += loss.item(); e_acc += m_['overall_acc']; nb += 1
    avg_tl, avg_ta = e_loss / nb, e_acc / nb

    model.eval()
    e_loss, e_acc, nb = 0.0, 0.0, 0
    cls_ = {'gov': [], 'opo': [], 'uncertain': []}
    with torch.no_grad():
        for Xb, yb, mb in va_ld:
            Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            preds = model(Xb)
            loss  = combined_loss(preds, yb, mb)
            m_    = compute_metrics(preds, yb, mb)
            e_loss += loss.item(); e_acc += m_['overall_acc']; nb += 1
            for c in CLASS_NAMES:
                if not np.isnan(m_.get(c, float('nan'))):
                    cls_[c].append(m_[c])
    avg_vl, avg_va = e_loss / nb, e_acc / nb
    avg_cls = {c: np.mean(v) if v else float('nan') for c, v in cls_.items()}

    sched.step(avg_vl)
    history['train_loss'].append(avg_tl); history['val_loss'].append(avg_vl)
    history['train_acc'].append(avg_ta);  history['val_acc'].append(avg_va)
    history['val_acc_gov'].append(avg_cls['gov'])
    history['val_acc_opo'].append(avg_cls['opo'])
    history['val_acc_uncertain'].append(avg_cls['uncertain'])
    history['lr'].append(opt.param_groups[0]['lr'])

    if avg_vl < best_val_loss:
        best_val_loss = avg_vl
        best_wts      = copy.deepcopy(model.state_dict())
        torch.save(best_wts, WEIGHTS_PATH)
        ck = ' <- best'
    else:
        ck = ''

    if epoch % 10 == 0 or epoch == 1:
        ph = 'P1' if epoch <= FREEZE_EPOCHS else 'P2'
        print(f'  Ep {epoch:>3}/{NUM_EPOCHS} [{ph}]  '
              f'tr {avg_tl:.3f}/{avg_ta:.3f}  val {avg_vl:.3f}/{avg_va:.3f}  '
              f'[g:{avg_cls["gov"]:.2f} o:{avg_cls["opo"]:.2f} u:{avg_cls["uncertain"]:.2f}]'
              f'{ck}')

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f'\nBest val loss: {best_val_loss:.4f}  |  weights: {WEIGHTS_PATH}')

# ── Training curves ───────────────────────────────────────────────────────────
epochs_range = range(1, len(history['train_loss']) + 1)
col = '#2266CC'

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#F8F6F0')
fig.suptitle('Training History — focal_mask', fontsize=12, fontweight='bold')

ax = axes[0]
ax.plot(epochs_range, history['train_loss'], color=col, lw=2, label='Train')
ax.plot(epochs_range, history['val_loss'],   color=col, lw=2, ls='--', label='Val')
ax.axvline(FREEZE_EPOCHS, color='grey', ls=':', alpha=0.6, label='Phase 2 start')
ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(epochs_range, history['val_acc'],          color=col,       lw=2,   label='Overall')
ax.plot(epochs_range, history['val_acc_gov'],       color='#CC3333', lw=1.5, ls='--', label='gov')
ax.plot(epochs_range, history['val_acc_opo'],       color='#2266CC', lw=1.5, ls='--', label='opo')
ax.plot(epochs_range, history['val_acc_uncertain'], color='#FF9900', lw=1.5, ls='--', label='unc')
ax.axvline(FREEZE_EPOCHS, color='grey', ls=':', alpha=0.6)
ax.set_title('Val Accuracy by Class'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(epochs_range, history['lr'], color=col, lw=2)
ax.axvline(FREEZE_EPOCHS, color='grey', ls=':', alpha=0.6)
ax.set_title('Learning Rate'); ax.grid(True, alpha=0.3)

plt.tight_layout()
curves_path = OUT_DIR / 'training_curves.png'
fig.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Training curves saved: {curves_path}')

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
