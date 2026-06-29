#!/usr/bin/env python3
"""
feature_ablation.py — Feature-set size ablation for the production focal-mask model.

Trains the same U-Net ResNet-34 varying only the input feature set:
  f10  : 10 manual features (production set)
  f20  : top-20 by permutation importance (intermediate)
  f32  : 32 permutation-importance features
  f95  : all 95 features in the feature store

Everything else (arch, loss, SEED, cutoff, epochs, focal mask) is identical to
focal_mask_train.py.

Metric of interest: ROC-AUC one-vs-rest on the 31 transition cell-months.

Output: models/feature_ablation_results.csv
"""

import copy
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
from pathlib import Path
from scipy.ndimage import binary_dilation
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

# ── Hyperparameters (identical to focal_mask_train.py) ───────────────────────
SEED             = 20269999
TRAIN_CUTOFF     = '2025-09'
NEIGHBOR_RADIUS  = 1
FOCAL_OUT_WEIGHT = 0.0
NUM_EPOCHS       = 200
BATCH_SIZE       = 8
LEARNING_RATE    = 1e-4
FREEZE_EPOCHS    = 30
UNFREEZE_LR      = 1e-5
NUM_CLASSES      = 3
ENCODER_NAME     = 'resnet34'
ENCODER_WEIGHTS  = 'imagenet'
CLASS_NAMES      = ['gov', 'opo', 'uncertain']
LABEL_MAP        = {'gov': 0, 'opo': 1, 'uncertain': 2}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR   = PROJECT_ROOT / 'models'
OUT_CSV      = MODELS_DIR / 'feature_ablation_results.csv'

# ── Feature variants ──────────────────────────────────────────────────────────
FEATS_10 = [
    'total_events', 'total_fatalities', 'events_gov', 'events_nug', 'events_ula',
    'events_kio', 'gov_vs_civilians', 'gov_event_share',
    'total_events_lag1', 'total_fatalities_lag1',
]

_sel32   = pd.read_csv(DATA_PROC / 'selected_features_32.csv')
FEATS_32 = _sel32['feature'].tolist()
FEATS_20 = FEATS_32[:20]   # top-20 by permutation importance

_fs_sample  = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv', nrows=100)
_skip       = {'priogrid_gid', 'year_month', 'adm_1', 'year_month_dt'}
FEATS_93    = [c for c in _fs_sample.columns
               if c not in _skip and pd.api.types.is_numeric_dtype(_fs_sample[c])]

VARIANTS = [
    ('f10', FEATS_10),
    ('f20', FEATS_20),
    ('f32', FEATS_32),
    ('f93', FEATS_93),
]

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}')

# ── Load shared data ──────────────────────────────────────────────────────────
print('\nLoading shared data …')
target = pd.read_parquet(DATA_PROC / '10_target_variable.parquet')
target['year_month'] = target['year_month'].astype(str)

features_full = pd.read_csv(DATA_PROC / 'myanmar_feature_store.csv')
features_full['year_month'] = features_full['year_month'].astype(str)

# ── Grid coordinates (shared; determined by feature store gids) ───────────────
def gid_to_rowcol(gid):
    return (gid - 1) // 720, (gid - 1) % 720

all_gids   = features_full['priogrid_gid'].unique()
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

# Base labelled frame to determine cell set and class weights
_base_labelled = target[['priogrid_gid', 'year_month', 'target']].merge(
    features_full[['priogrid_gid', 'year_month']], on=['priogrid_gid', 'year_month'], how='left'
)

# Extend gid_to_local for any labelled cells missing from the feature store
for gid in _base_labelled['priogrid_gid'].unique():
    if gid in gid_to_local:
        continue
    r0, c0 = gid_to_rowcol(gid)
    lr = r0 - min_global_row
    lc = c0 - min_global_col
    if 0 <= lr < PAD_H and 0 <= lc < PAD_W:
        gid_to_local[gid] = {'local_row': lr, 'local_col': lc}

# Pre-built lookup for vectorised raster fill
_gid_local_df = pd.DataFrame(
    [(gid, v['local_row'], v['local_col']) for gid, v in gid_to_local.items()],
    columns=['priogrid_gid', 'local_row', 'local_col'],
)

# Class weights (same for all variants — derived from all labelled rows)
_base_labelled['label_int'] = _base_labelled['target'].map(LABEL_MAP)
_all_labels   = _base_labelled['label_int'].dropna().astype(int)
label_counts  = np.array([(_all_labels == i).sum() for i in range(3)])
class_weights_np = len(_all_labels) / (NUM_CLASSES * label_counts)

# Focal mask (same for all variants — only depends on which cells ever change)
label_by_gid  = _base_labelled.groupby('priogrid_gid')['label_int'].nunique()
changing_gids = set(label_by_gid[label_by_gid > 1].index)

changing_raster = np.zeros((PAD_H, PAD_W), dtype=bool)
for gid in changing_gids:
    if gid in gid_to_local:
        lr, lc = gid_to_local[gid]['local_row'], gid_to_local[gid]['local_col']
        changing_raster[lr, lc] = True

struct        = np.ones((2 * NEIGHBOR_RADIUS + 1, 2 * NEIGHBOR_RADIUS + 1), dtype=bool)
focal_mask_np = binary_dilation(changing_raster, structure=struct)
print(f'Focal mask: {focal_mask_np.sum()} focal pixels  ({len(changing_gids)} changing cells)')

# Transition keys (identical logic to reporting.ipynb)
_t = target[['priogrid_gid', 'year_month', 'target']].copy()
_t = _t.sort_values(['priogrid_gid', 'year_month']).reset_index(drop=True)
_t['prev_target'] = _t.groupby('priogrid_gid')['target'].shift(1)
_t_val = _t[_t['year_month'] > TRAIN_CUTOFF].copy()
_t_val['prev_target'] = _t_val['prev_target'].fillna(_t_val['target'])
_t_val['is_transition'] = _t_val['target'] != _t_val['prev_target']
transition_keys = frozenset(zip(
    _t_val.loc[_t_val['is_transition'], 'priogrid_gid'],
    _t_val.loc[_t_val['is_transition'], 'year_month'],
))
print(f'Transition events (val set): {len(transition_keys)}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def reset_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(SEED)


def build_rasters(feat_list):
    """Build X/y/mask raster arrays for the given feature list."""
    n_ch = len(feat_list)
    labelled = target[['priogrid_gid', 'year_month', 'target']].merge(
        features_full[['priogrid_gid', 'year_month'] + feat_list],
        on=['priogrid_gid', 'year_month'], how='left',
    )
    labelled[feat_list] = labelled[feat_list].fillna(0.0)
    labelled['label_int'] = labelled['target'].map(LABEL_MAP)

    train_mask = labelled['year_month'] <= TRAIN_CUTOFF
    feat_means = labelled.loc[train_mask, feat_list].mean()
    feat_stds  = labelled.loc[train_mask, feat_list].std().replace(0, 1)
    labelled[feat_list] = (labelled[feat_list] - feat_means) / feat_stds

    # Attach local row/col for vectorised fill
    labelled = labelled.merge(_gid_local_df, on='priogrid_gid', how='inner')

    months = sorted(labelled['year_month'].unique())
    X_rasters, y_rasters, masks_r = [], [], []

    for month in months:
        md = labelled[labelled['year_month'] == month]
        X_r = np.zeros((PAD_H, PAD_W, n_ch), dtype=np.float32)
        y_r = np.full((PAD_H, PAD_W), fill_value=-1, dtype=np.int64)
        msk = np.zeros((PAD_H, PAD_W), dtype=np.uint8)

        lr_v = md['local_row'].values
        lc_v = md['local_col'].values
        X_r[lr_v, lc_v, :] = md[feat_list].values
        y_r[lr_v, lc_v]    = md['label_int'].values.astype(np.int64)
        msk[lr_v, lc_v]    = 1

        X_rasters.append(np.transpose(X_r, (2, 0, 1)))
        y_rasters.append(y_r)
        masks_r.append(msk)

    X = np.stack(X_rasters)
    y = np.stack(y_rasters)
    m = np.stack(masks_r)

    train_idx = [i for i, mo in enumerate(months) if mo <= TRAIN_CUTOFF]
    val_idx   = [i for i, mo in enumerate(months) if mo >  TRAIN_CUTOFF]

    return (X[train_idx], y[train_idx], m[train_idx],
            X[val_idx],   y[val_idx],   m[val_idx],
            months, train_idx, val_idx, labelled)


class MyanmarRasterDataset(Dataset):
    def __init__(self, X, y, mask, augment=False):
        self.X, self.y, self.mask, self.augment = (
            X.astype(np.float32), y.astype(np.int64),
            mask.astype(np.uint8), augment,
        )

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        m = torch.from_numpy(self.mask[idx])
        if self.augment:
            if torch.rand(1) > 0.5:
                X, y, m = torch.flip(X, [2]), torch.flip(y, [1]), torch.flip(m, [1])
            if torch.rand(1) > 0.7:
                X, y, m = torch.flip(X, [1]), torch.flip(y, [0]), torch.flip(m, [0])
            X = X + torch.randn_like(X) * 0.05
        return X, y, m


def build_model(in_channels):
    reset_seeds()
    return smp.Unet(
        encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
        in_channels=in_channels, classes=NUM_CLASSES,
        activation=None, decoder_channels=(128, 64, 32, 16), encoder_depth=4,
    ).to(DEVICE)


def make_loaders(X_tr, y_tr, m_tr, X_va, y_va, m_va):
    g  = torch.Generator(); g.manual_seed(SEED)
    tr = DataLoader(MyanmarRasterDataset(X_tr, y_tr, m_tr, augment=True),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                    generator=g)
    va = DataLoader(MyanmarRasterDataset(X_va, y_va, m_va, augment=False),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr, va


def _class_weights():
    return torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)


def make_loss_fns():
    cw             = _class_weights()
    dice_fn        = smp.losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1.0)
    focal_t        = torch.from_numpy(focal_mask_np).bool().to(DEVICE)

    def combined_loss(preds, tgts, mask):
        safe = tgts.clone(); safe[mask == 0] = 0
        vp   = preds.permute(0, 2, 3, 1)[mask == 1]
        vt   = tgts[mask == 1]
        if vp.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True, device=preds.device)
        ce = F.cross_entropy(vp, vt, weight=cw)
        dc = dice_fn(preds, safe)
        return 0.5 * ce + 0.5 * dc

    def focal_combined_loss(preds, tgts, label_mask):
        B  = preds.shape[0]
        fb = focal_t.unsqueeze(0).expand(B, -1, -1)
        pw = torch.where(fb, label_mask.float(), label_mask.float() * FOCAL_OUT_WEIGHT)
        safe = tgts.clone(); safe[label_mask == 0] = 0
        ce_pp    = F.cross_entropy(preds, safe, weight=cw, reduction='none')
        n_eff    = pw.sum() + 1e-8
        ce       = (ce_pp * pw).sum() / n_eff
        dc       = dice_fn(preds, safe)
        return 0.7 * ce + 0.3 * dc

    return combined_loss, focal_combined_loss


def _auc_metrics(yt, yproba, prefix):
    out = {}
    vals = []
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (yt == i).astype(int)
        if y_bin.sum() == 0:
            v = float('nan')
        else:
            try:
                v = float(roc_auc_score(y_bin, yproba[:, i]))
            except ValueError:
                v = float('nan')
        k = f'{cls}_auc' if cls != 'uncertain' else 'unc_auc'
        out[f'{prefix}_{k}'] = round(v, 4) if not np.isnan(v) else float('nan')
        vals.append(v)
    macro = float(np.nanmean(vals))
    out[f'{prefix}_macro_auc'] = round(macro, 4) if not np.isnan(macro) else float('nan')
    return out


def _f1_metrics(yt, yp, prefix):
    per = f1_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2])
    return {
        f'{prefix}_macro_f1' : round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4),
        f'{prefix}_gov_f1'   : round(float(per[0]), 4),
        f'{prefix}_opo_f1'   : round(float(per[1]), 4),
        f'{prefix}_unc_f1'   : round(float(per[2]), 4),
    }


def evaluate(model, X_val, y_val, m_val, months, val_idx, labelled_df, feat_list):
    """Return dict with overall + transition val metrics."""
    model.eval()
    records = []

    with torch.no_grad():
        for rank, g_idx in enumerate(val_idx):
            month = months[g_idx]
            X_t   = torch.from_numpy(X_val[rank]).unsqueeze(0).to(DEVICE)
            logits = model(X_t)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3,H,W)
            preds  = logits.argmax(dim=1).squeeze(0).cpu().numpy()           # (H,W)

            md = labelled_df[labelled_df['year_month'] == month]
            lr_v = md['local_row'].values
            lc_v = md['local_col'].values
            true_v = md['label_int'].values.astype(int)

            for k in range(len(md)):
                records.append({
                    'gid':   int(md.iloc[k]['priogrid_gid']),
                    'month': month,
                    'true':  int(true_v[k]),
                    'pred':  int(preds[lr_v[k], lc_v[k]]),
                    'prob':  probs[:, lr_v[k], lc_v[k]].copy(),
                })

    if not records:
        return {}

    yt_all  = np.array([r['true'] for r in records])
    yp_all  = np.array([r['pred'] for r in records])
    ypr_all = np.stack([r['prob'] for r in records])  # (N,3)

    # Transition subset
    is_trans = np.array([(r['gid'], r['month']) in transition_keys for r in records])
    yt_tr  = yt_all[is_trans]
    yp_tr  = yp_all[is_trans]
    ypr_tr = ypr_all[is_trans]

    out = {'n_overall': len(yt_all), 'n_transition': int(is_trans.sum())}
    out.update(_auc_metrics(yt_all, ypr_all, 'ovr'))
    out.update(_f1_metrics(yt_all, yp_all, 'ovr'))
    if len(yt_tr) > 0:
        out.update(_auc_metrics(yt_tr, ypr_tr, 'tr'))
        out.update(_f1_metrics(yt_tr, yp_tr, 'tr'))
    else:
        for pfx in ('tr',):
            for k in ('macro_auc','gov_auc','opo_auc','unc_auc',
                      'macro_f1','gov_f1','opo_f1','unc_f1'):
                out[f'{pfx}_{k}'] = float('nan')
    return out


# ── Main ablation loop ────────────────────────────────────────────────────────
all_rows = []

for v_name, feat_list in VARIANTS:
    print(f'\n{"="*60}')
    print(f'Variant: {v_name}  |  {len(feat_list)} features')
    print(f'{"="*60}')
    t0 = time.time()
    reset_seeds()

    # Build rasters
    (X_tr, y_tr, m_tr,
     X_va, y_va, m_va,
     months, train_idx, val_idx, labelled) = build_rasters(feat_list)
    print(f'  Rasters: train={len(train_idx)} months  val={len(val_idx)} months  '
          f'shape={X_tr.shape}')

    # Model, optimiser, losses
    model = build_model(in_channels=len(feat_list))
    params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    opt    = optim.Adam(params, lr=LEARNING_RATE, weight_decay=1e-5)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    combined_loss, focal_combined_loss = make_loss_fns()

    # Freeze encoder initially
    for p in model.encoder.parameters():
        p.requires_grad = False

    tr_ld, va_ld = make_loaders(X_tr, y_tr, m_tr, X_va, y_va, m_va)
    best_val_loss, best_wts = float('inf'), None

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt.add_param_group({'params': list(model.encoder.parameters()),
                                 'lr': UNFREEZE_LR})

        model.train()
        for Xb, yb, mb in tr_ld:
            Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            loss = focal_combined_loss(model(Xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for Xb, yb, mb in va_ld:
                Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
                vl = combined_loss(model(Xb), yb, mb).item()
                vl_sum += vl; vl_n += 1
        avg_vl = vl_sum / vl_n
        sched.step(avg_vl)

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            best_wts      = copy.deepcopy(model.state_dict())

        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            ph = 'P1' if epoch <= FREEZE_EPOCHS else 'P2'
            print(f'  ep {epoch:>3} [{ph}]  val_loss={avg_vl:.4f}  best={best_val_loss:.4f}')

    model.load_state_dict(best_wts)
    elapsed = (time.time() - t0) / 60

    # Evaluate
    metrics = evaluate(model, X_va, y_va, m_va, months, val_idx, labelled, feat_list)

    row = {
        'variant':       v_name,
        'n_features':    len(feat_list),
        'best_val_loss': round(best_val_loss, 4),
        'elapsed_min':   round(elapsed, 1),
    }
    row.update(metrics)
    all_rows.append(row)

    # Save after every variant (crash-safe)
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f'  → elapsed: {elapsed:.1f} min  |  results saved')
    print(f'  Overall  : macro_AUC={metrics.get("ovr_macro_auc","n/a"):.3f}  '
          f'macro_F1={metrics.get("ovr_macro_f1","n/a"):.3f}')
    print(f'  Transition: macro_AUC={metrics.get("tr_macro_auc","n/a"):.3f}  '
          f'macro_F1={metrics.get("tr_macro_f1","n/a"):.3f}')

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('ABLATION COMPLETE')
print(f'Results → {OUT_CSV}')
print(f'{"="*60}\n')

df = pd.DataFrame(all_rows)
cols_show = ['variant', 'n_features',
             'tr_macro_auc', 'tr_gov_auc', 'tr_opo_auc', 'tr_unc_auc',
             'tr_macro_f1',
             'ovr_macro_auc', 'ovr_macro_f1']
cols_show = [c for c in cols_show if c in df.columns]
print(df[cols_show].to_string(index=False))
print()
print('Note: f10 and f32 use DIFFERENT feature selection strategies')
print('(only 1 feature overlaps). Comparison is selection strategy × size, ')
print('not a pure additive ablation.')
