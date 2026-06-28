#!/usr/bin/env python
# spatial_cv_production.py
# Spatial cross-validation for the production focal_mask model.
#
# Mechanics: identical to 05_spatial_cv_comparison.ipynb
#   - K=10 geographic blocks via k-means on cell centroids (SEED-fixed, n_init=20)
#   - 5 folds, each holds out 1 northern + 1 southern block (N+S pairing by latitude)
#   - Left join: target (185 cells) LEFT JOIN features → 0.0 fill for quiet cells
#   - Focal mask: binary dilation (radius=1) of ever-changing cells
#   - Reports NH macro F1 (non-holdout val cells) and H macro F1 (holdout val cells)
#
# Model config: EXACTLY focal_mask_train.py
#   - 10 features, ResNet-34 depth=4, decoder=(128,64,32,16)
#   - Phase1: encoder frozen 30ep (decoder only, lr=1e-4)
#   - Phase2: encoder unfrozen (lr=1e-5), 170ep more
#   - Loss: 0.7*focal_CE + 0.3*Dice  ← only diff vs notebook v_felipe_focal (0.5/0.5)
#   - FOCAL_OUT_WEIGHT=0.0 (CE zeroed outside focal region)
#   - Class weights: inverse frequency, recomputed per fold non-holdout training set
#
# Outputs:
#   models/spatial_cv_results.csv   — per-fold results
#   models/unet_cv_prod_fold{N}.pt  — temporary checkpoints (deleted after use)

import os, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from pathlib import Path
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 20269999
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if hasattr(torch.backends, "mps"):
    torch.mps.manual_seed(SEED) if hasattr(torch.mps, "manual_seed") else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
OUT_CSV      = MODELS_DIR / "spatial_cv_results.csv"

# ── Production config (mirrors focal_mask_train.py exactly) ──────────────────
TRAIN_CUTOFF     = "2025-09"
NUM_CLASSES      = 3
LABEL_MAP        = {"gov": 0, "opo": 1, "uncertain": 2}
INPUT_FEATURES   = [
    "total_events", "total_fatalities", "events_gov", "events_nug",
    "events_ula", "events_kio", "gov_vs_civilians", "gov_event_share",
    "total_events_lag1", "total_fatalities_lag1",
]
NUM_CHANNELS     = len(INPUT_FEATURES)           # 10
ENCODER_NAME     = "resnet34"
ENCODER_WEIGHTS  = "imagenet"
ENCODER_DEPTH    = 4
DECODER_CHANNELS = (128, 64, 32, 16)
BATCH_SIZE       = 8
NUM_EPOCHS       = 200
BASE_LR          = 1e-4
ENCODER_LR       = 1e-5
FREEZE_EPOCHS    = 30
FOCAL_OUT_WEIGHT = 0.0
NEIGHBOR_RADIUS  = 1
CE_WEIGHT        = 0.7          # production ratio (notebook used 0.5)
DICE_WEIGHT      = 0.3

# ── Spatial CV config (mirrors notebook Cell 0/1) ─────────────────────────────
K_PIECES = 10
N_FOLDS  = 5

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps");  print("Device: MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda"); print("Device: CUDA")
else:
    DEVICE = torch.device("cpu");  print("Device: CPU")

# =============================================================================
# STEP 1 — Load data (left join, same as production)
# =============================================================================
print("\n" + "="*60)
print("STEP 1 — Loading data")
print("="*60)

target   = pd.read_parquet(DATA_PROC / "10_target_variable.parquet")
features = pd.read_csv(DATA_PROC / "myanmar_feature_store.csv")
target["year_month"]   = target["year_month"].astype(str)
features["year_month"] = features["year_month"].astype(str)
target["target_int"]   = target["target"].map(LABEL_MAP)

# Left join from target: keeps all 185 Wikipedia-labelled cells;
# 0.0 fill for quiet cells (no UCDP events) — same logic as production.
labelled = target[["priogrid_gid", "year_month", "target", "target_int"]].merge(
    features, on=["priogrid_gid", "year_month"], how="left"
)
labelled[INPUT_FEATURES] = labelled[INPUT_FEATURES].fillna(0.0)

print(f"Left join: {labelled['priogrid_gid'].nunique()} cells  "
      f"| {labelled['year_month'].nunique()} months  "
      f"| {len(labelled):,} rows")

# =============================================================================
# STEP 2 — PRIOGRID coordinate system and raster layout
# =============================================================================
all_gids = features["priogrid_gid"].unique()
min_gr   = ((all_gids - 1) // 720).min()
min_gc   = ((all_gids - 1) % 720).min()
max_gr   = ((all_gids - 1) // 720).max()
max_gc   = ((all_gids - 1) % 720).max()

RASTER_H = int(max_gr - min_gr + 1)
RASTER_W = int(max_gc - min_gc + 1)
PAD_H    = RASTER_H if RASTER_H % 8 == 0 else RASTER_H + (8 - RASTER_H % 8)
PAD_W    = RASTER_W if RASTER_W % 8 == 0 else RASTER_W + (8 - RASTER_W % 8)
print(f"Raster: {RASTER_H}×{RASTER_W}  →  padded {PAD_H}×{PAD_W}")

# Build coordinate mapping for all cells that appear in either source
gid_to_local = {}
for gid in labelled["priogrid_gid"].unique():
    gr = (gid - 1) // 720; gc = (gid - 1) % 720
    lr = int(gr - min_gr);  lc = int(gc - min_gc)
    if 0 <= lr < PAD_H and 0 <= lc < PAD_W:
        gid_to_local[int(gid)] = {"r": lr, "c": lc}

# =============================================================================
# STEP 3 — k-means geographic blocks (mirrors notebook Cell 1)
# =============================================================================
labelled_gids = labelled["priogrid_gid"].unique()
gid_coords = pd.DataFrame({
    "priogrid_gid": labelled_gids,
    "lat": ((labelled_gids - 1) // 720) * 0.5 - 90  + 0.25,
    "lon": ((labelled_gids - 1) %  720) * 0.5 - 180 + 0.25,
})

kmeans = KMeans(n_clusters=K_PIECES, random_state=SEED, n_init=20)
gid_coords["block"] = kmeans.fit_predict(gid_coords[["lat", "lon"]])
labelled = labelled.merge(
    gid_coords[["priogrid_gid", "lat", "lon", "block"]], on="priogrid_gid", how="left"
)
gid_to_block = gid_coords.set_index("priogrid_gid")["block"].to_dict()

print("\n" + "="*60)
print(f"STEP 3 — {K_PIECES} geographic blocks (k-means, n_init=20, SEED={SEED})")
print("="*60)
block_stats = []
for b in range(K_PIECES):
    bd = labelled[labelled["block"] == b]
    nc = bd["priogrid_gid"].nunique()
    clat = gid_coords[gid_coords["block"] == b]["lat"].mean()
    clon = gid_coords[gid_coords["block"] == b]["lon"].mean()
    block_stats.append({"block": b, "n_cells": nc, "lat": clat, "lon": clon})
    print(f"  Block {b}: {nc:3d} cells  ({clat:.1f}°N, {clon:.1f}°E)")

# Fold assignments: pair northern + southern blocks by latitude
blocks_by_lat = sorted(range(K_PIECES),
                        key=lambda b: block_stats[b]["lat"], reverse=True)
FOLD_ASSIGNMENTS = [
    [blocks_by_lat[i], blocks_by_lat[i + N_FOLDS]]
    for i in range(N_FOLDS)
]

print(f"\nFold assignments ({N_FOLDS} folds, hold out N+S pair each):")
for fi, (bn, bs) in enumerate(FOLD_ASSIGNMENTS):
    nh_cells = sum(1 for g in gid_to_block if gid_to_block[g] in [bn, bs])
    print(f"  Fold {fi}: blocks ({bn}, {bs})  →  {nh_cells} holdout cells")

# Verify all 3 classes present in every holdout
all_ok = True
for fi, (bn, bs) in enumerate(FOLD_ASSIGNMENTS):
    classes = set(labelled[labelled["block"].isin([bn, bs])]["target"].unique())
    ok = {"gov", "opo", "uncertain"}.issubset(classes)
    if not ok:
        print(f"  WARNING: Fold {fi} missing class in holdout: {classes}")
        all_ok = False
if all_ok:
    print(f"Class coverage: all {N_FOLDS} folds have all 3 classes in holdout ✓")

# =============================================================================
# STEP 4 — Focal mask (mirrors production)
# =============================================================================
label_by_gid  = labelled.groupby("priogrid_gid")["target_int"].nunique()
changing_gids = set(label_by_gid[label_by_gid > 1].index)

changing_raster = np.zeros((PAD_H, PAD_W), dtype=bool)
for gid in changing_gids:
    if gid in gid_to_local:
        r, c = gid_to_local[gid]["r"], gid_to_local[gid]["c"]
        changing_raster[r, c] = True

struct     = np.ones((2 * NEIGHBOR_RADIUS + 1, 2 * NEIGHBOR_RADIUS + 1), dtype=bool)
focal_mask_np = binary_dilation(changing_raster, structure=struct)
focal_mask_t  = torch.from_numpy(focal_mask_np).bool().to(DEVICE)
print(f"\nFocal mask: {focal_mask_np.sum()} focal pixels "
      f"({len(changing_gids)} ever-changing cells, radius={NEIGHBOR_RADIUS})")

# =============================================================================
# STEP 5 — Build raster arrays (one pass, normalise on train rows only)
# =============================================================================
print("\nBuilding raster arrays...")
train_rows = labelled[labelled["year_month"] <= TRAIN_CUTOFF]
feat_means = train_rows[INPUT_FEATURES].mean()
feat_stds  = train_rows[INPUT_FEATURES].std().replace(0, 1)

month_keys = sorted(labelled["year_month"].unique())
train_idx  = [i for i, m in enumerate(month_keys) if m <= TRAIN_CUTOFF]
val_idx    = [i for i, m in enumerate(month_keys) if m >  TRAIN_CUTOFF]

X_list, y_list, m_list = [], [], []
for month in month_keys:
    md = labelled[labelled["year_month"] == month]
    X_r = np.zeros((NUM_CHANNELS, PAD_H, PAD_W), dtype=np.float32)
    y_r = np.full((PAD_H, PAD_W), -1, dtype=np.int64)
    m_r = np.zeros((PAD_H, PAD_W), dtype=np.float32)
    for _, row in md.iterrows():
        gid = int(row["priogrid_gid"])
        if gid not in gid_to_local or pd.isna(row["target_int"]):
            continue
        r, c = gid_to_local[gid]["r"], gid_to_local[gid]["c"]
        for ci, feat in enumerate(INPUT_FEATURES):
            raw = row[feat] if not pd.isna(row[feat]) else 0.0
            X_r[ci, r, c] = (raw - feat_means[feat]) / feat_stds[feat]
        y_r[r, c] = int(row["target_int"])
        m_r[r, c] = 1.0
    X_list.append(X_r); y_list.append(y_r); m_list.append(m_r)

X_array = np.stack(X_list)
y_array = np.stack(y_list)
m_array = np.stack(m_list)

X_train = X_array[train_idx]; X_val = X_array[val_idx]
y_train = y_array[train_idx]; y_val = y_array[val_idx]
m_base  = m_array            # full array, used for fold mask construction

print(f"Raster arrays: X={X_array.shape}  train_months={len(train_idx)}  "
      f"val_months={len(val_idx)}")

# =============================================================================
# HELPERS — Dataset, loss, fold masks, evaluation (same as notebook Cell 3)
# =============================================================================

class MyanmarDataset(Dataset):
    def __init__(self, X, y, m, augment=False):
        self.X = X; self.y = y; self.m = m; self.augment = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].copy(); y = self.y[idx].copy(); m = self.m[idx].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy(); y = y[:, ::-1].copy(); m = m[:, ::-1].copy()
            if np.random.rand() < 0.3:
                x = x[:, ::-1, :].copy(); y = y[::-1, :].copy(); m = m[::-1, :].copy()
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(m, dtype=torch.float32))


def focal_combined_loss_prod(predictions, targets, label_mask, cw_tensor):
    """Production loss: 0.7*focal_CE + 0.3*Dice (mirrors focal_mask_train.py)."""
    B = predictions.shape[0]
    focal_broad = focal_mask_t.unsqueeze(0).expand(B, -1, -1)
    pixel_w = torch.where(
        focal_broad, label_mask.float(), label_mask.float() * FOCAL_OUT_WEIGHT
    )
    safe_targets = targets.clone()
    safe_targets[label_mask == 0] = 0
    ce_per = nn.functional.cross_entropy(
        predictions, safe_targets, weight=cw_tensor, reduction="none"
    )
    ce   = (ce_per * pixel_w).sum() / (pixel_w.sum() + 1e-8)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True, smooth=1.0)(
        predictions, safe_targets
    )
    return CE_WEIGHT * ce + DICE_WEIGHT * dice


def build_fold_masks(fold_id):
    """Construct train/val/holdout-val masks for one fold (no data leak)."""
    holdout_blocks = FOLD_ASSIGNMENTS[fold_id]
    holdout_gids   = set(
        g for g, b in gid_to_block.items() if b in holdout_blocks
    )
    m_fold  = np.zeros_like(m_base)
    m_hold  = np.zeros_like(m_base)
    for mi, month in enumerate(month_keys):
        for _, row in labelled[labelled["year_month"] == month].iterrows():
            gid = int(row["priogrid_gid"])
            if gid not in gid_to_local or pd.isna(row["target_int"]):
                continue
            r, c = gid_to_local[gid]["r"], gid_to_local[gid]["c"]
            if gid in holdout_gids:
                m_hold[mi, r, c] = 1.0
            else:
                m_fold[mi, r, c] = 1.0
    assert (m_fold * m_hold).sum() == 0, f"LEAK in fold {fold_id}!"
    return m_fold[train_idx], m_fold[val_idx], m_hold[val_idx]


def get_metrics(model, X_v, y_v, mask_v):
    """Evaluate model on val months; restricted to cells in mask_v."""
    model.eval()
    yt_all, yp_all = [], []
    ds = MyanmarDataset(X_v, y_v, mask_v, augment=False)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, m = ds[i]
            preds = model(x.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze(0).cpu().numpy()
            yt_all.extend(y.numpy()[m.numpy() == 1].astype(int).tolist())
            yp_all.extend(preds[m.numpy() == 1].tolist())
    if len(yt_all) == 0:
        return None
    yt = np.array(yt_all); yp = np.array(yp_all)
    per = f1_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2])
    return dict(
        n        = len(yt),
        accuracy = round(float(accuracy_score(yt, yp)), 4),
        macro_f1 = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        gov_f1   = round(float(per[0]), 4),
        opo_f1   = round(float(per[1]), 4),
        unc_f1   = round(float(per[2]), 4),
    )


# =============================================================================
# STEP 6 — Spatial CV loop
# =============================================================================
print("\n" + "="*60)
print(f"STEP 6 — Spatial CV  ({N_FOLDS} folds × 200 epochs)")
print("="*60)

all_results = []
grand_start = time.time()

for fold_id in range(N_FOLDS):
    t0 = time.time()
    bn, bs = FOLD_ASSIGNMENTS[fold_id]
    print(f"\n{'─'*60}")
    print(f"Fold {fold_id+1}/{N_FOLDS}  |  holdout blocks ({bn}, {bs})")
    print(f"{'─'*60}")

    m_tr, m_v, m_h = build_fold_masks(fold_id)

    # Class weights from non-holdout training rows only
    holdout_gids_fold = set(
        g for g, b in gid_to_block.items() if b in FOLD_ASSIGNMENTS[fold_id]
    )
    nh_train = labelled[
        (labelled["year_month"] <= TRAIN_CUTOFF) &
        (~labelled["priogrid_gid"].isin(holdout_gids_fold))
    ]
    lc  = np.array([(nh_train["target_int"] == i).sum() for i in range(NUM_CLASSES)])
    cw  = lc.sum() / (NUM_CLASSES * lc)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: gov={cw[0]:.3f}  opo={cw[1]:.3f}  unc={cw[2]:.3f}")
    print(f"Train cell-months: {int(m_tr.sum()):,}  |  "
          f"Val non-holdout: {int(m_v.sum()):,}  |  "
          f"Val holdout: {int(m_h.sum()):,}")

    # Fresh model (same init every fold)
    torch.manual_seed(SEED)
    model = smp.Unet(
        encoder_name    = ENCODER_NAME,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels     = NUM_CHANNELS,
        classes         = NUM_CLASSES,
        activation      = None,
        decoder_channels= DECODER_CHANNELS,
        encoder_depth   = ENCODER_DEPTH,
    ).to(DEVICE)

    dec_params = (list(model.decoder.parameters()) +
                  list(model.segmentation_head.parameters()))
    optimizer  = optim.Adam(dec_params, lr=BASE_LR, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    tr_ld = DataLoader(
        MyanmarDataset(X_train, y_train, m_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    v_ld = DataLoader(
        MyanmarDataset(X_val, y_val, m_v, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    ckpt     = MODELS_DIR / f"unet_cv_prod_fold{fold_id}.pt"
    best_vl  = float("inf")
    phase2   = False

    # Phase 1: freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    for epoch in range(1, NUM_EPOCHS + 1):
        # Phase 2: unfreeze encoder
        if epoch == FREEZE_EPOCHS + 1 and not phase2:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group(
                {"params": list(model.encoder.parameters()), "lr": ENCODER_LR}
            )
            phase2 = True

        model.train()
        for bX, by, bm in tr_ld:
            bX, by, bm = bX.to(DEVICE), by.to(DEVICE), bm.to(DEVICE)
            optimizer.zero_grad()
            loss = focal_combined_loss_prod(model(bX), by, bm, cw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bX, by, bm in v_ld:
                bX, by, bm = bX.to(DEVICE), by.to(DEVICE), bm.to(DEVICE)
                vl += focal_combined_loss_prod(model(bX), by, bm, cw_t).item()
        avg_vl = vl / len(v_ld)
        scheduler.step(avg_vl)
        if avg_vl < best_vl:
            best_vl = avg_vl
            torch.save(model.state_dict(), ckpt)
        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            ph = "P1" if epoch <= FREEZE_EPOCHS else "P2"
            print(f"  ep {epoch:3d} [{ph}]  val_loss={avg_vl:.4f}  "
                  f"best={best_vl:.4f}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    res_nh = get_metrics(model, X_val, y_val, m_v)
    res_h  = get_metrics(model, X_val, y_val, m_h)
    delta  = round(res_h["macro_f1"] - res_nh["macro_f1"], 4) if res_nh and res_h else None

    elapsed = (time.time() - t0) / 60
    print(f"\n  Non-holdout: acc={res_nh['accuracy']:.3f}  F1={res_nh['macro_f1']:.3f}  "
          f"[gov={res_nh['gov_f1']:.3f} opo={res_nh['opo_f1']:.3f} unc={res_nh['unc_f1']:.3f}]")
    print(f"  Holdout:     acc={res_h['accuracy']:.3f}  F1={res_h['macro_f1']:.3f}  "
          f"[gov={res_h['gov_f1']:.3f} opo={res_h['opo_f1']:.3f} unc={res_h['unc_f1']:.3f}]")
    print(f"  Δ macro F1:  {delta:+.4f}  |  time: {elapsed:.1f} min")

    row = {
        "model":           "focal_mask_production",
        "fold":            fold_id,
        "holdout_blocks":  str(FOLD_ASSIGNMENTS[fold_id]),
        "best_val_loss":   round(best_vl, 4),
        "elapsed_min":     round(elapsed, 1),
    }
    for k, v in (res_nh or {}).items(): row[f"nh_{k}"] = v
    for k, v in (res_h  or {}).items(): row[f"h_{k}"]  = v
    row["delta_macro_f1"] = delta
    all_results.append(row)

    # Clean up fold checkpoint to save disk space
    if ckpt.exists():
        ckpt.unlink()

# =============================================================================
# STEP 7 — Aggregate and save
# =============================================================================
df = pd.DataFrame(all_results)
df.to_csv(OUT_CSV, index=False)

total_min = (time.time() - grand_start) / 60
print("\n" + "="*60)
print(f"SPATIAL CV COMPLETE — {N_FOLDS} folds  |  {total_min:.1f} min total")
print(f"Results saved → {OUT_CSV}")
print("="*60)

nh_mean = df["nh_macro_f1"].mean(); nh_std = df["nh_macro_f1"].std()
h_mean  = df["h_macro_f1"].mean();  h_std  = df["h_macro_f1"].std()
d_mean  = df["delta_macro_f1"].mean()

print(f"\n  Non-holdout macro F1:  {nh_mean:.3f} ± {nh_std:.3f}")
print(f"  Holdout     macro F1:  {h_mean:.3f} ± {h_std:.3f}")
print(f"  Δ (H − NH)  macro F1: {d_mean:+.3f}  ← geographic generalisation gap")

print(f"\n  Per-fold holdout F1:")
for _, row in df.iterrows():
    print(f"    Fold {int(row['fold'])}  blocks {row['holdout_blocks']:<12}  "
          f"NH={row['nh_macro_f1']:.3f}  H={row['h_macro_f1']:.3f}  "
          f"Δ={row['delta_macro_f1']:+.3f}")
