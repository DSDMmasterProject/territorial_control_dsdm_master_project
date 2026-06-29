#!/usr/bin/env python
"""
export_pub_figures.py — Export reporting.ipynb figures in publication quality.

Generates PDF (vector) + PNG (300 dpi) for each figure.
Data pipeline and metric computations are identical to reporting.ipynb.
Only the presentation layer (style, background, fonts, export) differs.

Output directory: src/reporting/figures/

Run from project root:
    python src/visualizations/export_pub_figures.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR   = PROJECT_ROOT / "src" / "reporting" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
import pub_style

# ── Apply publication style globally ──────────────────────────────────────────
pub_style.apply()

# ── Constants (identical to reporting.ipynb) ───────────────────────────────────
PRED_PATHS = {
    "focal_mask"        : MODELS_DIR / "focal_mask"                 / "predictions.csv",
    "random_proportion" : MODELS_DIR / "baseline_random_proportion" / "predictions.csv",
    "no_change"         : MODELS_DIR / "baseline_no_change"         / "predictions.csv",
}
CLASS_NAMES = ["gov", "opo", "uncertain"]
LABEL_MAP   = {"gov": 0, "opo": 1, "uncertain": 2}

# Two-line titles prevent panel-title collision at 6.5in figure width
MODEL_LABELS = {
    "focal_mask"        : "U-net",
    "random_proportion" : "Random\n(proportional)",
    "no_change"         : "No-change\n(persistence)",
}

# ── Load predictions ───────────────────────────────────────────────────────────
dfs = {}
for name, path in PRED_PATHS.items():
    df = pd.read_csv(path)
    df["year_month"] = df["year_month"].astype(str)
    df["true_int"]   = df["true_class"].map(LABEL_MAP)
    df["pred_int"]   = df["pred_class"].map(LABEL_MAP)
    dfs[name] = df

# ── Transition keys (same logic as notebook) ───────────────────────────────────
def build_transition_keys(any_df):
    df = any_df[["priogrid_gid", "year_month", "true_class", "is_train_month"]].copy()
    df = df.sort_values(["priogrid_gid", "year_month"]).reset_index(drop=True)
    df["prev_true"] = df.groupby("priogrid_gid")["true_class"].shift(1)
    val = df[~df["is_train_month"]].copy()
    val["prev_true"] = val["prev_true"].fillna(val["true_class"])
    val["is_transition"] = val["true_class"] != val["prev_true"]
    return frozenset(zip(
        val.loc[val["is_transition"], "priogrid_gid"],
        val.loc[val["is_transition"], "year_month"],
    ))

transition_keys = build_transition_keys(dfs["focal_mask"])

# ── Helper: get val subset, optionally restricted to transitions ───────────────
def get_val(df, trans_keys=None):
    v = df[~df["is_train_month"]].copy()
    if trans_keys is not None:
        mask = pd.Series(
            list(zip(v["priogrid_gid"], v["year_month"])), index=v.index
        ).isin(trans_keys)
        v = v[mask]
    return v

def compute_metrics(df_subset):
    y_true  = df_subset["true_int"].values
    y_pred  = df_subset["pred_int"].values
    report  = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    return {"report": report, "cm": cm, "cm_norm": cm_norm,
            "acc": (y_true == y_pred).mean(), "n": len(df_subset)}

results_A = {name: compute_metrics(get_val(df)) for name, df in dfs.items()}
results_B = {name: compute_metrics(get_val(df, transition_keys)) for name, df in dfs.items()}

# =============================================================================
# FIGURE A2 — Overall confusion matrices
# 3 panels: Focal Mask | Random (proportional) | No-change (persistence)
# figsize = 6.5 in wide (full text width), ~2.9 in tall
# =============================================================================
print("Generating A2 — Overall confusion matrices ...")

# figsize: 6.5in wide (full text width), 3.4in tall.
# Extra height accommodates rotated x-tick labels and two-line panel titles.
fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.4))
fig.suptitle(
    "Confusion matrices — Overall validation set ($N = 1{,}098$)",
    fontsize=12, fontweight="bold",
)

im_last = None
for ax, (name, m) in zip(axes, results_A.items()):
    im = ax.imshow(m["cm_norm"], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    im_last = im

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    # Rotate x-axis labels 45° so they don't collide at 6.5in width
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(MODEL_LABELS[name], fontsize=11, fontweight="bold", pad=4)

    # Cell annotations: value + absolute count
    for i in range(3):
        for j in range(3):
            v = m["cm_norm"][i, j]
            n = m["cm"][i, j]
            color = "white" if v > 0.55 else "#222222"
            ax.text(
                j, i, f"{v:.2f}\n({n})",
                ha="center", va="center", fontsize=9, color=color,
                fontfamily="serif",
            )

# Shared colorbar attached to rightmost axis only
cbar = fig.colorbar(
    im_last, ax=axes[-1],
    fraction=0.046, pad=0.04, shrink=0.95,
)
cbar.set_label("Row-normalised recall", fontsize=9)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout(pad=0.5)
pub_style.save(fig, FIGURES_DIR / "A2_confusion_matrices_overall")
plt.close(fig)

print("Done.\n")
print("=" * 55)
print(f"All exports written to: {FIGURES_DIR}")
print("=" * 55)
