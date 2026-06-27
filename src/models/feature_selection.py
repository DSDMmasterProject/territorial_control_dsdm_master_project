#!/usr/bin/env python
# feature_selection.py
# Load the ordered list of selected features, and helpers for determining the
# valid GID set (cells included in focal_mask predictions).

import math
from pathlib import Path

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / 'data' / 'processed' / 'selected_features_32.csv'
)


def load_selected_features(path=None) -> list:
    """Return feature names in the order they appear in the CSV.

    The CSV must have a 'feature' column (e.g. selected_features_32.csv).
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    import pandas as pd

    p = Path(path) if path is not None else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Selected-features file not found: {p}\n"
            "Run notebook 10_focal_mask_all_features.ipynb to generate it, "
            "then save the top features to that path."
        )
    df = pd.read_csv(p)
    if 'feature' not in df.columns:
        raise ValueError(
            f"Expected a 'feature' column in {p}, got: {list(df.columns)}"
        )
    return df['feature'].tolist()


def build_valid_gids(features_gids: set, target_gids: set) -> set:
    """Return the set of GIDs that focal_mask writes to predictions.csv.

    Replicates the gid_to_local construction in focal_mask_train.py exactly:
    1. Build raster bounds from feature-store GIDs.
    2. Extend with any target GIDs that fall inside those bounds.
    3. GIDs outside the bounds are silently excluded (same as focal_mask).

    Parameters
    ----------
    features_gids : set of int
        priogrid_gid values from myanmar_feature_store.csv.
    target_gids : set of int
        priogrid_gid values from the target parquet (labelled data).
    """
    def _rowcol(gid):
        return (gid - 1) // 720, (gid - 1) % 720

    rows = [_rowcol(g)[0] for g in features_gids]
    cols = [_rowcol(g)[1] for g in features_gids]
    min_row, min_col = min(rows), min(cols)
    raster_h = max(r - min_row for r in rows) + 1
    raster_w = max(c - min_col for c in cols) + 1
    pad_h = math.ceil(raster_h / 8) * 8
    pad_w = math.ceil(raster_w / 8) * 8

    valid = set(features_gids)
    for gid in target_gids:
        if gid in valid:
            continue
        r0, c0 = _rowcol(gid)
        if 0 <= (r0 - min_row) < pad_h and 0 <= (c0 - min_col) < pad_w:
            valid.add(gid)

    return valid
