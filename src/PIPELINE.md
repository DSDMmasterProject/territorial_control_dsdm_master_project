# Pipeline Documentation

All commands are run from the **project root**.

---

## Overview

Three stages in sequence: data collection → feature engineering → modelling.

```
data/raw/
  ucdp/          ← 3 raw UCDP GED CSV files (manual download)
  acled/         ← acled_myanmar_*.csv (manual download)
  validation/    ← Wikipedia ground truth (fetched by script)

       Stage 1: data_collection/
                     ↓
data/processed/
  myanmar_ucdp_events.csv
  myanmar_priogrid_features.csv
  myanmar_priogrid_labels.csv
  myanmar_training_dataset.csv
  myanmar_prediction_dataset.csv

       Stage 2: feature_engineering/
                     ↓
data/processed/
  acled_wiki_comparison.parquet
  myanmar_feature_store.csv
checkpoints/
  10_target_variable.parquet        ← ⚠ see known issues

       Stage 3: models/
                     ↓
models/focal_mask/
  weights.pt
  predictions.csv
  training_curves.png
  eval_confusion_matrix.png
  eval_transitions.png
```

---

## Stage 1 — Data Collection

Scripts in `src/data_collection/`. Run in this order:

```bash
# 1. Merge 3 raw UCDP GED CSV files into one
python3 src/data_collection/merge_ucdp_sources.py

# 2. Build actor taxonomy (crosswalk for classifying conflict actors)
python3 src/data_collection/build_actor_taxonomy.py

# 3. Fetch Wikipedia temporal snapshots via API (slow — hits the web)
python3 src/data_collection/fetch_wikipedia_ground_truth.py

# 4. Filter and label UCDP events with actor taxonomy
python3 src/data_collection/load_ucdp_events.py

# 5. Aggregate UCDP events → PRIOGRID cell × month feature matrix
python3 src/data_collection/build_priogrid_features.py

# 6. Project Wikipedia labels → PRIOGRID cells (y vector)
python3 src/data_collection/build_priogrid_labels.py

# 7. Join X features + y labels → training and prediction datasets
python3 src/data_collection/build_training_dataset.py

# 8. Filter PRIOGRID catalog (terrain + resource features)
#    Required before build_feature_store.py can include geo features
python3 src/data_collection/filter_priogrid_catalog.py
```

**Prerequisites:** raw UCDP GED CSV files in `data/raw/ucdp/` and ACLED file in `data/raw/acled/`.

---

## Stage 2 — Feature Engineering

Scripts in `src/feature_engineering/`. Run in this order:

```bash
# 1. Preprocess ACLED territorial transfer events → acled_wiki_comparison.parquet
python3 src/feature_engineering/acled_preprocess.py

# 2. Build target variable (3-class: gov / opo / uncertain)
#    Reads acled_wiki_comparison.parquet, applies 6-rule decision tree
python3 src/feature_engineering/target_variable_construction.py

# 3. Build full feature store (base UCDP + extended temporal + geo if available)
#    This is what the model reads from — all features in one place
python3 src/feature_engineering/build_feature_store.py
```

**`build_feature_store.py` adds on top of the base feature matrix:**
- Extended lags: 9 and 12 months (base has 1/3/6)
- Actor-level rolling averages at 3m / 6m / 12m (gov, nug, kio, ula, knu, mndaa, pslf)
- Rolling std at 3m / 6m — captures conflict volatility
- Trend features: roll3m − roll6m (positive = escalating)
- Actor event shares: kio, ula, knu
- Geo features from PRIOGRID catalog (terrain, resources) — merged automatically if `filter_priogrid_catalog.py` has been run

---

## Stage 3 — Models

Scripts in `src/models/`. Train and evaluate are separate steps:

```bash
# Train — runs 200 epochs, saves weights + predictions (~hours on CPU, faster on GPU)
python3 src/models/focal_mask_train.py

# Evaluate — reads predictions.csv, produces metrics + figures (seconds)
python3 src/models/focal_mask_eval.py
```

**Model:** U-Net with ResNet34 encoder (ImageNet pre-trained), 3-class output (gov / opo / uncertain).

**To change which features the model uses:** edit `INPUT_FEATURES` in `focal_mask_train.py`. All columns in `myanmar_feature_store.csv` are valid candidates.

**To create a model variant:** copy both `focal_mask_train.py` and `focal_mask_eval.py` with a new name, then change `INPUT_FEATURES` and `OUT_DIR` in the train script and `OUT_DIR` in the eval script.

---

## Data Sources

| Data | Source | Location |
|---|---|---|
| UCDP GED conflict events | Manual download from ucdp.uu.se | `data/raw/ucdp/` |
| ACLED territorial transfers | Manual download from acleddata.com | `data/raw/acled/` |
| Wikipedia control labels | Fetched by `fetch_wikipedia_ground_truth.py` | `data/raw/validation/` |
| PRIOGRID catalog | Fetched by `filter_priogrid_catalog.py` | `data/processed/` |
