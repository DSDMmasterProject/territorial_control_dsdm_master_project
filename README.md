# Territorial Control Prediction — Myanmar

Predicting month-by-month territorial control (government / opposition / uncertain) at
PRIOGRID cell level in Myanmar, using conflict event data from UCDP and ACLED alongside
Wikipedia-derived ground truth labels. Developed as the Master's thesis project for the
BSE MSc in Data Science for Decision Making (2025–2026), in response to the World Bank
Fragility, Conflict, and Violence (FCV) initiative on conflict monitoring.

**Authors:** Corneel Moons · Felipe Manzi · Tizian Schenk

---

## Main results

The production U-Net model (ResNet-34 backbone, 10 conflict-event features) is competitive
with the no-change persistence baseline on overall cell-month accuracy, and outperforms it
on the 31 transition cell-months in the validation set — exactly where persistence fails
by definition. Spatial cross-validation reveals limited geographic generalisation: the model
learns patterns from cells it has seen during training but struggles on held-out spatial
blocks.

---

## Repository structure

```
.
├── src/
│   ├── data_collection/        # Downloads and aggregates raw UCDP + Wikipedia data
│   ├── feature_engineering/    # ACLED preprocessing, target cascade, feature store
│   ├── models/                 # U-Net training, baselines, CV, ablation, evaluation
│   └── visualizations/         # Publication figures, reporting notebook, pub_style.py
│
├── data/
│   ├── raw/                    # Original downloaded files (not committed — see below)
│   │   ├── ucdp/               # GED annual + Candidate CSVs
│   │   ├── acled/              # ACLED Myanmar CSVs (one per download date)
│   │   ├── gadm/               # gadm41_MMR.gpkg — Myanmar admin boundaries
│   │   └── priogrid_static.csv / priogrid_yearly.csv  (optional)
│   └── processed/              # Pipeline checkpoints (parquet/csv, reproducible)
│
├── models/
│   ├── focal_mask/             # Production model: weights.pt + predictions.csv
│   ├── baseline_no_change/     # Persistence baseline predictions
│   ├── baseline_random_proportion/
│   └── feature_ablation_results.csv
│
├── notebooks/                  # Experimental notebooks (not part of production pipeline)
├── src/reporting/figures/      # Publication-quality figures (PDF + PNG)
└── pyproject.toml              # All Python dependencies
```

---

## Raw data required

The following files must be downloaded manually before running the pipeline.
Processed checkpoints in `data/processed/` are committed to the repo and allow
skipping re-collection if you trust the existing ground truth.

| File / pattern | Source | Required? | Notes |
|----------------|--------|-----------|-------|
| `data/raw/ucdp/GEDEvent_v25_1.csv` | [ucdp.uu.se](https://ucdp.uu.se/downloads/) (GED 25.1 annual) | **Yes** | ~250 MB; covers 1989–2024 |
| `data/raw/ucdp/GEDEvent_v25_01_25_12.csv` | ucdp.uu.se (Candidate Jan–Dec 2025) | **Yes** | Currently missing from repo — `merge_ucdp_sources.py` will fail without it |
| `data/raw/ucdp/GEDEvent_v26_01_26_03.csv` | ucdp.uu.se (Candidate Q1 2026) | **Yes** | Extends panel to Mar 2026 |
| `data/raw/acled/acled_myanmar_YYYY-MM-DD.csv` | [acleddata.com](https://developer.acleddata.com/) | **Yes** | Requires free registration + API key. The pipeline picks the file with the most recent date in the filename. |
| `data/raw/gadm/gadm41_MMR.gpkg` | [gadm.org](https://gadm.org/download_country.html) | **Yes** | Myanmar admin boundaries (layers ADM_ADM_0 and ADM_ADM_1) |
| `data/raw/priogrid_static.csv` | [grid.prio.org](https://grid.prio.org/) | Optional | Static geographic covariates; extends feature store from 10 to ~93 features |
| `data/raw/priogrid_yearly.csv` | grid.prio.org | Optional | Annual geographic covariates; same use case |

> **Note on PRIOGRID files:** if absent, `build_feature_store.py` runs without them
> and produces a feature store with only the UCDP-derived features. The production
> model uses only the 10 UCDP features listed below and does **not** require these files.

---

## Setup

Requires **Python 3.11+**. All dependencies are declared in `pyproject.toml`.

```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create environment and install all dependencies
uv sync

# Activate (or prefix commands with `uv run`)
source .venv/bin/activate
```

Key packages: `torch>=2.1`, `segmentation-models-pytorch`, `geopandas`, `pandas`,
`scikit-learn`, `scipy`, `matplotlib`.

---

## How to run

The pipeline has five stages. Run them in order from the project root. For per-script
details and flags, see `src/pipeline.md`.

### Stage 1 — Feature store
```bash
python src/data_collection/build_actor_taxonomy.py
python src/data_collection/merge_ucdp_sources.py
python src/data_collection/load_ucdp_events.py
python src/data_collection/build_priogrid_features.py
python src/feature_engineering/build_feature_store.py
# Optional: filter_priogrid_catalog.py (only needed for geo features)
```
Produces `data/processed/myanmar_feature_store.csv`.

### Stage 2 — Target variable
```bash
python src/data_collection/fetch_wikipedia_ground_truth.py   # ~5 min, requires internet
python src/data_collection/build_priogrid_labels.py
python src/feature_engineering/acled_preprocess.py
python src/feature_engineering/target_variable_construction.py
```
Produces `data/processed/10_target_variable.parquet` — 5,238 cell-months, 185 cells,
Nov 2023–Mar 2026, 6-rule decision cascade (Wikipedia + ACLED).

### Stage 3 — Training
```bash
python src/models/focal_mask_train.py                 # ~5–10 min on GPU/MPS
python src/models/baseline_no_change.py
python src/models/baseline_random_proportion.py
```
Produces `models/focal_mask/weights.pt` and `predictions.csv` for all three models.

### Stage 4 — Evaluation
```bash
python src/models/focal_mask_eval.py
python src/models/evaluate_baselines.py
python src/models/spatial_cv_production.py            # ~20–30 min
python src/models/feature_ablation.py                 # ~5 min per variant × 4 variants
```
Produces `models/baseline_comparison.csv`, `models/spatial_cv_results.csv`,
`models/feature_ablation_results.csv`.

### Stage 5 — Figures
```bash
python src/visualizations/export_pub_figures.py
python src/visualizations/export_figure_e.py
# Then run src/visualizations/reporting.ipynb (Jupyter) to generate C2–C6 figures
```
Outputs go to `src/reporting/figures/` as PDF + PNG at 300 dpi.

---

## Production configuration

The model reported in the paper is defined by the following constants, which are
hardcoded consistently across `focal_mask_train.py`, `spatial_cv_production.py`,
`feature_ablation.py`, and the baselines:

```
TRAIN_CUTOFF     = '2025-09'          # val set: Oct 2025 – Mar 2026 (6 months)
SEED             = 20269999
NEIGHBOR_RADIUS  = 1                  # focal mask dilation radius (queen neighbors)
FOCAL_OUT_WEIGHT = 0.0                # CE weight outside focal region

# Loss
combined_loss    = 0.7 × CrossEntropy + 0.3 × DiceLoss   (class-weighted)

# Architecture
encoder          = resnet34 (ImageNet init, frozen for first 30 epochs)
encoder_depth    = 4
decoder_channels = (128, 64, 32, 16)
NUM_EPOCHS       = 200

# Input features (10)
INPUT_FEATURES = [
    'total_events', 'total_fatalities',
    'events_gov', 'events_nug', 'events_ula', 'events_kio',
    'gov_vs_civilians', 'gov_event_share',
    'total_events_lag1', 'total_fatalities_lag1',
]
```

The panel covers 185 PRIOGRID cells; the model uses **183** — two cells
(`gid=160401`, `gid=161121` at lon=100.25°E) fall outside the raster bounds
and are silently dropped. Both are stable Wikipedia-only cells with no
transitions; their exclusion does not affect metrics.

---

## Known limitations

- **`selected_features_32.csv`** (used by `feature_ablation.py` and
  `feature_selection.py`) is generated by the experimental notebook
  `notebooks/10_focal_mask_all_features.ipynb`. There is no standalone script
  in `src/` to reproduce it. The file is committed to `data/processed/` so the
  ablation can be reproduced without re-running the notebook, but re-generating
  it from scratch requires running the notebook manually.

- **`TRAIN_CUTOFF` is hardcoded** in five scripts (`focal_mask_train.py`,
  `focal_mask_export_predictions.py`, `spatial_cv_production.py`,
  `feature_ablation.py`, `baseline_*.py`). Changing the cutoff for a new
  experiment requires editing all five files.

- **`fetch_wikipedia_ground_truth.py`** makes ~30 Wikipedia API requests with
  a 1.5-second sleep between each. It takes several minutes and requires
  internet connectivity. The output (`myanmar_wikipedia_temporal_groundtruth.csv`)
  is committed to the repo, so re-running this script is only necessary when
  extending the panel to new months.

- **`focal_mask_export_predictions.py`** is an alternative to re-running the
  full training loop when weights already exist. Use it only with the committed
  `weights.pt`; do not use it to re-export predictions from a different run
  without verifying the config matches.
