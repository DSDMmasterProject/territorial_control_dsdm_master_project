#!/usr/bin/env python3
"""
Train and evaluate conflict prediction models.

Target: did this cell change faction next month? (1=yes, 0=no)

Models:
    1. Logistic Regression  — simple, interpretable baseline
    2. Random Forest        — captures non-linear patterns, feature importance
    3. Gradient Boosting    — typically best performance

Usage:
    python train_models.py

Output:
    model/results/model_comparison.csv
    model/results/feature_importance.csv
    model/results/predictions.csv

Requirements:
    pip install pandas numpy scikit-learn
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'model', 'feature_matrix.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'model', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'is_pro_junta', 'months_in_current_faction',
    'n_events', 'total_deaths', 'n_unique_groups',
    'junta_events', 'anti_junta_events',
    'flag_junta', 'flag_pdf', 'flag_kia', 'flag_aa',
    'flag_karen', 'flag_karenni', 'flag_tnla', 'flag_mndaa',
    'flag_chin_resistance',
    'junta_event_share', 'anti_junta_event_share',
    'n_events_lag1', 'n_events_lag3', 'n_events_lag6',
    'n_events_roll3', 'n_events_roll6',
    'total_deaths_lag1', 'total_deaths_lag3', 'total_deaths_lag6',
    'total_deaths_roll3', 'total_deaths_roll6',
    'event_trend_3m', 'months_since_last_event',
    'neighbor_n_events', 'neighbor_changed_faction',
    'attacker_matches_controller',
]
TARGET_COL = 'changed'

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading feature matrix...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df):,} rows, {len(FEATURE_COLS)} features")

train = df[df['split'] == 'train'].copy()
val   = df[df['split'] == 'validate'].copy()
print(f"  Train: {len(train):,} | Validate: {len(val):,}")
print(f"  Change rate — Train: {train[TARGET_COL].mean()*100:.2f}% | Validate: {val[TARGET_COL].mean()*100:.2f}%")

train[FEATURE_COLS] = train[FEATURE_COLS].fillna(0)
val[FEATURE_COLS]   = val[FEATURE_COLS].fillna(0)

X_train = train[FEATURE_COLS].values
y_train = train[TARGET_COL].values
X_val   = val[FEATURE_COLS].values
y_val   = val[TARGET_COL].values

# Baseline: always predict no change
baseline_acc = 1 - val[TARGET_COL].mean()
print(f"\nBaseline (always predict no change): {baseline_acc:.4f}")
print(f"Baseline ROC-AUC: 0.5000")

# ── Models ────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5,
        learning_rate=0.1, random_state=42
    ),
}

results        = []
all_importances = []
val_preds_df   = val[['lat','lon','year_month','level1_side','level1_side_next', TARGET_COL]].copy()

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"  Accuracy : {acc:.4f}  (baseline: {baseline_acc:.4f})")
    print(f"  ROC-AUC  : {auc:.4f}  (baseline: 0.5000)")
    print(f"\n  Classification report:")
    print(classification_report(y_val, y_pred,
                                target_names=['No change', 'Changed'],
                                digits=3))

    cm = confusion_matrix(y_val, y_pred)
    print(f"  Confusion matrix:")
    print(f"                    Predicted no change  Predicted change")
    print(f"  Actual no change  {cm[0][0]:>19}  {cm[0][1]:>16}")
    print(f"  Actual changed    {cm[1][0]:>19}  {cm[1][1]:>16}")

    results.append({
        'model':    name,
        'accuracy': round(acc, 4),
        'roc_auc':  round(auc, 4),
        'baseline_acc': round(baseline_acc, 4),
        'recall_changed': round(
            cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0]+cm[1][1]) > 0 else 0, 4
        ),
        'precision_changed': round(
            cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1]+cm[1][1]) > 0 else 0, 4
        ),
    })

    val_preds_df[f'pred_{name.replace(" ","_").lower()}'] = y_pred
    val_preds_df[f'prob_{name.replace(" ","_").lower()}'] = y_prob

    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature':    FEATURE_COLS,
            'importance': model.feature_importances_,
            'model':      name
        }).sort_values('importance', ascending=False)
        all_importances.append(imp)
        print(f"\n  Top 15 features:")
        print(imp.head(15)[['feature','importance']].to_string(index=False))

    elif hasattr(model, 'coef_'):
        coef = pd.DataFrame({
            'feature':     FEATURE_COLS,
            'coefficient': model.coef_[0],
        })
        coef['abs_coef'] = coef['coefficient'].abs()
        coef = coef.sort_values('abs_coef', ascending=False)
        print(f"\n  Top 15 features (by abs coefficient):")
        print(coef.head(15)[['feature','coefficient']].to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)
if all_importances:
    pd.concat(all_importances).to_csv(
        os.path.join(OUT_DIR, 'feature_importance.csv'), index=False
    )
val_preds_df.to_csv(os.path.join(OUT_DIR, 'predictions.csv'), index=False)

print("\nFINAL SUMMARY")
print(f"{'='*60}")
print(f"  Baseline accuracy  : {baseline_acc:.4f}")
print(f"  Baseline ROC-AUC   : 0.5000")
print(f"\n  {'Model':<25} {'Accuracy':>9} {'ROC-AUC':>9} {'Recall(change)':>15} {'Precision(change)':>18}")
print(f"  {'-'*78}")
for _, r in results_df.iterrows():
    print(f"  {r['model']:<25} {r['accuracy']:>9.4f} {r['roc_auc']:>9.4f} "
          f"{r['recall_changed']:>15.4f} {r['precision_changed']:>18.4f}")
