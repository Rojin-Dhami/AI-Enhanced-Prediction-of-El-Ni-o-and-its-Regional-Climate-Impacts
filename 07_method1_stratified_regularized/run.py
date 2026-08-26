"""
Stratified Regularized Model — Full Pipeline Runner

Combines:
  - Stratified quota-based MIFS prefiltering (Section 3.4.5)
    Per-variable-type quotas ensure weaker-but-mechanistically-important
    predictors (OHC, thermocline, wind stress) are not crowded out by SST.
  - Anti-overfitting regularized XGBoost
    Shallow trees (max_depth ≤ 3), strong min_child_weight, L1/L2 penalties,
    subsample and colsample_bytree restrictions.

All outputs saved to:
  results/method1_stratified_regularized/

Usage:
    python run.py
"""

import sys
import json
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

THIS_DIR   = Path(__file__).resolve().parent
BASE_DIR   = THIS_DIR.parent
OUTPUT_DIR = BASE_DIR / "results" / "method1_stratified_regularized"

# Ensure imports resolve to this folder's modules
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ─── Imports ─────────────────────────────────────────────────────────────────

from data_pipeline      import run_pipeline
from mifs_selection     import run_mifs
from xgboost_regularized import (
    prepare_data,
    grid_search,
    final_evaluation,
    save_results,
    PARAM_GRID,
)

# ─── Step 1: Data Pipeline ───────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)

splits, meta = run_pipeline(save=True)

# ─── Step 2: Stratified MIFS Feature Selection ───────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Stratified MIFS Feature Selection")
print("=" * 70)

final_features, best_k = run_mifs(splits)

# ─── Step 3: Regularized XGBoost — Grid Search + Final Evaluation ────────────

print("\n" + "=" * 70)
print("STEP 3: Regularized XGBoost (Anti-Overfitting Grid)")
print("=" * 70)

# prepare_data reads mifs_best_k.json from the output dir
X_train, y_train, X_val, y_val, X_test, y_test, feature_names, k = \
    prepare_data(splits, output_dir=OUTPUT_DIR)

best_params, grid_results = grid_search(
    X_train, y_train, X_val, y_val, param_grid=PARAM_GRID
)

(train_rmse, val_rmse, test_rmse,
 train_acc,  val_acc,  test_acc,
 y_pred_test, model) = final_evaluation(
    X_train, y_train, X_val, y_val, X_test, y_test, best_params
)

times_test = splits["test"][2]
summary = save_results(
    best_params, k, feature_names, grid_results,
    train_rmse, val_rmse, test_rmse,
    train_acc,  val_acc,  test_acc,
    y_test, y_pred_test, times_test,
    model=model,
    output_dir=OUTPUT_DIR,
)

# ─── Final Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STRATIFIED REGULARIZED MODEL — FINAL SUMMARY")
print("=" * 70)
print(f"  Model:        {summary['model']}")
print(f"  Features (K): {k}")
print(f"  Best params:  {best_params}")
print(f"  Train RMSE:   {train_rmse:.4f}   Train ACC: {train_acc:.4f}")
print(f"  Val   RMSE:   {val_rmse:.4f}   Val   ACC: {val_acc:.4f}")
print(f"  Test  RMSE:   {test_rmse:.4f}   Test  ACC: {test_acc:.4f}")
print(f"  Train→Test gap: {test_rmse - train_rmse:+.4f}")
print(f"  Output dir:   {OUTPUT_DIR}")
print("=" * 70)
