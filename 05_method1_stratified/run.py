"""
Stratified MIFS Model — Full Pipeline Runner

Executes all three stages in sequence:
  1. Data Pipeline      — build feature matrix and train/val/test splits
  2. Stratified MIFS    — quota-based pre-filter + greedy MI selection + K-sweep
  3. XGBoost            — grid search hyperparameters, final test evaluation

All output is written to:  data/pipeline_output_stratified/

Usage:
    python run.py
"""

import sys
import json
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output_stratified"

# Ensure imports resolve to this folder's modules
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ─── Imports ─────────────────────────────────────────────────────────────────

from data_pipeline import run_pipeline
from mifs_selection import run_mifs
from xgboost_baseline import (
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

# ─── Step 3: XGBoost — Grid Search + Final Evaluation ───────────────────────

print("\n" + "=" * 70)
print("STEP 3: XGBoost Model (Stratified Features)")
print("=" * 70)

# prepare_data reads mifs_best_k.json; point it at the stratified output dir
X_train, y_train, X_val, y_val, X_test, y_test, feature_names, k = \
    prepare_data(splits, output_dir=OUTPUT_DIR)

best_params, grid_results = grid_search(X_train, y_train, X_val, y_val,
                                        param_grid=PARAM_GRID)

test_rmse, test_acc, y_pred, model = \
    final_evaluation(X_train, y_train, X_test, y_test, best_params)

times_test = splits["test"][2]
summary = save_results(
    best_params, k, feature_names, grid_results,
    test_rmse, test_acc, y_test, y_pred, times_test,
    model=model,
    output_dir=OUTPUT_DIR,
)

# ─── Final Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STRATIFIED MIFS MODEL — FINAL SUMMARY")
print("=" * 70)
print(f"  Model:        {summary['model']}")
print(f"  Features (K): {k}")
print(f"  Best params:  {best_params}")
print(f"  Test RMSE:    {test_rmse:.4f}")
print(f"  Test ACC:     {test_acc:.4f}")
print(f"  Output dir:   {OUTPUT_DIR}")
print("=" * 70)
