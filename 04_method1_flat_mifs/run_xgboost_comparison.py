"""
Run MIFS (Step 2) + XGBoost Baseline (Step 3) using actual XGBoost.
Saves outputs to a separate folder for comparison against HistGradientBoosting.
"""

import sys
import json
import shutil
from pathlib import Path

# Force XGBoost — fail fast if not available
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output"
XGBOOST_DIR = BASE_DIR / "data" / "pipeline_output_xgboost"
HGBR_DIR = BASE_DIR / "data" / "pipeline_output_histgb"

# ─── 1. Back up existing HistGradientBoosting results ────────────────────────

if OUTPUT_DIR.exists():
    HGBR_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("mifs_*"):
        shutil.copy2(f, HGBR_DIR / f.name)
    for f in OUTPUT_DIR.glob("xgb_*"):
        shutil.copy2(f, HGBR_DIR / f.name)
    print(f"Backed up HistGB results to {HGBR_DIR.name}/")

# ─── 2. Run pipeline ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)
splits, _ = run_pipeline(save=True)

# ─── 3. Run MIFS with XGBoost for K-sweep ────────────────────────────────────

from mifs_selection import (
    compute_relevance, prefilter_candidates, greedy_mifs,
    save_mifs_results, physical_sanity_check,
    N_CANDIDATES, MAX_K
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

print("\n" + "=" * 70)
print("STEP 2: MIFS Feature Selection (with XGBoost)")
print("=" * 70)

X_train, y_train, _ = splits["train"]
X_val, y_val, _ = splits["val"]

relevance = compute_relevance(X_train, y_train)
top_relevance = prefilter_candidates(relevance, n_candidates=N_CANDIDATES)
selected_order = greedy_mifs(X_train, y_train, top_relevance, max_k=MAX_K)

# K-sweep with actual XGBoost
print(f"\n[MIFS-4] K-sweep with XGBoost (actual)...")
feature_names_ordered = [name for name, _ in selected_order]
k_values = [1] + list(range(2, min(MAX_K, len(feature_names_ordered)) + 1, 2))
k_results = []
best_rmse = np.inf
best_k = k_values[0]

for k in k_values:
    top_k = feature_names_ordered[:k]
    Xtr_k = np.nan_to_num(X_train[top_k].values, nan=0.0)
    Xval_k = np.nan_to_num(X_val[top_k].values, nan=0.0)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                         random_state=42, verbosity=0)
    model.fit(Xtr_k, y_train)
    y_pred = model.predict(Xval_k)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    k_results.append((k, rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_k = k
    print(f"         K={k:3d}  val RMSE={rmse:.4f}"
          f"{'  ← best' if rmse == best_rmse else ''}")

print(f"\n         Best K = {best_k} (val RMSE = {best_rmse:.4f})")

final_features = save_mifs_results(relevance, selected_order, best_k, k_results)
physical_sanity_check(final_features)

# ─── 4. XGBoost grid search & final eval ─────────────────────────────────────

from xgboost_baseline import (
    prepare_data, anomaly_correlation, save_results,
    PARAM_GRID,
)
from itertools import product

print("\n" + "=" * 70)
print("STEP 3: XGBoost Baseline Model (actual XGBoost)")
print("=" * 70)

X_tr, y_tr, X_v, y_v, X_te, y_te, feat_names, bk = prepare_data(splits)

# Grid search with XGBoost
print("\n[XGB-2] Grid search (XGBoost)...")
keys = list(PARAM_GRID.keys())
combos = list(product(*PARAM_GRID.values()))
print(f"         {len(combos)} combinations")

grid_results = []
best_gs_rmse = np.inf
best_params = None

for i, values in enumerate(combos):
    params = dict(zip(keys, values))
    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    yp = model.predict(X_v)
    rmse = np.sqrt(mean_squared_error(y_v, yp))
    acc = anomaly_correlation(y_v, yp)
    grid_results.append({**params, "val_rmse": rmse, "val_acc": acc})
    if rmse < best_gs_rmse:
        best_gs_rmse = rmse
        best_params = params
    if (i + 1) % 16 == 0:
        print(f"         [{i+1}/{len(combos)}] best: RMSE={best_gs_rmse:.4f} {best_params}")

print(f"\n         Best params: {best_params}")
print(f"         Best val RMSE: {best_gs_rmse:.4f}")
grid_results.sort(key=lambda x: x["val_rmse"])

# Final test eval
print("\n[XGB-3] Final test evaluation...")
final_model = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    random_state=42, verbosity=0,
)
final_model.fit(X_tr, y_tr)
y_pred_test = final_model.predict(X_te)
test_rmse = np.sqrt(mean_squared_error(y_te, y_pred_test))
test_acc = anomaly_correlation(y_te, y_pred_test)

print(f"         Test RMSE: {test_rmse:.4f}")
print(f"         Test ACC:  {test_acc:.4f}")

# Save
times_test = splits["test"][2]
summary = save_results(
    best_params, bk, feat_names, grid_results,
    test_rmse, test_acc, y_te, y_pred_test, times_test,
)

# ─── 5. Copy XGBoost results to comparison folder ────────────────────────────

XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
for f in OUTPUT_DIR.glob("mifs_*"):
    shutil.copy2(f, XGBOOST_DIR / f.name)
for f in OUTPUT_DIR.glob("xgb_*"):
    shutil.copy2(f, XGBOOST_DIR / f.name)
print(f"\nXGBoost results copied to {XGBOOST_DIR.name}/")

# ─── 6. Comparison summary ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON: HistGradientBoosting vs XGBoost")
print("=" * 70)

hgb_summary_path = HGBR_DIR / "xgb_final_summary.json"
if hgb_summary_path.exists():
    with open(hgb_summary_path) as f:
        hgb = json.load(f)
    print(f"  {'Metric':<20s} {'HistGB':>12s} {'XGBoost':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'Test RMSE':<20s} {hgb['test_rmse']:>12.4f} {test_rmse:>12.4f}")
    print(f"  {'Test ACC':<20s} {hgb['test_acc']:>12.4f} {test_acc:>12.4f}")
    print(f"  {'Best K':<20s} {hgb['best_k']:>12d} {bk:>12d}")
    print(f"  {'Best params':<20s}")
    print(f"    HistGB:  {hgb['best_params']}")
    print(f"    XGBoost: {best_params}")
else:
    print("  No HistGB results found for comparison.")

print("=" * 70)
