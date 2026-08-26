"""
Run Tuned MIFS + XGBoost Pipeline

Runs both the tuned MIFS feature selection and tuned XGBoost model.
All outputs are saved to:  data/pipeline_output_tuned/

Usage:
    cd "Tuned Model"
    python run_tuned.py

At the end it prints a comparison table against the baseline results
(from data/pipeline_output_xgboost/) if that folder exists.
"""

import sys
import json
from pathlib import Path

# Allow imports from Baseline Model (data_pipeline lives there)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "Baseline Model"))

from data_pipeline import run_pipeline
from mifs_selection_tuned import run_mifs
from xgboost_tuned import run_xgboost_tuned

OUTPUT_DIR   = BASE_DIR / "data" / "pipeline_output_tuned"
BASELINE_DIR = BASE_DIR / "data" / "pipeline_output_xgboost"

# ─── Step 1: Data Pipeline ───────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)
splits, _ = run_pipeline(save=True)

# ─── Step 2: Tuned MIFS ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Tuned MIFS Feature Selection")
print("  MI_N_NEIGHBORS=7  |  N_CANDIDATES=400  |  MAX_K=80  |  β=1.0")
print("=" * 70)
final_features, best_k = run_mifs(splits, output_dir=OUTPUT_DIR)

# ─── Step 3: Tuned XGBoost ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Tuned XGBoost")
print("  + subsample  + colsample_bytree  + min_child_weight  + reg_alpha")
print("=" * 70)
summary = run_xgboost_tuned(splits, output_dir=OUTPUT_DIR)

# ─── Comparison against baseline ─────────────────────────────────────────────

baseline_path = BASELINE_DIR / "xgb_final_summary.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline = json.load(f)

    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Tuned")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Baseline':>12s} {'Tuned':>12s} {'Δ':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    b_rmse = baseline["test_rmse"]
    t_rmse = summary["test_rmse"]
    b_acc  = baseline["test_acc"]
    t_acc  = summary["test_acc"]
    b_k    = baseline["best_k"]
    t_k    = summary["best_k"]

    print(f"  {'Test RMSE':<25s} {b_rmse:>12.4f} {t_rmse:>12.4f} "
          f"{'↓' if t_rmse < b_rmse else '↑'}{abs(t_rmse - b_rmse):>8.4f}")
    print(f"  {'Test ACC':<25s} {b_acc:>12.4f} {t_acc:>12.4f} "
          f"{'↑' if t_acc > b_acc else '↓'}{abs(t_acc - b_acc):>8.4f}")
    print(f"  {'Features (K)':<25s} {b_k:>12d} {t_k:>12d}")

    b_params = baseline.get("best_params", {})
    t_params = summary.get("best_params", {})
    print(f"\n  Baseline best params: {b_params}")
    print(f"  Tuned    best params: {t_params}")
    print("=" * 70)
else:
    print(f"\nBaseline results not found at {BASELINE_DIR} — skipping comparison.")

print(f"\nAll tuned outputs saved to: {OUTPUT_DIR}")
