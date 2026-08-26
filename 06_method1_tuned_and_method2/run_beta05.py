"""
Run MIFS + XGBoost with REDUNDANCY_BETA = 0.5

Lower beta weakens the redundancy penalty in the MIFS score:
  S(X_ij) = I(X_ij; Y) - 0.5 * (1/|S|) * Σ I(X_ij; X_kl)

This allows more correlated features (e.g. SST grid points) to be selected
alongside thermocline/OHC variables, rather than being penalised out.

All outputs saved to: data/pipeline_output_beta05/
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "Baseline Model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import run_pipeline
from mifs_selection_tuned import run_mifs
from xgboost_tuned import run_xgboost_tuned

BETA       = 0.5
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output_beta05"
TUNED_DIR  = BASE_DIR / "data" / "pipeline_output_tuned"   # beta=1.0 baseline

# ─── Step 1: Data Pipeline ───────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)
splits, _ = run_pipeline(save=True)

# ─── Step 2: MIFS with β = 0.5 ───────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"STEP 2: MIFS Feature Selection  (β = {BETA})")
print("  Lower redundancy penalty → allows more SST grid points through")
print("=" * 70)
final_features, best_k = run_mifs(splits, output_dir=OUTPUT_DIR, beta=BETA)

# ─── Step 3: XGBoost ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Tuned XGBoost")
print("=" * 70)
summary = run_xgboost_tuned(splits, output_dir=OUTPUT_DIR)

# Save model
import joblib
import numpy as np
from xgboost_tuned import prepare_data, _make_model

X_train, y_train, *_ = prepare_data(splits, output_dir=OUTPUT_DIR)[:3]
best_params = summary["best_params"]
model = _make_model(**best_params)
model.fit(X_train, y_train)
joblib.dump(model, OUTPUT_DIR / "xgb_beta05_model.joblib")
print(f"Trained model saved to xgb_beta05_model.joblib")

# ─── Comparison: β=1.0 vs β=0.5 ─────────────────────────────────────────────

tuned_path = TUNED_DIR / "xgb_final_summary.json"
if tuned_path.exists():
    with open(tuned_path) as f:
        tuned = json.load(f)

    print("\n" + "=" * 70)
    print("COMPARISON: β = 1.0  vs  β = 0.5")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'β=1.0 (tuned)':>14s} {'β=0.5':>10s} {'Δ':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*10}")

    b_rmse = tuned["test_rmse"];   t_rmse = summary["test_rmse"]
    b_acc  = tuned["test_acc"];    t_acc  = summary["test_acc"]
    b_k    = tuned["best_k"];      t_k    = summary["best_k"]

    print(f"  {'Test RMSE':<25s} {b_rmse:>14.4f} {t_rmse:>10.4f} "
          f"{'↓' if t_rmse < b_rmse else '↑'}{abs(t_rmse - b_rmse):>8.4f}")
    print(f"  {'Test ACC':<25s} {b_acc:>14.4f} {t_acc:>10.4f} "
          f"{'↑' if t_acc > b_acc else '↓'}{abs(t_acc - b_acc):>8.4f}")
    print(f"  {'Features (K)':<25s} {b_k:>14d} {t_k:>10d}")

    # Show SST count in each
    def count_sst(features):
        return sum(1 for f in features if "sst_anom_z" in f)

    b_sst = count_sst(tuned["selected_features"])
    t_sst = count_sst(summary["selected_features"])
    print(f"  {'SST features selected':<25s} {b_sst:>14d} {t_sst:>10d}")
    print("=" * 70)

print(f"\nAll β=0.5 outputs saved to: {OUTPUT_DIR}")
