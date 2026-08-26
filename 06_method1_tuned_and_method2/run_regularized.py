"""
Anti-Overfitting XGBoost Run

The tuned model (pipeline_output_tuned) achieved Train RMSE=0.0005 / ACC=1.00,
meaning it memorised the training set entirely. This script re-runs the grid
search with a grid specifically designed to prevent overfitting:

  - max_depth      : [2, 3]        ← much shallower trees (was up to 6)
  - n_estimators   : [100, 200, 300]
  - learning_rate  : [0.01, 0.05, 0.1]
  - subsample      : [0.6, 0.7, 0.8]
  - colsample_bytree: [0.6, 0.7, 0.8]
  - min_child_weight: [5, 10, 20]   ← much higher (was 1–5)
  - reg_alpha      : [0, 0.5, 1.0]  ← L1
  - reg_lambda     : [1, 5, 10]     ← L2 (new)

Reuses MIFS features from pipeline_output_tuned (β=1.0, K=18).
Outputs saved to: data/pipeline_output_regularized/
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "Baseline Model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import run_pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

MIFS_DIR   = BASE_DIR / "data" / "pipeline_output_tuned"   # reuse β=1.0 features
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output_regularized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Anti-overfitting grid ────────────────────────────────────────────────────

PARAM_GRID = {
    "max_depth":          [2, 3],
    "n_estimators":       [100, 200, 300],
    "learning_rate":      [0.01, 0.05, 0.1],
    "subsample":          [0.6, 0.7, 0.8],
    "colsample_bytree":   [0.6, 0.7, 0.8],
    "min_child_weight":   [5, 10, 20],
    "reg_alpha":          [0, 0.5, 1.0],
    "reg_lambda":         [1, 5, 10],
}


def make_model(**params):
    if HAS_XGBOOST:
        return XGBRegressor(**params, random_state=42, verbosity=0)
    return HistGradientBoostingRegressor(
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_samples_leaf=max(1, params["min_child_weight"] * 5),
        l2_regularization=params.get("reg_lambda", 1),
        random_state=42,
    )


def acc(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ─── Step 1: Load data ────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)
splits, _ = run_pipeline(save=False)

with open(MIFS_DIR / "mifs_best_k.json") as f:
    mifs = json.load(f)
feature_names = mifs["selected_features"]
best_k        = mifs["best_k"]
print(f"\nReusing {best_k} MIFS features from pipeline_output_tuned (β=1.0)")

X_train = np.nan_to_num(splits["train"][0][feature_names].values, nan=0.0)
y_train = splits["train"][1].values
X_val   = np.nan_to_num(splits["val"][0][feature_names].values,   nan=0.0)
y_val   = splits["val"][1].values
X_test  = np.nan_to_num(splits["test"][0][feature_names].values,  nan=0.0)
y_test  = splits["test"][1].values
times_test = splits["test"][2]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ─── Step 2: Grid search ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Anti-Overfitting Grid Search")
print("=" * 70)

keys   = list(PARAM_GRID.keys())
combos = list(product(*PARAM_GRID.values()))
print(f"  {len(combos)} combinations | focus: shallow trees + strong regularization\n")

grid_results = []
best_val_rmse = np.inf
best_params   = None

for i, values in enumerate(combos):
    params = dict(zip(keys, values))
    model  = make_model(**params)
    model.fit(X_train, y_train)

    yp_val = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, yp_val))
    val_acc_ = acc(y_val, yp_val)

    # Also check train RMSE to flag overfitting during search
    yp_tr  = model.predict(X_train)
    tr_rmse = np.sqrt(mean_squared_error(y_train, yp_tr))

    grid_results.append({**params,
                         "train_rmse": tr_rmse,
                         "val_rmse":   val_rmse,
                         "val_acc":    val_acc_})

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params   = params

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(combos)}] best val RMSE={best_val_rmse:.4f}  {best_params}")

print(f"\n  Best params : {best_params}")
print(f"  Best val RMSE: {best_val_rmse:.4f}")

# Save grid results
grid_df = pd.DataFrame(grid_results).sort_values("val_rmse")
grid_df.to_csv(OUTPUT_DIR / "xgb_grid_search.csv", index=False)

# ─── Step 3: Final model ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Final Evaluation")
print("=" * 70)

final_model = make_model(**best_params)
final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
y_pred_val   = final_model.predict(X_val)
y_pred_test  = final_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse_  = np.sqrt(mean_squared_error(y_val,   y_pred_val))
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
train_acc_ = acc(y_train, y_pred_train)
val_acc_   = acc(y_val,   y_pred_val)
test_acc_  = acc(y_test,  y_pred_test)

print(f"\n  {'Split':<12s} {'RMSE':>10s} {'ACC':>10s}")
print(f"  {'-'*12} {'-'*10} {'-'*10}")
print(f"  {'Train':<12s} {train_rmse:>10.4f} {train_acc_:>10.4f}")
print(f"  {'Val':<12s} {val_rmse_:>10.4f} {val_acc_:>10.4f}")
print(f"  {'Test':<12s} {test_rmse:>10.4f} {test_acc_:>10.4f}")
print(f"\n  Train→Test RMSE gap: {test_rmse - train_rmse:+.4f}  "
      f"(was +0.8900 before regularization)")

# ─── Step 4: Compare with previous runs ───────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON ACROSS RUNS")
print("=" * 70)
print(f"  {'Run':<25s} {'Train RMSE':>11s} {'Val RMSE':>10s} {'Test RMSE':>10s} {'Test ACC':>10s}")
print(f"  {'-'*25} {'-'*11} {'-'*10} {'-'*10} {'-'*10}")

# Load previous summaries if available
for label, path in [
    ("Baseline (XGBoost)", BASE_DIR / "data/pipeline_output_xgboost/xgb_final_summary.json"),
    ("Tuned (β=1.0)",      BASE_DIR / "data/pipeline_output_tuned/xgb_final_summary.json"),
]:
    if path.exists():
        with open(path) as f:
            s = json.load(f)
        print(f"  {label:<25s} {'N/A':>11s} {'N/A':>10s} {s['test_rmse']:>10.4f} {s['test_acc']:>10.4f}")

print(f"  {'Regularized'::<25s} {train_rmse:>11.4f} {val_rmse_:>10.4f} {test_rmse:>10.4f} {test_acc_:>10.4f}")

# ─── Step 5: Save outputs ────────────────────────────────────────────────────

pred_df = pd.DataFrame({
    "window_end":       times_test,
    "actual_nino34":    y_test,
    "predicted_nino34": y_pred_test,
})
pred_df.to_csv(OUTPUT_DIR / "xgb_test_predictions.csv", index=False)

summary = {
    "model": "XGBRegressor (regularized)" if HAS_XGBOOST
             else "HistGradientBoostingRegressor (regularized)",
    "best_k":     best_k,
    "best_params": best_params,
    "train_rmse": round(train_rmse, 4),
    "train_acc":  round(train_acc_, 4),
    "val_rmse":   round(val_rmse_,  4),
    "val_acc":    round(val_acc_,   4),
    "test_rmse":  round(test_rmse,  4),
    "test_acc":   round(test_acc_,  4),
    "selected_features": feature_names,
    "train_samples": len(y_train),
    "val_samples":   len(y_val),
    "test_samples":  len(y_test),
    "notes": (
        "Anti-overfitting grid: max_depth<=3, min_child_weight up to 20, "
        "reg_lambda added, subsample/colsample restricted. "
        "MIFS features reused from pipeline_output_tuned (beta=1.0, K=18)."
    ),
}
with open(OUTPUT_DIR / "xgb_final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

joblib.dump(final_model, OUTPUT_DIR / "xgb_regularized_model.joblib")

print(f"\n  Saved to: {OUTPUT_DIR.name}/")
print(f"    xgb_final_summary.json")
print(f"    xgb_test_predictions.csv")
print(f"    xgb_regularized_model.joblib")
print(f"    xgb_grid_search.csv")
