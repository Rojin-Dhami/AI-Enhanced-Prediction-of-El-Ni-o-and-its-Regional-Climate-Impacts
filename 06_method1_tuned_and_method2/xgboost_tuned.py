"""
XGBoost Tuned Model

Changes from baseline (xgboost_baseline.py):
  - Expanded PARAM_GRID with four new regularization parameters:
      subsample       : fraction of training rows per tree  [0.7, 0.85, 1.0]
      colsample_bytree: fraction of features per tree       [0.7, 0.85, 1.0]
      min_child_weight: min sum of instance weight in leaf  [1, 3, 5]
      reg_alpha       : L1 regularization on leaf weights   [0, 0.1, 1.0]
  - Core params narrowed around the baseline best (n_estimators=500,
    max_depth=6, lr=0.1) so total grid stays manageable (~486 combos).

Output saved to: data/pipeline_output_tuned/
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from itertools import product
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output_tuned"

# Expanded hyperparameter grid
# Core params: narrowed to 2-3 values centred on baseline best
# Regularization params: new additions
PARAM_GRID = {
    "n_estimators":        [300, 500],          # baseline best: 500
    "max_depth":           [4, 5, 6],           # baseline best: 6
    "learning_rate":       [0.05, 0.1],         # baseline best: 0.1
    "subsample":           [0.7, 0.85, 1.0],    # NEW: row subsampling
    "colsample_bytree":    [0.7, 0.85, 1.0],    # NEW: feature subsampling per tree
    "min_child_weight":    [1, 3, 5],           # NEW: leaf regularization
    "reg_alpha":           [0, 0.1, 1.0],       # NEW: L1 regularization
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_model(n_estimators, max_depth, learning_rate,
                subsample=1.0, colsample_bytree=1.0,
                min_child_weight=1, reg_alpha=0):
    """Create XGBoost (or fallback) model with all tunable hyperparameters."""
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            random_state=42,
            verbosity=0,
        )
    # HistGradientBoosting fallback (supports subsample via max_samples,
    # and L2 via l2_regularization; no direct L1/colsample equivalent)
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_leaf_nodes=2 ** max_depth,
        min_samples_leaf=max(1, min_child_weight * 5),
        l2_regularization=reg_alpha,
        random_state=42,
    )


def anomaly_correlation(y_true, y_pred):
    """Anomaly Correlation Coefficient (Pearson r between predictions and actuals)."""
    return np.corrcoef(y_true, y_pred)[0, 1]


# ─── Step 1: Load MIFS features and prepare data ────────────────────────────

def prepare_data(splits, output_dir=OUTPUT_DIR):
    """Load best-K feature list from tuned MIFS output and slice splits."""
    print("[XGB-1] Loading MIFS-selected features (tuned)...")
    with open(Path(output_dir) / "mifs_best_k.json") as f:
        mifs_result = json.load(f)
    best_k = mifs_result["best_k"]
    feature_names = mifs_result["selected_features"]
    print(f"         Using {best_k} features from tuned MIFS")

    X_train = splits["train"][0][feature_names].values
    y_train = splits["train"][1].values
    X_val   = splits["val"][0][feature_names].values
    y_val   = splits["val"][1].values
    X_test  = splits["test"][0][feature_names].values
    y_test  = splits["test"][1].values

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    print(f"         Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, best_k


# ─── Step 2: Grid search (expanded) ─────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val, param_grid=PARAM_GRID):
    """Exhaustive grid search over expanded hyperparameter space."""
    print("\n[XGB-2] Expanded grid search over hyperparameters (tuned)...")
    model_type = "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor"
    print(f"         Model: {model_type}")

    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"         {len(combos)} parameter combinations to evaluate")
    print(f"         New params: subsample, colsample_bytree, min_child_weight, reg_alpha")

    grid_results = []
    best_rmse  = np.inf
    best_params = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        model  = _make_model(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
        acc    = anomaly_correlation(y_val, y_pred)
        grid_results.append({**params, "val_rmse": rmse, "val_acc": acc})

        if rmse < best_rmse:
            best_rmse  = rmse
            best_params = params

        if (i + 1) % 50 == 0:
            print(f"         [{i+1}/{len(combos)}] best so far: "
                  f"RMSE={best_rmse:.4f} {best_params}")

    print(f"\n         Best params: {best_params}")
    print(f"         Best val RMSE: {best_rmse:.4f}")

    grid_results.sort(key=lambda x: x["val_rmse"])
    return best_params, grid_results


# ─── Step 3: Final evaluation on test set ────────────────────────────────────

def final_evaluation(X_train, y_train, X_test, y_test, best_params):
    """Train with best params, evaluate on held-out test set once."""
    print("\n[XGB-3] Final evaluation on test set (2023-2025)...")
    model = _make_model(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_acc  = anomaly_correlation(y_test, y_pred)

    print(f"         Test RMSE: {test_rmse:.4f} (raw Niño 3.4 units)")
    print(f"         Test ACC:  {test_acc:.4f}")

    return test_rmse, test_acc, y_pred, model


# ─── Step 4: Save results ────────────────────────────────────────────────────

def save_results(best_params, best_k, feature_names, grid_results,
                 test_rmse, test_acc, y_test, y_pred, times_test,
                 model=None, output_dir=OUTPUT_DIR):
    """Save predictions, grid search results, and final summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[XGB-4] Saving results...")

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(output_dir / "xgb_grid_search.csv", index=False)
    print(f"         Grid search results: {len(grid_df)} combinations")

    pred_df = pd.DataFrame({
        "window_end":       times_test,
        "actual_nino34":    y_test,
        "predicted_nino34": y_pred,
    })
    pred_df.to_csv(output_dir / "xgb_test_predictions.csv", index=False)
    print(f"         Predictions saved ({len(pred_df)} test samples)")

    # Save trained model
    if model is not None:
        model_path = output_dir / "xgb_tuned_model.joblib"
        joblib.dump(model, model_path)
        print(f"         Trained model saved to xgb_tuned_model.joblib")

    summary = {
        "model": "XGBRegressor (tuned)" if HAS_XGBOOST
                 else "HistGradientBoostingRegressor (tuned)",
        "best_k":       best_k,
        "best_params":  best_params,
        "test_rmse":    round(test_rmse, 4),
        "test_acc":     round(test_acc,  4),
        "selected_features": feature_names,
        "train_samples": 457,
        "val_samples":   48,
        "test_samples":  len(y_test),
        "tuning_notes": (
            "Added subsample, colsample_bytree, min_child_weight, reg_alpha to grid. "
            "MIFS: MI_N_NEIGHBORS=7, N_CANDIDATES=400, MAX_K=80, REDUNDANCY_BETA=1.0."
        ),
    }
    with open(output_dir / "xgb_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"         Final summary saved to xgb_final_summary.json")

    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def run_xgboost_tuned(splits, output_dir=OUTPUT_DIR):
    """Execute the complete tuned XGBoost pipeline."""
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, best_k = \
        prepare_data(splits, output_dir=output_dir)

    best_params, grid_results = grid_search(X_train, y_train, X_val, y_val)

    test_rmse, test_acc, y_pred, model = \
        final_evaluation(X_train, y_train, X_test, y_test, best_params)

    times_test = splits["test"][2]
    summary = save_results(
        best_params, best_k, feature_names, grid_results,
        test_rmse, test_acc, y_test, y_pred, times_test,
        model=model,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("TUNED MODEL SUMMARY")
    print("=" * 60)
    print(f"  Model:          {summary['model']}")
    print(f"  Features (K):   {best_k}")
    print(f"  Best params:    {best_params}")
    print(f"  Test RMSE:      {test_rmse:.4f}")
    print(f"  Test ACC:       {test_acc:.4f}")
    print("=" * 60)

    return summary
