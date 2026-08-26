"""
XGBoost Baseline Model — Section 3.4.5

Trains an XGBoost regression model (or HistGradientBoostingRegressor fallback)
on the MIFS-selected features to predict the Niño 3.4 index at 6-month lead.

Steps:
  1. Load MIFS-selected features (best K from Step 2)
  2. Grid search hyperparameters on validation set (2019-2022)
  3. Final evaluation on test set (2023-2025): RMSE and ACC
  4. Save results

Note: Target is raw Niño 3.4 (per user confirmation), so predictions are
already in physical units — no de-normalization needed.
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
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output"

# Hyperparameter grid
PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_model(n_estimators, max_depth, learning_rate):
    """Create model with given hyperparameters."""
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42, verbosity=0,
        )
    return HistGradientBoostingRegressor(
        max_iter=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42,
    )


def anomaly_correlation(y_true, y_pred):
    """Anomaly Correlation Coefficient — Pearson correlation between
    predicted and actual Niño 3.4 index."""
    return np.corrcoef(y_true, y_pred)[0, 1]


# ─── Step 1: Load MIFS features and prepare data ────────────────────────────

def prepare_data(splits, output_dir=OUTPUT_DIR):
    """Load best-K feature list from MIFS and slice splits accordingly.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: arrays
        feature_names: list of selected feature names
        best_k: int
    """
    print("[XGB-1] Loading MIFS-selected features...")
    with open(output_dir / "mifs_best_k.json") as f:
        mifs_result = json.load(f)
    best_k = mifs_result["best_k"]
    feature_names = mifs_result["selected_features"]
    print(f"         Using {best_k} features from MIFS")

    X_train = splits["train"][0][feature_names].values
    y_train = splits["train"][1].values
    X_val = splits["val"][0][feature_names].values
    y_val = splits["val"][1].values
    X_test = splits["test"][0][feature_names].values
    y_test = splits["test"][1].values

    # Replace NaN if any
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"         Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, best_k


# ─── Step 2: Grid search ────────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val, param_grid=PARAM_GRID):
    """Exhaustive grid search over hyperparameters, evaluated on validation set.
    
    Returns:
        best_params: dict with best hyperparameters
        grid_results: list of (params, val_rmse) sorted by RMSE
    """
    print("\n[XGB-2] Grid search over hyperparameters...")
    model_type = "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor"
    print(f"         Model: {model_type}")

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"         {len(combos)} parameter combinations to evaluate")

    grid_results = []
    best_rmse = np.inf
    best_params = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        model = _make_model(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        acc = anomaly_correlation(y_val, y_pred)
        grid_results.append({**params, "val_rmse": rmse, "val_acc": acc})

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

        if (i + 1) % 16 == 0:
            print(f"         [{i+1}/{len(combos)}] best so far: "
                  f"RMSE={best_rmse:.4f} {best_params}")

    print(f"\n         Best params: {best_params}")
    print(f"         Best val RMSE: {best_rmse:.4f}")

    grid_results.sort(key=lambda x: x["val_rmse"])
    return best_params, grid_results


# ─── Step 3: Final evaluation on test set ────────────────────────────────────

def final_evaluation(X_train, y_train, X_test, y_test, best_params):
    """Train with best hyperparameters, evaluate on held-out test set ONCE.
    
    Returns:
        train_rmse, train_acc, test_rmse, test_acc, y_pred, model
    """
    print("\n[XGB-3] Final evaluation on test set (2023-2025)...")
    model = _make_model(**best_params)
    model.fit(X_train, y_train)
    
    # Training set evaluation
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_acc = anomaly_correlation(y_train, y_train_pred)
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_acc = anomaly_correlation(y_test, y_pred)

    print(f"         Train RMSE: {train_rmse:.4f} (raw Niño 3.4 units)")
    print(f"         Train ACC:  {train_acc:.4f}")
    print(f"         Test RMSE: {test_rmse:.4f} (raw Niño 3.4 units)")
    print(f"         Test ACC:  {test_acc:.4f}")

    return train_rmse, train_acc, test_rmse, test_acc, y_pred, model


# ─── Step 4: Save everything ────────────────────────────────────────────────

def save_results(best_params, best_k, feature_names, grid_results,
                 train_rmse, train_acc, test_rmse, test_acc, y_test, y_pred, times_test,
                 model, output_dir=OUTPUT_DIR):
    """Save hyperparameters, metrics, predictions, grid search results,
    the trained model, and a full history dict for later exploration."""
    output_dir = Path(output_dir)
    print("\n[XGB-4] Saving results...")

    # Grid search results
    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(output_dir / "xgb_grid_search.csv", index=False)
    print(f"         Grid search results: {len(grid_df)} combinations")

    # Predictions vs actual
    pred_df = pd.DataFrame({
        "window_end": times_test,
        "actual_nino34": y_test,
        "predicted_nino34": y_pred,
        "residual": y_test - y_pred,
    })
    pred_df.to_csv(output_dir / "xgb_test_predictions.csv", index=False)

    # Trained model — joblib is recommended for sklearn-compatible objects
    model_path = output_dir / "xgb_model.joblib"
    joblib.dump(model, model_path)
    print(f"         Model saved  → xgb_model.joblib")

    # Final summary
    summary = {
        "model": "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor",
        "best_k": best_k,
        "best_params": best_params,
        "train_rmse": round(train_rmse, 4),
        "train_acc": round(train_acc, 4),
        "test_rmse": round(test_rmse, 4),
        "test_acc": round(test_acc, 4),
        "selected_features": feature_names,
        "train_samples": 457,
        "val_samples": 48,
        "test_samples": len(y_test),
    }
    with open(output_dir / "xgb_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"         Final summary saved → xgb_final_summary.json")

    # Full history — everything needed to reproduce plots / diagnostics
    history = {
        "summary": summary,
        "grid_search": grid_results,           # list of dicts: params + val_rmse/acc
        "test_predictions": {
            "window_end": [str(t) for t in times_test],
            "actual_nino34": y_test.tolist(),
            "predicted_nino34": y_pred.tolist(),
            "residual": (y_test - y_pred).tolist(),
        },
    }
    with open(output_dir / "xgb_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"         Full history saved  → xgb_history.json")

    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def run_xgboost_baseline(splits):
    """Execute the complete XGBoost baseline pipeline (Step 3 from .md).
    
    Args:
        splits: dict from data_pipeline.run_pipeline()
    
    Returns:
        summary: dict with all results
    """
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, best_k = \
        prepare_data(splits)

    best_params, grid_results = grid_search(X_train, y_train, X_val, y_val)

    train_rmse, train_acc, test_rmse, test_acc, y_pred, model = \
        final_evaluation(X_train, y_train, X_test, y_test, best_params)

    times_test = splits["test"][2]
    summary = save_results(
        best_params, best_k, feature_names, grid_results,
        train_rmse, train_acc, test_rmse, test_acc, y_test, y_pred, times_test,
        model=model,
    )

    # Final printed summary
    print("\n" + "=" * 60)
    print("BASELINE MODEL SUMMARY (Section 3.4.5)")
    print("=" * 60)
    print(f"  Model:          {summary['model']}")
    print(f"  Features (K):   {best_k}")
    print(f"  Best params:    {best_params}")
    print(f"  Train RMSE:     {train_rmse:.4f}")
    print(f"  Train ACC:      {train_acc:.4f}")
    print(f"  Test RMSE:      {test_rmse:.4f}")
    print(f"  Test ACC:       {test_acc:.4f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from mifs_selection import run_mifs

    print("=" * 70)
    print("STEP 1: Data Pipeline")
    print("=" * 70)
    splits, _ = run_pipeline(save=True)

    print("\n" + "=" * 70)
    print("STEP 2: MIFS Feature Selection")
    print("=" * 70)
    final_features, best_k = run_mifs(splits)

    print("\n" + "=" * 70)
    print("STEP 3: XGBoost Baseline Model")
    print("=" * 70)
    summary = run_xgboost_baseline(splits)
