"""
XGBoost Regularized Model — Stratified MIFS + Anti-Overfitting Grid

Trains a regularized XGBoost model on the stratified MIFS-selected features
to predict the Niño 3.4 index at 6-month lead.

Regularization grid (anti-overfitting):
  max_depth        : [2, 3]          — shallow trees only
  n_estimators     : [100, 200, 300]
  learning_rate    : [0.01, 0.05, 0.1]
  subsample        : [0.6, 0.7, 0.8] — row subsampling
  colsample_bytree : [0.6, 0.7, 0.8] — feature subsampling per tree
  min_child_weight : [5, 10, 20]     — large leaf regularization
  reg_alpha        : [0, 0.5, 1.0]   — L1
  reg_lambda       : [1, 5, 10]      — L2

Output saved to: results/method1_stratified_regularized/
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

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "method1_stratified_regularized"

# Anti-overfitting regularized grid
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_model(max_depth, n_estimators, learning_rate,
                subsample=1.0, colsample_bytree=1.0,
                min_child_weight=1, reg_alpha=0, reg_lambda=1):
    """Create XGBoost (or fallback) model with full regularization params."""
    if HAS_XGBOOST:
        return XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0,
        )
    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=max(1, min_child_weight * 5),
        l2_regularization=reg_lambda,
        random_state=42,
    )


def anomaly_correlation(y_true, y_pred):
    """Anomaly Correlation Coefficient — Pearson r between predictions and actuals."""
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# ─── Step 1: Load MIFS features and prepare data ────────────────────────────

def prepare_data(splits, output_dir=OUTPUT_DIR):
    """Load best-K feature list from stratified MIFS output and slice splits."""
    print("[XGB-1] Loading stratified MIFS-selected features...")
    output_dir = Path(output_dir)
    with open(output_dir / "mifs_best_k.json") as f:
        mifs_result = json.load(f)
    best_k        = mifs_result["best_k"]
    feature_names = mifs_result["selected_features"]
    print(f"         Using {best_k} stratified MIFS features")

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


# ─── Step 2: Grid search ────────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val, param_grid=PARAM_GRID):
    """Exhaustive anti-overfitting grid search evaluated on validation set.

    Tracks train RMSE alongside val RMSE to monitor overfitting at each combo.
    """
    print("\n[XGB-2] Anti-overfitting grid search over hyperparameters...")
    model_type = "XGBRegressor" if HAS_XGBOOST else "HistGradientBoostingRegressor"
    print(f"         Model: {model_type}")

    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"         {len(combos)} combinations | "
          f"focus: shallow trees + strong regularization")

    grid_results  = []
    best_val_rmse = np.inf
    best_params   = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        model  = _make_model(**params)
        model.fit(X_train, y_train)

        yp_val   = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, yp_val))
        val_acc  = anomaly_correlation(y_val, yp_val)

        yp_tr    = model.predict(X_train)
        tr_rmse  = np.sqrt(mean_squared_error(y_train, yp_tr))

        grid_results.append({
            **params,
            "train_rmse": round(tr_rmse, 5),
            "val_rmse":   round(val_rmse, 5),
            "val_acc":    round(val_acc,  5),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params   = params

        if (i + 1) % 200 == 0:
            print(f"         [{i+1}/{len(combos)}] best val RMSE={best_val_rmse:.4f}  "
                  f"{best_params}")

    print(f"\n         Best params:  {best_params}")
    print(f"         Best val RMSE: {best_val_rmse:.4f}")

    grid_results.sort(key=lambda x: x["val_rmse"])
    return best_params, grid_results


# ─── Step 3: Final evaluation on test set ────────────────────────────────────

def final_evaluation(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    """Train with best params on train set, evaluate on val and test sets."""
    print("\n[XGB-3] Final evaluation...")
    model = _make_model(**best_params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse   = np.sqrt(mean_squared_error(y_val,   y_pred_val))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    train_acc  = anomaly_correlation(y_train, y_pred_train)
    val_acc    = anomaly_correlation(y_val,   y_pred_val)
    test_acc   = anomaly_correlation(y_test,  y_pred_test)

    print(f"\n         {'Split':<8s} {'RMSE':>10s} {'ACC':>10s}")
    print(f"         {'-'*8} {'-'*10} {'-'*10}")
    print(f"         {'Train':<8s} {train_rmse:>10.4f} {train_acc:>10.4f}")
    print(f"         {'Val':<8s} {val_rmse:>10.4f} {val_acc:>10.4f}")
    print(f"         {'Test':<8s} {test_rmse:>10.4f} {test_acc:>10.4f}")
    print(f"\n         Train→Test RMSE gap: {test_rmse - train_rmse:+.4f}")

    return (train_rmse, val_rmse, test_rmse,
            train_acc,  val_acc,  test_acc,
            y_pred_test, model)


# ─── Step 4: Save everything ────────────────────────────────────────────────

def save_results(best_params, best_k, feature_names, grid_results,
                 train_rmse, val_rmse, test_rmse,
                 train_acc,  val_acc,  test_acc,
                 y_test, y_pred_test, times_test,
                 model=None, output_dir=OUTPUT_DIR):
    """Save hyperparameters, metrics, predictions, grid search, and model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[XGB-4] Saving results...")

    # Grid search results
    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(output_dir / "xgb_grid_search.csv", index=False)
    print(f"         Grid search results: {len(grid_df)} combinations")

    # Predictions vs actual
    pred_df = pd.DataFrame({
        "window_end":       times_test,
        "actual_nino34":    y_test,
        "predicted_nino34": y_pred_test,
        "residual":         y_test - y_pred_test,
    })
    pred_df.to_csv(output_dir / "xgb_test_predictions.csv", index=False)
    print(f"         Predictions saved ({len(pred_df)} test samples)")

    # Trained model
    if model is not None:
        joblib.dump(model, output_dir / "xgb_stratified_regularized_model.joblib")
        print(f"         Model saved → xgb_stratified_regularized_model.joblib")

    # Final summary
    summary = {
        "model": ("XGBRegressor (stratified+regularized)"
                  if HAS_XGBOOST else
                  "HistGradientBoostingRegressor (stratified+regularized)"),
        "best_k":       best_k,
        "best_params":  best_params,
        "train_rmse":   round(train_rmse, 4),
        "train_acc":    round(train_acc,  4),
        "val_rmse":     round(val_rmse,   4),
        "val_acc":      round(val_acc,    4),
        "test_rmse":    round(test_rmse,  4),
        "test_acc":     round(test_acc,   4),
        "selected_features": feature_names,
        "train_samples": 457,
        "val_samples":   48,
        "test_samples":  len(y_test),
        "notes": (
            "Stratified quota-based MIFS prefilter (per-variable-type quotas) "
            "combined with anti-overfitting regularized XGBoost grid: "
            "max_depth<=3, min_child_weight up to 20, reg_lambda added, "
            "subsample/colsample restricted."
        ),
    }
    with open(output_dir / "xgb_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"         Final summary saved → xgb_final_summary.json")

    return summary


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from mifs_selection import run_mifs

    splits, _ = run_pipeline(save=True)
    final_features, best_k = run_mifs(splits)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, k = \
        prepare_data(splits, output_dir=OUTPUT_DIR)

    best_params, grid_results = grid_search(X_train, y_train, X_val, y_val)

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
        model=model, output_dir=OUTPUT_DIR,
    )
