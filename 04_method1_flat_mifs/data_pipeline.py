"""
Data Pipeline Module — Section 3.4.3 (Foundation for MIFS & XGBoost)

Reshapes the combined ERA5+ORAS5 long-format CSV into the tabular input
required by MIFS feature selection (Sec 3.4.4) and XGBoost baseline (Sec 3.4.5).

Steps:
  1. Load data, keep only _anom_z columns + time/lat/lon
  2. Pivot long → wide: one row per month, columns = (variable, lat, lon)
  3. Drop columns that are entirely NaN (land cells for ocean vars, etc.)
  4. Build 12-month rolling windows (T=12)
  5. Align target: Niño 3.4 at t+6 (window end + 6 months)
  6. Chronological split: train 1980-2018, val 2019-2022, test 2023-2025

Conventions:
  - Lead time reference: window END (most recent month t), target = t + 6
  - All 9 _anom_z variables used (C=9)
  - Target: raw Niño 3.4 index (not z-scored)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "combined_era5_oras5.csv"
NINO34_CSV = DATA_DIR / "ERA5" / "nino34_index_monthly.csv"
OUTPUT_DIR = DATA_DIR / "pipeline_output"

ANOM_Z_COLS = [
    "avg_iews_anom_z", "avg_inss_anom_z", "ttr_anom_z",
    "sst_anom_z", "t2m_anom_z", "msl_anom_z",
    "so20chgt_anom_z", "sohtc300_anom_z", "sossheig_anom_z",
]

WINDOW_SIZE = 12  # months
LEAD_TIME = 6     # months ahead of window end

TRAIN_END = "2018-12-01"
VAL_END = "2022-12-01"
# test: 2023-01 onward (up to available data minus lead time)


# ─── Step 1-2: Load and pivot long → wide ───────────────────────────────────

def load_and_pivot(csv_path=INPUT_CSV):
    """Load long-format CSV, keep _anom_z columns, pivot to wide format.
    
    Returns:
        wide_df: DataFrame with one row per month, columns = 'var_(lat,lon)'
                 Index is sorted datetime.
    """
    print("[1] Loading data...")
    usecols = ["time", "latitude", "longitude"] + ANOM_Z_COLS
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["time"])
    print(f"    Loaded: {len(df):,} rows, {df['time'].nunique()} months, "
          f"{df['latitude'].nunique()} lats × {df['longitude'].nunique()} lons")

    print("[2] Pivoting long → wide format...")
    # Create a location identifier for each (lat, lon) pair
    df["loc"] = df["latitude"].astype(str) + "," + df["longitude"].astype(str)

    # Melt the _anom_z columns into variable/value, then combine var+loc as column name
    melted = df.melt(
        id_vars=["time", "loc"],
        value_vars=ANOM_Z_COLS,
        var_name="variable",
        value_name="value",
    )
    melted["col_name"] = melted["variable"] + "_(" + melted["loc"] + ")"

    # Pivot to wide: one row per time, one column per (variable, lat, lon)
    wide_df = melted.pivot_table(
        index="time", columns="col_name", values="value", aggfunc="first"
    )
    wide_df.sort_index(inplace=True)
    wide_df.columns.name = None

    print(f"    Wide shape: {wide_df.shape[0]} months × {wide_df.shape[1]} features")
    return wide_df


# ─── Step 3: Drop all-NaN columns, flag sporadic NaNs ───────────────────────

def drop_nan_columns(wide_df):
    """Drop columns that are entirely NaN. Flag sporadic NaN columns.
    
    Returns:
        cleaned_df: DataFrame with all-NaN columns removed.
    """
    print("[3] Dropping all-NaN columns...")
    n_before = wide_df.shape[1]
    all_nan_mask = wide_df.isna().all(axis=0)
    n_all_nan = all_nan_mask.sum()
    cleaned_df = wide_df.loc[:, ~all_nan_mask]
    print(f"    Dropped {n_all_nan} all-NaN columns "
          f"({n_before} → {cleaned_df.shape[1]})")

    # Check for sporadic NaNs in remaining columns
    sporadic_nan = cleaned_df.isna().any(axis=0) & ~cleaned_df.isna().all(axis=0)
    n_sporadic = sporadic_nan.sum()
    if n_sporadic > 0:
        # Report summary by variable type
        sporadic_cols = cleaned_df.columns[sporadic_nan]
        var_counts = pd.Series([c.split("_(")[0] for c in sporadic_cols]).value_counts()
        print(f"    ⚠ {n_sporadic} columns have sporadic NaNs:")
        for var, count in var_counts.items():
            # What fraction of time steps are NaN on average?
            subset = cleaned_df[sporadic_cols[
                [c.startswith(var + "_(") for c in sporadic_cols]
            ]]
            avg_pct = subset.isna().mean().mean() * 100
            print(f"      {var}: {count} columns, avg {avg_pct:.1f}% missing")
    else:
        print("    No sporadic NaNs found — data is clean.")

    return cleaned_df


# ─── Step 4: Build 12-month rolling windows ─────────────────────────────────

def build_rolling_windows(cleaned_df, window_size=WINDOW_SIZE):
    """Stack T consecutive months into one row per sample.
    
    For each window ending at month t (index position i), creates columns:
      feature_lag0  (month t, most recent)
      feature_lag1  (month t-1)
      ...
      feature_lag11 (month t-11, oldest)
    
    Returns:
        windowed_df: DataFrame with (N_months - window_size + 1) rows,
                     each with (N_features × window_size) columns.
        window_end_times: DatetimeIndex of the window end month for each row.
    """
    print(f"[4] Building {window_size}-month rolling windows...")
    n_months, n_features = cleaned_df.shape
    n_samples = n_months - window_size + 1
    feature_names = cleaned_df.columns.tolist()
    values = cleaned_df.values  # (n_months, n_features)

    # Build column names: feature_lagK for each feature and each lag
    col_names = []
    for lag in range(window_size):
        for feat in feature_names:
            col_names.append(f"{feat}_lag{lag}")

    # Construct the windowed array
    # Each row i corresponds to window ending at time index (i + window_size - 1)
    windowed = np.empty((n_samples, n_features * window_size), dtype=np.float32)
    for i in range(n_samples):
        end_idx = i + window_size - 1
        for lag in range(window_size):
            row_idx = end_idx - lag  # lag0 = most recent, lag11 = oldest
            windowed[i, lag * n_features: (lag + 1) * n_features] = values[row_idx]

    window_end_times = cleaned_df.index[window_size - 1:]
    windowed_df = pd.DataFrame(windowed, columns=col_names, index=window_end_times)

    print(f"    Windowed shape: {windowed_df.shape[0]} samples × "
          f"{windowed_df.shape[1]:,} features")
    return windowed_df


# ─── Step 5: Align target (Niño 3.4 at t+6) ─────────────────────────────────

def align_target(windowed_df, nino34_path=NINO34_CSV, lead_time=LEAD_TIME):
    """Align Niño 3.4 target at window_end + lead_time months.
    
    Returns:
        X: feature DataFrame (rows with valid targets only)
        y: target Series (raw Niño 3.4)
        times: DatetimeIndex of window end months (for split reference)
    """
    print(f"[5] Aligning target: Niño 3.4 at window_end + {lead_time} months...")
    nino34 = pd.read_csv(nino34_path, parse_dates=["time"])
    nino34 = nino34.set_index("time").sort_index()

    # For each window ending at month t, target = nino34 at t + lead_time months
    target_times = windowed_df.index + pd.DateOffset(months=lead_time)

    # Map target times to nino34 values
    y_values = nino34.reindex(target_times)["nino34"].values
    valid_mask = ~np.isnan(y_values)

    X = windowed_df.loc[valid_mask].copy()
    y = pd.Series(y_values[valid_mask], index=X.index, name="nino34_target")
    times = X.index

    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"    Dropped {n_dropped} samples with no target available "
              f"(target month beyond data range)")
    print(f"    Final: {len(X)} samples with valid targets")
    print(f"    Target range: {y.min():.3f} to {y.max():.3f}")
    return X, y, times


# ─── Step 6: Chronological train/val/test split ─────────────────────────────

def chronological_split(X, y, times, train_end=TRAIN_END, val_end=VAL_END):
    """Split by window end time. No date overlap allowed.
    
    Train: window end ≤ 2018-12 (and target ≤ 2019-06, within train+buffer)
    Val:   window end in 2019-01 to 2022-12
    Test:  window end ≥ 2023-01
    
    Returns dict with keys 'train', 'val', 'test', each containing (X, y, times).
    """
    print("[6] Chronological split...")
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_mask = times <= train_end_dt
    val_mask = (times > train_end_dt) & (times <= val_end_dt)
    test_mask = times > val_end_dt

    splits = {
        "train": (X.loc[train_mask], y.loc[train_mask], times[train_mask]),
        "val": (X.loc[val_mask], y.loc[val_mask], times[val_mask]),
        "test": (X.loc[test_mask], y.loc[test_mask], times[test_mask]),
    }

    for name, (X_s, y_s, t_s) in splits.items():
        print(f"    {name:5s}: {len(X_s):4d} samples | "
              f"window end: {t_s.min().strftime('%Y-%m')} to {t_s.max().strftime('%Y-%m')}")

    # Verify no overlap: last train window end + lead must not have target in val period
    last_train_target = times[train_mask].max() + pd.DateOffset(months=LEAD_TIME)
    first_val_window = times[val_mask].min()
    print(f"    Last train target month: {last_train_target.strftime('%Y-%m')}")
    print(f"    First val window end:    {first_val_window.strftime('%Y-%m')}")
    assert last_train_target <= first_val_window + pd.DateOffset(months=LEAD_TIME), \
        "Data leakage: train target overlaps with validation input!"

    return splits


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline(save=True):
    """Execute full Step 1 pipeline and optionally save outputs."""
    wide_df = load_and_pivot()
    cleaned_df = drop_nan_columns(wide_df)
    windowed_df = build_rolling_windows(cleaned_df)
    X, y, times = align_target(windowed_df)
    splits = chronological_split(X, y, times)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Save split metadata
        meta = {
            "n_features_wide": cleaned_df.shape[1],
            "n_features_windowed": X.shape[1],
            "window_size": WINDOW_SIZE,
            "lead_time": LEAD_TIME,
            "variables_used": ANOM_Z_COLS,
            "train_samples": len(splits["train"][0]),
            "val_samples": len(splits["val"][0]),
            "test_samples": len(splits["test"][0]),
        }
        import json
        with open(OUTPUT_DIR / "pipeline_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n    Metadata saved to {OUTPUT_DIR / 'pipeline_meta.json'}")

        # Save feature names (before windowing — base features)
        pd.Series(cleaned_df.columns, name="feature").to_csv(
            OUTPUT_DIR / "base_feature_names.csv", index=False
        )
        print(f"    Base feature names saved ({cleaned_df.shape[1]} features)")

    print("\n[✓] Pipeline complete.")
    return splits, cleaned_df


if __name__ == "__main__":
    splits, cleaned_df = run_pipeline(save=True)
