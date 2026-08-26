"""Load the combined climate CSVs into pandas DataFrames.

Datasets
--------
- ERA5             : data/ERA5/era5_combined_1980_2026.csv
- ERA5_PRESSURE    : data/ERA5_pressure_levels/era5_pressure_levels_1980_2025.csv
- ERA5_SOUTHASIA   : data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025.csv
- ORAS5            : data/ORAS5/oras5_combined_1980_2026.csv

Usage
-----
    from load_combined_data import load_all, load_dataset

    frames = load_all()              # dict[str, DataFrame]
    era5 = frames["ERA5"]

    # or load one at a time
    oras5 = load_dataset("ORAS5")
"""
import os

import pandas as pd

# Resolve paths relative to the project root (one level above this scripts/ folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "ERA5": os.path.join(DATA_DIR, "ERA5", "era5_combined_1980_2026.csv"),
    "ERA5_PRESSURE": os.path.join(
        DATA_DIR, "ERA5_pressure_levels", "era5_pressure_levels_1980_2025.csv"
    ),
    "ERA5_SOUTHASIA": os.path.join(
        DATA_DIR,
        "ERA5_single_levels_southasia",
        "era5_single_levels_southasia_1980_2025.csv",
    ),
    "ORAS5": os.path.join(DATA_DIR, "ORAS5", "oras5_combined_1980_2026.csv"),
}


def load_dataset(name):
    """Load one combined CSV into a clean DataFrame.

    Handles padded headers/values (some files have whitespace), strips empty
    strings to NaN, parses the time column and orders the key columns first.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    path = DATASETS[name]

    # skipinitialspace handles the space-padded columns in some files
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from any object columns and turn blanks into NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})

    # Parse time and coerce remaining columns to numeric
    df["time"] = pd.to_datetime(df["time"])
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Key columns first
    lead = [c for c in ("time", "latitude", "longitude") if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest].sort_values(lead).reset_index(drop=True)
    return df


def load_all():
    """Load every dataset, returning a dict of {name: DataFrame}."""
    return {name: load_dataset(name) for name in DATASETS}


# Where the Parquet DataFrames are written (one file per dataset)
PARQUET_PATHS = {name: os.path.splitext(path)[0] + ".parquet"
                 for name, path in DATASETS.items()}


def save_all(frames=None):
    """Save each dataset as a Parquet file next to its source CSV."""
    if frames is None:
        frames = load_all()
    saved = {}
    for name, df in frames.items():
        out = PARQUET_PATHS[name]
        df.to_parquet(out, index=False)
        saved[name] = out
    return saved


def _summary(name, df):
    var_cols = [c for c in df.columns if c not in ("time", "latitude", "longitude")]
    print(f"\n=== {name} ===")
    print(f"  rows       : {len(df):,}")
    print(f"  time       : {df['time'].min().date()} .. {df['time'].max().date()} "
          f"({df['time'].dt.to_period('M').nunique()} months)")
    print(f"  latitude   : {df['latitude'].min()}..{df['latitude'].max()} "
          f"({df['latitude'].nunique()} pts)")
    print(f"  longitude  : {df['longitude'].min()}..{df['longitude'].max()} "
          f"({df['longitude'].nunique()} pts)")
    print(f"  variables  : {var_cols}")
    miss = (df[var_cols].isna().mean() * 100).round(1)
    print("  % missing  : " + ", ".join(f"{c}={v}" for c, v in miss.items()))


if __name__ == "__main__":
    frames = load_all()
    for name, df in frames.items():
        _summary(name, df)
    saved = save_all(frames)
    print("\nSaved Parquet files:")
    for name, path in saved.items():
        print(f"  {name:15s} -> {os.path.relpath(path, BASE_DIR)}")
    print("\nLoaded:", ", ".join(frames))
