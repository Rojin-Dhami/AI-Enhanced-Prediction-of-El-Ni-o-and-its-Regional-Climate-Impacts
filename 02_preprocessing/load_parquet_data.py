"""Load the saved Parquet files back into pandas DataFrames.

Datasets
--------
- ERA5             : data/ERA5/era5_combined_1980_2026.parquet
- ERA5_PRESSURE    : data/ERA5_pressure_levels/era5_pressure_levels_1980_2025.parquet
- ERA5_SOUTHASIA   : data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025.parquet
- ORAS5            : data/ORAS5/oras5_combined_1980_2026.parquet

Usage
-----
    from load_parquet_data import load_parquet, load_all_parquet

    era5 = load_parquet("ERA5")            # one DataFrame
    frames = load_all_parquet()            # dict[str, DataFrame]
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PARQUET_FILES = {
    "ERA5": os.path.join(DATA_DIR, "ERA5", "era5_combined_1980_2026.parquet"),
    "ERA5_PRESSURE": os.path.join(
        DATA_DIR, "ERA5_pressure_levels", "era5_pressure_levels_1980_2025.parquet"
    ),
    "ERA5_SOUTHASIA": os.path.join(
        DATA_DIR,
        "ERA5_single_levels_southasia",
        "era5_single_levels_southasia_1980_2025.parquet",
    ),
    "ORAS5": os.path.join(DATA_DIR, "ORAS5", "oras5_combined_1980_2026.parquet"),
}


def load_parquet(name, columns=None):
    """Read one Parquet file into a DataFrame.

    Pass `columns=[...]` to load only specific columns (faster, less memory).
    """
    if name not in PARQUET_FILES:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(PARQUET_FILES)}")
    return pd.read_parquet(PARQUET_FILES[name], columns=columns)


def load_all_parquet():
    """Read every Parquet file, returning a dict of {name: DataFrame}."""
    return {name: load_parquet(name) for name in PARQUET_FILES}


if __name__ == "__main__":
    frames = load_all_parquet()
    for name, df in frames.items():
        print(f"\n=== {name} ===")
        print(f"  shape  : {df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(f"  dtypes : time={df['time'].dtype}")
        print(df.head(3).to_string(index=False))
    print("\nLoaded:", ", ".join(frames))
