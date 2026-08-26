"""Convert the ERA5 landcheck CSV to a Parquet file.

Input : data/ERA5/era5_combined_1980_2026_landcheck.csv
Output: data/ERA5/era5_combined_1980_2026_landcheck.parquet
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "ERA5", "era5_combined_1980_2026_landcheck.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "ERA5", "era5_combined_1980_2026_landcheck.parquet")


def convert(csv_path: str = CSV_PATH, parquet_path: str = PARQUET_PATH) -> None:
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    print(f"Writing {parquet_path} ...")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    print("Done.")


if __name__ == "__main__":
    convert()
