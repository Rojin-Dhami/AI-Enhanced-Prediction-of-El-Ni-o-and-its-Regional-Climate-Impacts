"""Display null SST values that are NOT land for the year 1980.

Reads the landcheck CSV and filters for rows where SST is missing
but the location is classified as 'ocean' (not land). Prints the
latitude, longitude, and time for each such null.
"""

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                   "era5_sst_pacific_2x2_1980_landcheck.csv")


def main():
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()

    # Strip whitespace from string columns and convert sst to numeric
    df["sst"] = pd.to_numeric(df["sst"].astype(str).str.strip(), errors="coerce")
    df["land_ocean"] = df["land_ocean"].astype(str).str.strip()

    # Filter: SST is null AND land_ocean is 'ocean' (not land)
    ocean_nulls = df[(df["sst"].isna()) & (df["land_ocean"] == "ocean")].copy()

    print(f"Total rows in 1980 data: {len(df)}")
    print(f"Total missing SST: {df['sst'].isna().sum()}")
    print(f"Missing SST on land: {(df['land_ocean'] == 'land').sum()}")
    print(f"Missing SST on ocean (non-land nulls): {len(ocean_nulls)}")
    print()

    if ocean_nulls.empty:
        print("No ocean null values found.")
        return

    print(f"{'Time':<12} {'Latitude':>10} {'Longitude':>10}")
    print("-" * 36)
    for _, row in ocean_nulls.iterrows():
        print(f"{row['time']:<12} {row['latitude']:>10.1f} {row['longitude']:>10.1f}")


if __name__ == "__main__":
    main()
