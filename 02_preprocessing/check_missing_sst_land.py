"""Classify missing ERA5 2x2 SST cells as land or ocean using ORAS5.

For every row in the ERA5 2x2 Pacific SST CSV whose ``sst`` value is
missing, find the nearest ORAS5 ocean grid cell. ORAS5 stores NaN over
land, so if the nearest ORAS5 cell is NaN the ERA5 location is treated as
land. A new ``land_ocean`` column is written:

    - "land"  : sst is missing AND nearest ORAS5 cell is land (NaN)
    - "ocean" : sst is missing but nearest ORAS5 cell has a value
    - ""      : sst is present (not missing)

Input : data/ERA5_sst_pacific_2x2/era5_sst_pacific_2x2_1980.csv
Output: data/ERA5_sst_pacific_2x2/era5_sst_pacific_2x2_1980_landcheck.csv
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                      "era5_sst_pacific_2x2_1980.csv")
OUT_CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                       "era5_sst_pacific_2x2_1980_landcheck.csv")
ORAS5_GLOB = os.path.join(BASE, "data", "ORAS5_sst",
                          "oras5_sst_1980.zip_extracted", "*.nc")


def main():
    nc_files = sorted(glob.glob(ORAS5_GLOB))
    if not nc_files:
        raise FileNotFoundError(f"No ORAS5 files found for {ORAS5_GLOB}")

    ds = xr.open_dataset(nc_files[0])
    var = list(ds.data_vars)[0]

    oras_lat = ds["nav_lat"].values
    oras_lon = ds["nav_lon"].values % 360.0
    oras_sst = ds[var].values[0]  # (y, x) first time step

    df = pd.read_csv(IN_CSV)

    land_ocean = np.empty(len(df), dtype=object)
    land_ocean[:] = ""

    missing_mask = df["sst"].isna().values
    lats = df["latitude"].values
    lons = df["longitude"].values % 360.0

    n_land = 0
    n_ocean = 0
    for idx in np.where(missing_mask)[0]:
        d = (oras_lat - lats[idx]) ** 2 + (oras_lon - lons[idx]) ** 2
        j, i = np.unravel_index(np.nanargmin(d), d.shape)
        if np.isnan(oras_sst[j, i]):
            land_ocean[idx] = "land"
            n_land += 1
        else:
            land_ocean[idx] = "ocean"
            n_ocean += 1

    df["land_ocean"] = land_ocean
    df.to_csv(OUT_CSV, index=False)

    n_missing = int(missing_mask.sum())
    print(f"Total rows           : {len(df)}")
    print(f"Missing SST rows     : {n_missing}")
    print(f"  -> land  (ORAS5 NaN): {n_land}")
    print(f"  -> ocean (ORAS5 val): {n_ocean}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
