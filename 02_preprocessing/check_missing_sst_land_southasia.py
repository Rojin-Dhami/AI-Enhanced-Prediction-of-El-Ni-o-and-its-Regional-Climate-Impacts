"""Classify missing SST cells as land or ocean for the South Asia single-levels data.

Same approach as scripts/check_missing_sst_land_combined.py, but for
data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025.csv
using the South Asia ERA5 land-sea mask. A cell is land when its land
fraction ``lsm > 0.5``.

Adds two columns:
    - lsm        : ERA5 land fraction (0..1) at each grid point
    - land_ocean : "land"/"ocean" for rows with missing SST, else ""

Input : data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025.csv
Output: data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025_landcheck.csv
Mask  : data/ERA5_land_sea_mask_southasia/era5_land_sea_mask_southasia_extracted/*.nc
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(BASE, "data", "ERA5_single_levels_southasia",
                      "era5_single_levels_southasia_1980_2025.csv")
OUT_CSV = os.path.join(BASE, "data", "ERA5_single_levels_southasia",
                       "era5_single_levels_southasia_1980_2025_landcheck.csv")
LSM_GLOB = os.path.join(BASE, "data", "ERA5_land_sea_mask_southasia",
                        "era5_land_sea_mask_southasia_extracted", "*.nc")
LAND_THRESHOLD = 0.5


def main():
    nc_files = sorted(glob.glob(LSM_GLOB))
    if not nc_files:
        raise FileNotFoundError(f"No land-sea mask files found for {LSM_GLOB}")

    ds = xr.open_dataset(nc_files[0])
    lsm = ds["lsm"].isel(valid_time=0)
    lsm = lsm.assign_coords(longitude=(lsm["longitude"] % 360.0)).sortby(
        "longitude")

    # header is whitespace-padded; strip on read
    df = pd.read_csv(IN_CSV, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    missing_mask = df["sst"].isna().values
    lats = df["latitude"].values
    lons = df["longitude"].values % 360.0

    # Look up land fraction (lsm) once per unique (lat, lon) for ALL rows.
    all_coords = np.stack([lats, lons], axis=1)
    uniq_all, inverse_all = np.unique(all_coords, axis=0, return_inverse=True)
    uniq_lsm = np.empty(len(uniq_all), dtype=float)
    for k, (la, lo) in enumerate(uniq_all):
        uniq_lsm[k] = lsm.sel(latitude=la, longitude=lo, method="nearest").item()
    lsm_values = uniq_lsm[inverse_all]

    land_ocean = np.empty(len(df), dtype=object)
    land_ocean[:] = ""
    land_ocean[missing_mask] = np.where(
        lsm_values[missing_mask] > LAND_THRESHOLD, "land", "ocean")

    df["lsm"] = lsm_values
    df["land_ocean"] = land_ocean
    df.to_csv(OUT_CSV, index=False)

    miss_idx = np.where(missing_mask)[0]
    uniq_coords = np.unique(all_coords[miss_idx], axis=0)
    uniq_class = np.where(
        np.array([lsm.sel(latitude=la, longitude=lo, method="nearest").item()
                  for la, lo in uniq_coords]) > LAND_THRESHOLD,
        "land", "ocean")

    n_missing = int(missing_mask.sum())
    n_land = int((land_ocean == "land").sum())
    n_ocean = int((land_ocean == "ocean").sum())
    print(f"Mask file               : {os.path.basename(nc_files[0])}")
    print(f"Total rows              : {len(df)}")
    print(f"Missing SST rows        : {n_missing}")
    print(f"  -> land  (lsm > {LAND_THRESHOLD}) : {n_land}")
    print(f"  -> ocean (lsm <= {LAND_THRESHOLD}): {n_ocean}")
    print(f"Unique missing coords   : {len(uniq_coords)}")
    print(f"  -> land coords        : {int((uniq_class == 'land').sum())}")
    print(f"  -> ocean coords       : {int((uniq_class == 'ocean').sum())}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
