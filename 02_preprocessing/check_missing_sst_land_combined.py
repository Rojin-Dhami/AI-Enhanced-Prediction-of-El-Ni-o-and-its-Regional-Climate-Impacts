"""Classify missing ERA5 SST cells as land or ocean using the ERA5 land-sea mask.

Replaces the earlier ORAS5-SST-NaN proxy with ERA5's own static
``lsm`` (land-sea mask) field, so the land/ocean classification is
self-consistent with the ERA5 data itself. A cell is land when its
land fraction ``lsm > 0.5``.

Because the mask is time-invariant, classification is done once per
unique (latitude, longitude) pair and then mapped back to all rows.
A new ``land_ocean`` column is written:

    - "land"  : sst is missing AND lsm > 0.5
    - "ocean" : sst is missing AND lsm <= 0.5
    - ""      : sst is present (not missing)

Input : data/ERA5/era5_combined_1980_2026.csv
Output: data/ERA5/era5_combined_1980_2026_landcheck.csv
Mask  : data/ERA5_land_sea_mask/era5_land_sea_mask_extracted/*.nc
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(BASE, "data", "ERA5", "era5_combined_1980_2026.csv")
OUT_CSV = os.path.join(BASE, "data", "ERA5",
                       "era5_combined_1980_2026_landcheck.csv")
LSM_GLOB = os.path.join(BASE, "data", "ERA5_land_sea_mask",
                        "era5_land_sea_mask_extracted", "*.nc")
LAND_THRESHOLD = 0.5


def main():
    nc_files = sorted(glob.glob(LSM_GLOB))
    if not nc_files:
        raise FileNotFoundError(f"No land-sea mask files found for {LSM_GLOB}")

    ds = xr.open_dataset(nc_files[0])
    # mask is static in time: take the first time step
    lsm = ds["lsm"].isel(valid_time=0)
    # normalise longitude of the mask to 0..360 to match the CSV
    lsm = lsm.assign_coords(longitude=(lsm["longitude"] % 360.0)).sortby(
        "longitude")

    df = pd.read_csv(IN_CSV)

    missing_mask = df["sst"].isna().values
    lats = df["latitude"].values
    lons = df["longitude"].values % 360.0

    # Look up the land fraction (lsm) once per unique (lat, lon) for ALL rows.
    all_coords = np.stack([lats, lons], axis=1)
    uniq_all, inverse_all = np.unique(all_coords, axis=0, return_inverse=True)
    uniq_lsm = np.empty(len(uniq_all), dtype=float)
    for k, (la, lo) in enumerate(uniq_all):
        uniq_lsm[k] = lsm.sel(latitude=la, longitude=lo, method="nearest").item()
    lsm_values = uniq_lsm[inverse_all]

    # land/ocean label only for rows with missing SST.
    land_ocean = np.empty(len(df), dtype=object)
    land_ocean[:] = ""
    land_ocean[missing_mask] = np.where(
        lsm_values[missing_mask] > LAND_THRESHOLD, "land", "ocean")

    df["lsm"] = lsm_values
    df["land_ocean"] = land_ocean
    df.to_csv(OUT_CSV, index=False)

    # summary counts over missing rows / unique missing coords
    miss_idx = np.where(missing_mask)[0]
    uniq_coords = np.unique(all_coords[miss_idx], axis=0)
    uniq_class = np.where(
        np.array([lsm.sel(latitude=la, longitude=lo, method="nearest").item()
                  for la, lo in uniq_coords]) > LAND_THRESHOLD,
        "land", "ocean")

    n_missing = int(missing_mask.sum())
    n_land = int((land_ocean == "land").sum())
    n_ocean = int((land_ocean == "ocean").sum())
    n_uniq_land = int((uniq_class == "land").sum())
    n_uniq_ocean = int((uniq_class == "ocean").sum())
    print(f"Mask file               : {os.path.basename(nc_files[0])}")
    print(f"Total rows              : {len(df)}")
    print(f"Missing SST rows        : {n_missing}")
    print(f"  -> land  (lsm > {LAND_THRESHOLD}) : {n_land}")
    print(f"  -> ocean (lsm <= {LAND_THRESHOLD}): {n_ocean}")
    print(f"Unique missing coords   : {len(uniq_coords)}")
    print(f"  -> land coords        : {n_uniq_land}")
    print(f"  -> ocean coords       : {n_uniq_ocean}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
