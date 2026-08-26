"""Download ERA5 monthly-mean SST over the tropical Pacific for 1980.

Uses the ERA5 native ~0.25 grid (no 2x2 regridding). The request omits the
'grid' key so the CDS returns data on the original ERA5 resolution.

Region : tropical Pacific  (30N-30S, 120E-80W)  ->  area = [N, W, S, E]
Variable: sea_surface_temperature (K)
Output : NetCDF + long-format CSV + metadata JSON under data/ERA5_sst_pacific/
"""

import os
import json
import time
import zipfile
import glob

import pandas as pd
import xarray as xr
import cdsapi

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ERA5_sst_pacific")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLE = "sea_surface_temperature"
YEAR = 1980
AREA = [30, 120, -30, -80]  # [North, West, South, East] -> tropical Pacific

NC_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_{YEAR}.nc")
CSV_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_{YEAR}.csv")
META_PATH = os.path.join(DATA_DIR, f"era5_sst_pacific_{YEAR}_metadata.json")


def download(retries=3, wait=60):
    """Download the ERA5 SST NetCDF on the native grid; reuse if already present."""
    if os.path.exists(NC_PATH) and os.path.getsize(NC_PATH) > 0:
        print(f"Already downloaded: {NC_PATH}")
        return NC_PATH

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [VARIABLE],
        "year": [str(YEAR)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": AREA,                 # crop to Pacific; NO 'grid' -> native ~0.25 deg
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client = cdsapi.Client()
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(DATASET, request, NC_PATH)
            print(f"Downloaded: {NC_PATH}")
            return NC_PATH
        except Exception as e:
            if attempt == retries:
                raise
            print(f"Retrieve failed (attempt {attempt}/{retries}): {e}")
            print(f"Retrying in {wait}s ...")
            time.sleep(wait)


def open_era5(path):
    """Open an ERA5 download that may be a single NetCDF or a zip of NetCDFs."""
    if zipfile.is_zipfile(path):
        extract_dir = path + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
        ds = xr.open_mfdataset(nc_files, combine="by_coords")
    else:
        ds = xr.open_dataset(path)

    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for c in ("number", "expver"):
        if c in ds.coords:
            ds = ds.drop_vars(c)
    return ds


def to_csv_and_metadata(path):
    ds = open_era5(path).sortby("time")

    df = ds.to_dataframe().reset_index()
    lead = [c for c in ("time", "latitude", "longitude") if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV saved: {CSV_PATH} ({len(df)} rows)")

    times = pd.to_datetime(ds.time.values)
    metadata = {
        "dataset": DATASET,
        "region": {"name": "Tropical Pacific",
                    "north": AREA[0], "west": AREA[1], "south": AREA[2], "east": AREA[3]},
        "grid": {"resolution": "native (~0.25 deg, no regridding)",
                  "n_lat": int(ds.sizes["latitude"]),
                  "n_lon": int(ds.sizes["longitude"])},
        "latitude_range": [float(ds.latitude.min()), float(ds.latitude.max())],
        "longitude_range": [float(ds.longitude.min()), float(ds.longitude.max())],
        "time_range": [str(times[0]), str(times[-1])],
        "n_time": int(ds.sizes["time"]),
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "variables": {
            v: {
                "dims": list(ds[v].dims),
                "shape": list(ds[v].shape),
                "units": ds[v].attrs.get("units", ""),
                "long_name": ds[v].attrs.get("long_name", ds[v].attrs.get("standard_name", "")),
            }
            for v in ds.data_vars
        },
        "csv_rows": int(len(df)),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved: {META_PATH}")


if __name__ == "__main__":
    print(f"Output dir : {DATA_DIR}")
    print(f"Dataset    : {DATASET}")
    print(f"Variable   : {VARIABLE}")
    print(f"Year       : {YEAR}")
    print(f"Region     : N{AREA[0]} W{AREA[1]} S{AREA[2]} E{AREA[3]}  (native grid)\n")

    nc_path = download()
    to_csv_and_metadata(nc_path)
    print("\nDone.")
