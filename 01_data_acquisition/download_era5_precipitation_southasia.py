"""Download ERA5 monthly-mean total precipitation over South Asia for 1980-2025.

Region: South Asia (5°N-40°N, 60°E-100°E) on 2x2 degree grid
Variable: total_precipitation (m) - monthly accumulation
Time: 1980-2025 (552 months)
Output: data/ERA5_precipitation_southasia/
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
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ERA5_precipitation_southasia")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLE = "total_precipitation"
START_YEAR = 1980
END_YEAR = 2025
AREA = [40, 60, 5, 100]  # [North, West, South, East] -> South Asia
GRID = [2.0, 2.0]        # 2 x 2 degree regridding

NC_PATH = os.path.join(DATA_DIR, f"era5_precipitation_southasia_{START_YEAR}_{END_YEAR}.nc")
CSV_PATH = os.path.join(DATA_DIR, f"era5_precipitation_southasia_{START_YEAR}_{END_YEAR}.csv")
META_PATH = os.path.join(DATA_DIR, f"era5_precipitation_southasia_{START_YEAR}_{END_YEAR}_metadata.json")


def download(retries=3, wait=60):
    """Download the ERA5 precipitation NetCDF; reuse if already present."""
    if os.path.exists(NC_PATH) and os.path.getsize(NC_PATH) > 0:
        print(f"Already downloaded: {NC_PATH}")
        return NC_PATH

    years = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [VARIABLE],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "area": AREA,                 # crop to South Asia
        "grid": GRID,                 # 2 x 2 degree regridding
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"Requesting {len(years)} years × {len(months)} months = {len(years)*len(months)} time steps")
    print("This may take 10-20 minutes depending on CDS queue...")
    
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

    # Standardize coordinate names
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for c in ("number", "expver"):
        if c in ds.coords:
            ds = ds.drop_vars(c)
    return ds


def to_csv_and_metadata(path):
    """Convert NetCDF to long-format CSV and save metadata."""
    ds = open_era5(path).sortby("time")

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()
    
    # Reorder columns: time, lat, lon, then variables
    lead = [c for c in ("time", "latitude", "longitude") if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]
    
    # Save CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV saved: {CSV_PATH} ({len(df):,} rows)")

    # Create metadata
    times = pd.to_datetime(ds.time.values)
    metadata = {
        "dataset": DATASET,
        "region": {
            "name": "South Asia",
            "north": AREA[0],
            "west": AREA[1],
            "south": AREA[2],
            "east": AREA[3]
        },
        "grid_deg": GRID,
        "latitude_range": [float(ds.latitude.min()), float(ds.latitude.max())],
        "longitude_range": [float(ds.longitude.min()), float(ds.longitude.max())],
        "n_lat": int(ds.sizes["latitude"]),
        "n_lon": int(ds.sizes["longitude"]),
        "time_range": [str(times[0]), str(times[-1])],
        "n_time": int(ds.sizes["time"]),
        "variables": [VARIABLE],
        "variable_details": {
            v: {
                "dims": list(ds[v].dims),
                "shape": list(ds[v].shape),
                "units": ds[v].attrs.get("units", ""),
                "long_name": ds[v].attrs.get("long_name", ds[v].attrs.get("standard_name", "")),
            }
            for v in ds.data_vars
        },
        "csv_rows": int(len(df)),
        "note": "monthly_averaged_reanalysis - monthly accumulation in meters"
    }
    
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved: {META_PATH}")
    
    # Print summary statistics
    if VARIABLE in df.columns:
        print(f"\n{VARIABLE} statistics:")
        print(f"  Min:    {df[VARIABLE].min():.6f} m")
        print(f"  Max:    {df[VARIABLE].max():.6f} m")
        print(f"  Mean:   {df[VARIABLE].mean():.6f} m")
        print(f"  Median: {df[VARIABLE].median():.6f} m")


if __name__ == "__main__":
    print("="*70)
    print("ERA5 Precipitation Download for South Asia")
    print("="*70)
    print(f"Output dir : {DATA_DIR}")
    print(f"Dataset    : {DATASET}")
    print(f"Variable   : {VARIABLE}")
    print(f"Years      : {START_YEAR}-{END_YEAR}")
    print(f"Region     : N{AREA[0]}° W{AREA[1]}° S{AREA[2]}° E{AREA[3]}°")
    print(f"Grid       : {GRID[0]}° × {GRID[1]}°")
    print("="*70)
    print("\nNote: You need a valid CDS API key configured in ~/.cdsapirc")
    print("Register at: https://cds.climate.copernicus.eu/\n")

    nc_path = download()
    to_csv_and_metadata(nc_path)
    
    print("\n" + "="*70)
    print("Download complete! Next steps:")
    print("1. Merge with existing South Asia data")
    print("2. Compute anomalies and z-scores")
    print("3. Update CNN-TCN model to use 5 channels")
    print("="*70)
