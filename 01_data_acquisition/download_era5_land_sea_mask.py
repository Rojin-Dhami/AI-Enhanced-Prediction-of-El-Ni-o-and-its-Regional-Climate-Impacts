"""Download the ERA5 land-sea mask over the tropical Pacific.

The land-sea mask (``lsm``) is a static field (fraction of each grid cell
covered by land, 0..1). We request one year of monthly means and keep a
single field, since the mask does not change in time.

Output: data/ERA5_land_sea_mask/  (downloaded zip -> extracted .nc)
"""

import os

import cdsapi

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE, "data", "ERA5_land_sea_mask")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = os.path.join(OUT_DIR, "era5_land_sea_mask.zip")

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["land_sea_mask"],
    "year": ["2020"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
    ],
    "grid": [2.0, 2.0],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [30, 120, -30, 280],  # North, West, South, East
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(TARGET)
print(f"Downloaded: {TARGET}")
