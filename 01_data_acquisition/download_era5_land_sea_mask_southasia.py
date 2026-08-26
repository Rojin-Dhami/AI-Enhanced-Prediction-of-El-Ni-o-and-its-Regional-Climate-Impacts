"""Download the ERA5 land-sea mask over the South Asia pressure-levels region.

Covers the domain of era5_pressure_levels_1980_2025.csv
(lat 5..39 N, lon 60..100 E). The land-sea mask (``lsm``) is a static
land-fraction field (0..1).

Output: data/ERA5_land_sea_mask_southasia/ (zip -> extracted .nc)
"""

import os

import cdsapi

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE, "data", "ERA5_land_sea_mask_southasia")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = os.path.join(OUT_DIR, "era5_land_sea_mask_southasia.zip")

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["land_sea_mask"],
    "year": ["2020"],
    "month": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [40, 60, 5, 100],  # North, West, South, East (covers data up to 39N)
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(TARGET)
print(f"Downloaded: {TARGET}")
