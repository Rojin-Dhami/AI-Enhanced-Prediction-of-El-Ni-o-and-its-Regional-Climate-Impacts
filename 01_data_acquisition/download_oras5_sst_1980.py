"""Download ORAS5 sea surface temperature for 1980 and convert to CSV.

Dataset : reanalysis-oras5 (consolidated, single_level)
Variable: sea_surface_temperature
Output  : downloaded archive + long-format CSV + metadata JSON under
          data/ORAS5_sst/
"""

import os
import json
import time
import zipfile
import glob

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ORAS5_sst")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DATASET = "reanalysis-oras5"
VARIABLE = "sea_surface_temperature"
YEAR = 1980

# Tropical Pacific domain (matches ERA5 SST work): 30N-30S, 120E-80W.
# ORAS5 nav_lon is in [-180, 180], and the region crosses the dateline,
# so the longitude condition is (lon >= 120) OR (lon <= -80).
AREA = {"north": 30, "south": -30, "west": 120, "east": -80}

DOWNLOAD_PATH = os.path.join(DATA_DIR, f"oras5_sst_{YEAR}.zip")
CSV_PATH = os.path.join(DATA_DIR, f"oras5_sst_{YEAR}.csv")
CSV_PACIFIC_PATH = os.path.join(DATA_DIR, f"oras5_sst_pacific_{YEAR}.csv")
CSV_ERA5GRID_PATH = os.path.join(DATA_DIR, f"oras5_sst_pacific_era5grid_{YEAR}.csv")
META_PATH = os.path.join(DATA_DIR, f"oras5_sst_{YEAR}_metadata.json")
META_PACIFIC_PATH = os.path.join(DATA_DIR, f"oras5_sst_pacific_{YEAR}_metadata.json")
META_ERA5GRID_PATH = os.path.join(DATA_DIR, f"oras5_sst_pacific_era5grid_{YEAR}_metadata.json")

# Target ERA5 grid over the tropical Pacific (0.25 deg regular lat/lon),
# matching data/ERA5_sst_pacific: latitude 30 -> -30, longitude 120 -> 280.
ERA5_LAT = np.round(np.arange(30.0, -30.0001, -0.25), 3)   # 241 points, descending
ERA5_LON = np.round(np.arange(120.0, 280.0001, 0.25), 3)   # 641 points, 0-360


def download(retries=3, wait=60):
    """Download the ORAS5 SST archive; reuse if already present."""
    if os.path.exists(DOWNLOAD_PATH) and os.path.getsize(DOWNLOAD_PATH) > 0:
        print(f"Already downloaded: {DOWNLOAD_PATH}")
        return DOWNLOAD_PATH

    request = {
        "product_type": ["consolidated"],
        "vertical_resolution": "single_level",
        "variable": [VARIABLE],
        "year": [str(YEAR)],
        "month": [f"{m:02d}" for m in range(1, 13)],
    }

    client = cdsapi.Client()
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(DATASET, request, DOWNLOAD_PATH)
            print(f"Downloaded: {DOWNLOAD_PATH}")
            return DOWNLOAD_PATH
        except Exception as e:
            if attempt == retries:
                raise
            print(f"Retrieve failed (attempt {attempt}/{retries}): {e}")
            print(f"Retrying in {wait}s ...")
            time.sleep(wait)


def open_oras5(path):
    """Open an ORAS5 download that may be a single NetCDF or a zip of NetCDFs."""
    if zipfile.is_zipfile(path):
        extract_dir = path + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
        datasets = [xr.open_dataset(f) for f in nc_files]
        ds = xr.combine_by_coords(datasets, combine_attrs="override")
    else:
        ds = xr.open_dataset(path)

    if "time_counter" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"time_counter": "time"})
    return ds


def crop_pacific(ds):
    """Crop the curvilinear ORAS5 grid to the tropical Pacific domain.

    Uses the 2D nav_lat/nav_lon coordinates to build a boolean mask, then
    trims the y/x extent to the bounding box of matching cells. Cells
    outside the region are set to NaN.
    """
    lat = ds["nav_lat"]
    lon = ds["nav_lon"]

    lat_mask = (lat >= AREA["south"]) & (lat <= AREA["north"])
    # Region crosses the dateline: 120E .. 180 .. -80 (=280E)
    lon_mask = (lon >= AREA["west"]) | (lon <= AREA["east"])
    mask = lat_mask & lon_mask

    # Trim to the bounding box of selected cells to shrink the grid.
    ys = mask.any(dim="x")
    xs = mask.any(dim="y")
    y_idx = ys.values.nonzero()[0]
    x_idx = xs.values.nonzero()[0]
    ds = ds.isel(y=slice(y_idx.min(), y_idx.max() + 1),
                 x=slice(x_idx.min(), x_idx.max() + 1))

    mask = ((ds["nav_lat"] >= AREA["south"]) & (ds["nav_lat"] <= AREA["north"])
            & ((ds["nav_lon"] >= AREA["west"]) | (ds["nav_lon"] <= AREA["east"])))
    return ds.where(mask)


def _edges(centers):
    """Cell edges (n+1) from monotonic 1D cell centers, extrapolating the ends."""
    centers = np.asarray(centers, dtype=float)
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _overlap_matrix(src_edges, tgt_edges, use_sin=False):
    """First-order conservative overlap weights W[tgt, src] between 1D cells.

    Overlap is the length of intersection of source and target cells. For
    latitude, area on a sphere is proportional to sin(lat), so pass
    use_sin=True to weight by the difference of sines (spherical area).
    """
    s = np.sort(src_edges)
    t = np.sort(tgt_edges)
    if use_sin:
        s = np.sin(np.deg2rad(s))
        t = np.sin(np.deg2rad(t))
    s_lo, s_hi = s[:-1], s[1:]
    t_lo, t_hi = t[:-1], t[1:]
    # Broadcast: rows = target cells, cols = source cells.
    lo = np.maximum(t_lo[:, None], s_lo[None, :])
    hi = np.minimum(t_hi[:, None], s_hi[None, :])
    return np.clip(hi - lo, 0.0, None)


def regrid_to_era5(ds):
    """Conservatively regrid ORAS5 SST onto the ERA5 tropical-Pacific grid.

    In the tropics the ORAS5 (ORCA025) grid is effectively rectilinear, so we
    reduce nav_lat/nav_lon to 1D coordinates and apply exact separable
    first-order conservative (area-weighted) remapping. Land/NaN cells are
    excluded by normalising with the valid-overlap weights, so land never
    contaminates ocean cells and all-land target cells stay NaN.
    """
    var = list(ds.data_vars)[0]
    lon360 = (ds["nav_lon"] % 360)

    mask = ((ds["nav_lat"] >= AREA["south"]) & (ds["nav_lat"] <= AREA["north"])
            & (lon360 >= 120) & (lon360 <= 280))
    ys = mask.any("x").values.nonzero()[0]
    xs = mask.any("y").values.nonzero()[0]
    ysl = slice(ys.min(), ys.max() + 1)
    xsl = slice(xs.min(), xs.max() + 1)

    sub = ds.isel(y=ysl, x=xsl)
    sub_lat = sub["nav_lat"].values
    sub_lon = (sub["nav_lon"].values % 360)
    data = sub[var].values  # (time, y, x), degrees C

    # Collapse to 1D source coordinates (grid is rectilinear here).
    lat_src = np.nanmean(sub_lat, axis=1)   # varies with y
    lon_src = np.nanmean(sub_lon, axis=0)   # varies with x

    # Order source ascending for overlap computation; remember the reordering.
    lat_order = np.argsort(lat_src)
    lon_order = np.argsort(lon_src)
    lat_src_s = lat_src[lat_order]
    lon_src_s = lon_src[lon_order]

    # Target ERA5 grid (compute ascending, reorder to ERA5 layout at the end).
    lat_tgt_s = np.sort(ERA5_LAT)
    lon_tgt_s = np.sort(ERA5_LON)

    Wlat = _overlap_matrix(_edges(lat_src_s), _edges(lat_tgt_s), use_sin=True)   # (Tlat, Slat)
    Wlon = _overlap_matrix(_edges(lon_src_s), _edges(lon_tgt_s), use_sin=False)  # (Tlon, Slon)

    nt = data.shape[0]
    out = np.full((nt, len(lat_tgt_s), len(lon_tgt_s)), np.nan, dtype=float)
    for i in range(nt):
        d = data[i][lat_order][:, lon_order]          # (Slat, Slon), ascending
        valid = np.isfinite(d).astype(float)
        d0 = np.where(np.isfinite(d), d, 0.0)
        num = Wlat @ d0 @ Wlon.T                        # (Tlat, Tlon)
        den = Wlat @ valid @ Wlon.T
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(den > 0, num / den, np.nan)
        out[i] = res

    # Reorder target axes back to ERA5 layout (lat descending, lon ascending).
    asc_lat = np.sort(ERA5_LAT)
    lat_to_asc = np.searchsorted(asc_lat, ERA5_LAT)
    asc_lon = np.sort(ERA5_LON)
    lon_to_asc = np.searchsorted(asc_lon, ERA5_LON)
    out = out[:, lat_to_asc, :][:, :, lon_to_asc]

    # degC -> K
    out = out + 273.15

    times = pd.to_datetime(ds["time"].values)
    da = xr.DataArray(
        out,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": ERA5_LAT, "longitude": ERA5_LON},
        name="sst",
        attrs={"units": "K", "long_name": "Sea Surface Temperature",
               "regridding": "first-order conservative (area-weighted) from ORAS5 ORCA025"},
    )
    return da.to_dataset()


def _write_csv(ds, csv_path):
    df = ds.to_dataframe().reset_index()

    # Identify the SST data variable (ORAS5 name is 'sosstsst', units degC).
    data_vars = [v for v in ds.data_vars if v in df.columns]

    # Arrange like the ERA5 SST CSV: time, latitude, longitude, sst.
    df = df.rename(columns={"nav_lat": "latitude", "nav_lon": "longitude"})
    if data_vars:
        df = df.rename(columns={data_vars[0]: "sst"})
        # Drop land / outside-region cells, then convert degC -> K.
        df = df.dropna(subset=["sst"])
        df["sst"] = df["sst"] + 273.15
    # Longitude to 0-360 to match ERA5.
    df["longitude"] = df["longitude"] % 360

    df = df[["time", "latitude", "longitude", "sst"]]
    df = df.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path} ({len(df)} rows)")
    return df


def _metadata(ds, n_rows):
    metadata = {
        "dataset": DATASET,
        "variable": VARIABLE,
        "year": YEAR,
        "region": {"name": "Tropical Pacific", **AREA},
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "nav_lat_range": [float(ds["nav_lat"].min()), float(ds["nav_lat"].max())],
        "nav_lon_range": [float(ds["nav_lon"].min()), float(ds["nav_lon"].max())],
        "variables": {
            v: {
                "dims": list(ds[v].dims),
                "shape": list(ds[v].shape),
                "units": ds[v].attrs.get("units", ""),
                "long_name": ds[v].attrs.get("long_name", ds[v].attrs.get("standard_name", "")),
            }
            for v in ds.data_vars
        },
        "csv_rows": int(n_rows),
    }
    if "time" in ds.coords:
        times = pd.to_datetime(ds.time.values)
        metadata["time_range"] = [str(times[0]), str(times[-1])]
        metadata["n_time"] = int(len(times))
    return metadata


def _write_era5grid_csv(ds_grid, csv_path):
    """Write the ERA5-grid regridded SST as time, latitude, longitude, sst."""
    df = ds_grid.to_dataframe().reset_index()
    df = df[["time", "latitude", "longitude", "sst"]]
    df = df.dropna(subset=["sst"])
    df = df.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path} ({len(df)} rows)")
    return df


def to_csv_and_metadata(path):
    ds = open_oras5(path)
    if "time" in ds.coords:
        ds = ds.sortby("time")

    ds_pac = crop_pacific(ds)
    df_pac = _write_csv(ds_pac, CSV_PACIFIC_PATH)

    with open(META_PACIFIC_PATH, "w") as f:
        json.dump(_metadata(ds_pac, len(df_pac)), f, indent=2, default=str)
    print(f"Metadata saved: {META_PACIFIC_PATH}")

    # Conservative regrid onto the ERA5 grid.
    ds_grid = regrid_to_era5(ds)
    df_grid = _write_era5grid_csv(ds_grid, CSV_ERA5GRID_PATH)

    # Conservation check: area-weighted mean SST before vs after regridding.
    native = crop_pacific(ds)
    var = list(native.data_vars)[0]
    w = np.cos(np.deg2rad(native["nav_lat"]))
    native_mean = float((native[var] * w).sum() / (w * np.isfinite(native[var])).sum()) + 273.15
    wg = np.cos(np.deg2rad(ds_grid["latitude"]))
    grid_mean = float((ds_grid["sst"] * wg).sum() / (wg * np.isfinite(ds_grid["sst"])).sum())
    print(f"Area-weighted mean SST  native: {native_mean:.4f} K   "
          f"regridded: {grid_mean:.4f} K   diff: {grid_mean - native_mean:+.4f} K")

    times = pd.to_datetime(ds_grid.time.values)
    meta = {
        "dataset": DATASET,
        "variable": VARIABLE,
        "year": YEAR,
        "region": {"name": "Tropical Pacific", **AREA},
        "target_grid": {
            "description": "ERA5 tropical-Pacific 0.25 deg regular lat/lon",
            "latitude": {"start": 30.0, "stop": -30.0, "step": -0.25, "n": len(ERA5_LAT)},
            "longitude": {"start": 120.0, "stop": 280.0, "step": 0.25, "n": len(ERA5_LON)},
        },
        "regridding": "first-order conservative (area-weighted), spherical sin-lat weights",
        "units": "K",
        "conservation_check": {
            "native_area_weighted_mean_K": round(native_mean, 4),
            "regridded_area_weighted_mean_K": round(grid_mean, 4),
            "difference_K": round(grid_mean - native_mean, 4),
        },
        "time_range": [str(times[0]), str(times[-1])],
        "n_time": int(len(times)),
        "csv_rows": int(len(df_grid)),
    }
    with open(META_ERA5GRID_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Metadata saved: {META_ERA5GRID_PATH}")


if __name__ == "__main__":
    print(f"Output dir : {DATA_DIR}")
    print(f"Dataset    : {DATASET}")
    print(f"Variable   : {VARIABLE}")
    print(f"Year       : {YEAR}\n")

    path = download()
    to_csv_and_metadata(path)
    print("\nDone.")
