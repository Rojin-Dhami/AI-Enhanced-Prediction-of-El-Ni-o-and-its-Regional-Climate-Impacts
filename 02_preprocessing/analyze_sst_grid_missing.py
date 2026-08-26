"""Grid-neighbourhood missing-value analysis for ERA5 Pacific SST.

For each grid cell we look in a square window of +/- MAX_DEG (0.25 .. 2.0 deg)
and count how many neighbouring cells have a value vs are missing. We flag
"mixed" windows: cells where a value exists somewhere within the range but
some/most cells in the range are missing (i.e. land-sea boundary regions).

Example: (lat 20, lon 60) is missing but (lat 20.25, lon 60) has a value.
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(BASE, "data", "ERA5_sst_pacific", "era5_sst_pacific_1980.csv")
OUT = os.path.join(BASE, "data", "ERA5_sst_pacific", "sst_grid_boundary_analysis.csv")

STEP = 0.25          # native grid spacing (deg)
MAX_DEG = 2.0        # neighbourhood half-width (deg)
R = int(round(MAX_DEG / STEP))   # radius in cells (=8)


def main():
    df = pd.read_csv(CSV)
    t0 = sorted(df["time"].unique())[0]          # land mask ~constant in time
    d = df[df["time"] == t0]

    # pivot to 2D grid: rows=lat, cols=lon
    grid = d.pivot(index="latitude", columns="longitude", values="sst").sort_index()
    lats = grid.index.values
    lons = grid.columns.values
    vals = grid.values                            # NaN = missing (land)
    present = ~np.isnan(vals)
    ny, nx = vals.shape

    records = []
    for i in range(ny):
        for j in range(nx):
            i0, i1 = max(0, i - R), min(ny, i + R + 1)
            j0, j1 = max(0, j - R), min(nx, j + R + 1)
            win = present[i0:i1, j0:j1]
            total = win.size
            n_present = int(win.sum())
            n_missing = total - n_present
            # window is "mixed": has at least one value AND at least one missing
            if n_present > 0 and n_missing > 0:
                records.append({
                    "latitude": lats[i],
                    "longitude": lons[j],
                    "cell_missing": not present[i, j],
                    "window_deg": MAX_DEG,
                    "n_cells": total,
                    "n_present": n_present,
                    "n_missing": n_missing,
                    "pct_missing": round(100 * n_missing / total, 1),
                })

    res = pd.DataFrame(records)
    res.to_csv(OUT, index=False)

    print(f"time analysed        : {t0}")
    print(f"grid size            : {ny} lat x {nx} lon (step {STEP} deg)")
    print(f"window               : +/-{MAX_DEG} deg  ({2*R+1}x{2*R+1} cells max)")
    print(f"total cells          : {vals.size}")
    print(f"present / missing     : {int(present.sum())} / {int((~present).sum())}")
    print(f"mixed-window cells    : {len(res)}  -> {OUT}")
    print()

    # cells that are themselves missing but have neighbours with values
    boundary = res[res["cell_missing"]]
    print(f"MISSING cells that have a value within {MAX_DEG} deg : {len(boundary)}")
    print("  (e.g. this cell has no SST, but a nearby cell does)")
    print(boundary.sort_values("pct_missing").head(10).to_string(index=False))
    print()

    # windows where MOST values are missing (>50%) but some exist
    most = res[res["pct_missing"] > 50].sort_values("pct_missing", ascending=False)
    print(f"windows where MOST (>50%) values missing but some exist : {len(most)}")
    print(most.head(10).to_string(index=False))
    print()

    # windows where FEW values are missing (<=25%)
    few = res[res["pct_missing"] <= 25]
    print(f"windows where FEW (<=25%) values missing : {len(few)}")


if __name__ == "__main__":
    main()
