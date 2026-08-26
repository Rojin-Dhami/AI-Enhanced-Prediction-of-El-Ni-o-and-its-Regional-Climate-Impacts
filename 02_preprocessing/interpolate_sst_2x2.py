"""Linear interpolation of missing SST values in the 2x2 Pacific grid.

Rule: for each (time, latitude) line, walk along longitude and linearly
interpolate runs of missing values ONLY when the run has at most
MAX_GAP (=2) consecutive missing cells and is bounded by real values on
both sides (interior gaps). Longer gaps, and gaps at the edges (open on
one side, i.e. land), are left untouched.

Input : data/ERA5_sst_pacific_2x2/era5_sst_pacific_2x2_1980.csv
Output: data/ERA5_sst_pacific_2x2/era5_sst_pacific_2x2_1980_filled.csv
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                      "era5_sst_pacific_2x2_1980.csv")
OUT_CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                       "era5_sst_pacific_2x2_1980_filled.csv")
FILLED_LOG_CSV = os.path.join(BASE, "data", "ERA5_sst_pacific_2x2",
                              "era5_sst_pacific_2x2_1980_filled_locations.csv")
MAX_GAP = 2  # maximum number of consecutive missing cells to fill
def fill_line(values):
    """Linearly interpolate interior NaN runs of length <= MAX_GAP.

    values : 1D float array (ordered by longitude). Returns a filled copy.
    """
    v = values.copy()
    n = len(v)
    isnan = np.isnan(v)
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        # start of a NaN run
        j = i
        while j < n and isnan[j]:
            j += 1
        gap_len = j - i          # run covers indices [i, j)
        left = i - 1             # last valid index before the run
        right = j                # first valid index after the run
        interior = left >= 0 and right < n
        if interior and gap_len <= MAX_GAP:
            lo, hi = v[left], v[right]
            for k in range(i, j):
                frac = (k - left) / (right - left)
                v[k] = lo + frac * (hi - lo)
        i = j
    return v


def main():
    df = pd.read_csv(IN_CSV)
    before = int(df["sst"].isna().sum())

    filled_parts = []
    n_filled = 0
    for (t, lat), grp in df.groupby(["time", "latitude"], sort=False):
        grp = grp.sort_values("longitude").copy()
        orig = grp["sst"].to_numpy(dtype=float)
        new = fill_line(orig)
        # a cell was interpolated if it was NaN before and has a value now
        was_filled = np.isnan(orig) & ~np.isnan(new)
        n_filled += int(was_filled.sum())
        grp["sst"] = new
        grp["interpolated"] = np.where(was_filled, "True", "")
        filled_parts.append(grp)

    out = pd.concat(filled_parts).sort_index()
    out.to_csv(OUT_CSV, index=False)

    # separate log of exactly which cells were filled
    filled_log = out[out["interpolated"] == "True"][["time", "latitude", "longitude", "sst"]]
    filled_log.to_csv(FILLED_LOG_CSV, index=False)

    after = int(out["sst"].isna().sum())
    print(f"input file  : {IN_CSV}")
    print(f"output file : {OUT_CSV}")
    print(f"filled log  : {FILLED_LOG_CSV}")
    print(f"missing before : {before}")
    print(f"cells filled   : {n_filled}  (interior gaps <= {MAX_GAP} along longitude)")
    print(f"missing after  : {after}")


if __name__ == "__main__":
    main()
