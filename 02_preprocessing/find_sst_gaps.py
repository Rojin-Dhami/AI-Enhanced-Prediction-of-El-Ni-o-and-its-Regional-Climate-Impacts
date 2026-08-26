"""Find missing-value gaps in ERA5 Pacific SST along each latitude line.

A "gap" is a run of missing (NaN) SST cells bounded by valid values on both
sides. The whole gap is captured regardless of length, surfacing the
land-sea boundary pattern:  value -> missing... -> value.

Usage:
    python scripts/find_sst_gaps.py                # print first N gaps
    python scripts/find_sst_gaps.py --save         # also write all gaps to CSV
"""

import os
import argparse
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(BASE, "data", "ERA5_sst_pacific", "era5_sst_pacific_1980.csv")
OUT = os.path.join(BASE, "data", "ERA5_sst_pacific", "sst_missing_gaps.csv")

STEP = 0.25          # native grid spacing (deg)
PAD = 1              # extra valid cells to include before & after each gap


def find_gaps(d, time_label):
    """Yield gap blocks (value -> missing... -> value) for one timestamp."""
    rows = []
    for lat, g in d.groupby("latitude"):
        g = g.sort_values("longitude").reset_index(drop=True)
        missing = g["sst"].isna().values
        n = len(g)
        i = 0
        while i < n:
            if missing[i]:
                j = i
                while j < n and missing[j]:
                    j += 1
                gap = j - i
                # whole gap bounded by present values on both sides
                if gap >= 1 and i > 0 and j < n \
                        and not missing[i - 1] and not missing[j]:
                    lo = max(0, i - 1 - PAD)
                    hi = min(n, j + 1 + PAD)
                    block = g.iloc[lo:hi].copy()
                    # ensure the block begins and ends on a real value
                    bm = block["sst"].isna().values
                    first = bm.argmin()
                    last = len(bm) - bm[::-1].argmin()
                    block = block.iloc[first:last]
                    rows.append(block)
                i = j
            else:
                i += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write all gaps to CSV")
    ap.add_argument("--show", type=int, default=6, help="number of gaps to print")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    t0 = sorted(df["time"].unique())[0]          # land mask ~constant in time
    d = df[df["time"] == t0].sort_values(["latitude", "longitude"])

    blocks = find_gaps(d, t0)
    print(f"time analysed : {t0}")
    print(f"gaps found    : {len(blocks)}\n")

    for block in blocks[:args.show]:
        lat = block["latitude"].iloc[0]
        lo, hi = block["longitude"].iloc[0], block["longitude"].iloc[-1]
        print(f"# lat {lat}: gap (lon {lo}..{hi})")
        for _, r in block.iterrows():
            v = "" if pd.isna(r["sst"]) else round(r["sst"], 5)
            print(f"{r['time']},{r['latitude']},{r['longitude']},{v}")
        print()

    if args.save and blocks:
        out = pd.concat(blocks, ignore_index=True)[
            ["time", "latitude", "longitude", "sst"]
        ]
        out.to_csv(OUT, index=False)
        print(f"saved {len(out)} rows ({len(blocks)} gaps) -> {OUT}")


if __name__ == "__main__":
    main()
