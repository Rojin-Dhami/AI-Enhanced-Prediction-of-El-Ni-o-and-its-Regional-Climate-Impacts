"""Check the units of all variables in the ERA5 (and related) metadata files.

Each dataset ships a `*_metadata.json` alongside its CSV/Parquet. The metadata
stores per-variable attributes including `units` and `long_name`. This script
scans one or more metadata JSON files and prints a table of:

    variable | units | long_name

Usage
-----
    # Scan every *_metadata.json under data/
    python3 scripts/check_variable_units.py

    # Scan specific metadata file(s)
    python3 scripts/check_variable_units.py data/ERA5/combined_era5_metadata_2015-2025.json
"""
import glob
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _iter_variable_blocks(obj):
    """Yield (name, attributes) for every 'variables' block found in the JSON.

    Handles both flat metadata ({"variables": {...}}) and year-keyed metadata
    ({"2015": {"variables": {...}}, ...}).
    """
    if not isinstance(obj, dict):
        return

    if "variables" in obj and isinstance(obj["variables"], dict):
        for name, spec in obj["variables"].items():
            attrs = spec.get("attributes", {}) if isinstance(spec, dict) else {}
            yield name, attrs

    for value in obj.values():
        if isinstance(value, dict) and "variables" in value:
            yield from _iter_variable_blocks(value)


def extract_units(metadata_path):
    """Return {variable: {"units": ..., "long_name": ...}} for a metadata file."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    result = {}
    for name, attrs in _iter_variable_blocks(data):
        if name in result:
            continue
        result[name] = {
            "units": attrs.get("units", attrs.get("GRIB_units", "?")),
            "long_name": attrs.get("long_name", attrs.get("GRIB_name", "")),
        }
    return result


def find_metadata_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "**", "*metadata*.json"), recursive=True))


def main(argv):
    paths = argv[1:] or find_metadata_files()
    if not paths:
        print("No metadata JSON files found.")
        return

    for path in paths:
        rel = os.path.relpath(path, BASE_DIR)
        try:
            units = extract_units(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\n{rel}\n  ! could not read: {exc}")
            continue

        print(f"\n{rel}")
        if not units:
            print("  (no variables found)")
            continue

        width = max(len(v) for v in units)
        for var, info in sorted(units.items()):
            print(f"  {var:<{width}}  {info['units']:<10}  {info['long_name']}")


if __name__ == "__main__":
    main(sys.argv)
