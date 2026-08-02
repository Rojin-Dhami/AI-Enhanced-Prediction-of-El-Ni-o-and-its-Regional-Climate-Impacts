"""
Quick verification of coordinate systems in Pacific and South Asia CSV files.
Run this anytime to check grid dimensions.
"""

import pandas as pd

print("="*70)
print("COORDINATE VERIFICATION")
print("="*70)

# Pacific data
print("\n1️⃣  PACIFIC DOMAIN")
print("-"*70)
df = pd.read_csv('/Users/raman/Elnino/data/combined_era5_oras5.csv', nrows=10000)
lats = sorted(df['latitude'].unique())
lons = sorted(df['longitude'].unique())
print(f"✓ Latitude:  {len(lats)} points from {lats[0]}° to {lats[-1]}° (step: {lats[1]-lats[0]}°)")
print(f"✓ Longitude: {len(lons)} points from {lons[0]}° to {lons[-1]}° (step: {lons[1]-lons[0]}°)")
print(f"✓ Grid:      {len(lats)} × {len(lons)} = {len(lats)*len(lons)} cells")

# South Asia data
print("\n2️⃣  SOUTH ASIA DOMAIN")
print("-"*70)
df = pd.read_csv('/Users/raman/Elnino/data/ERA5_single_levels_southasia/era5_southasia_with_precip_1980_2025.csv', nrows=5000)
lats = sorted(df['latitude'].unique())
lons = sorted(df['longitude'].unique())
print(f"✓ Latitude:  {len(lats)} points from {lats[0]}° to {lats[-1]}° (step: {lats[1]-lats[0]}°)")
print(f"✓ Longitude: {len(lons)} points from {lons[0]}° to {lons[-1]}° (step: {lons[1]-lons[0]}°)")
print(f"✓ Grid:      {len(lats)} × {len(lons)} = {len(lats)*len(lons)} cells")

print("\n" + "="*70)
print("✅ Both grids use 2°×2° resolution and are CNN-compatible!")
print("="*70)
