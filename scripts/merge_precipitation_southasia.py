"""Merge ERA5 precipitation data with existing South Asia dataset.

Input:
  - data/ERA5_single_levels_southasia/era5_single_levels_southasia_1980_2025.csv
  - data/ERA5_precipitation_southasia/era5_precipitation_southasia_1980_2025.csv

Output:
  - data/ERA5_single_levels_southasia/era5_southasia_with_precip_1980_2025.csv
"""

import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SA_DIR = os.path.join(BASE_DIR, "ERA5_single_levels_southasia")
PRECIP_DIR = os.path.join(BASE_DIR, "ERA5_precipitation_southasia")

SA_CSV = os.path.join(SA_DIR, "era5_single_levels_southasia_1980_2025.csv")
PRECIP_CSV = os.path.join(PRECIP_DIR, "era5_precipitation_southasia_1980_2025.csv")
OUTPUT_CSV = os.path.join(SA_DIR, "era5_southasia_with_precip_1980_2025.csv")

def compute_anomalies_and_zscore(df, var_name, train_end='2018-12'):
    """Compute climatological anomalies and z-scores for a variable."""
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Add month column for climatology
    df['month'] = df['time'].dt.month
    
    # Compute climatology (mean per month per grid cell)
    clim = df.groupby(['latitude', 'longitude', 'month'])[var_name].transform('mean')
    df[f'{var_name}_anom'] = df[var_name] - clim
    
    # Compute z-score based on training period only (1980-2018)
    train_mask = df['time'] <= train_end
    train_data = df.loc[train_mask, f'{var_name}_anom']
    
    mean_train = train_data.mean()
    std_train = train_data.std()
    
    df[f'{var_name}_anom_z'] = (df[f'{var_name}_anom'] - mean_train) / std_train
    
    # Drop temporary month column
    df = df.drop('month', axis=1)
    
    return df, mean_train, std_train


def main():
    print("="*70)
    print("Merging Precipitation with South Asia Dataset")
    print("="*70)
    
    # Load existing South Asia data
    print(f"\n1. Loading existing SA data: {SA_CSV}")
    df_sa = pd.read_csv(SA_CSV)
    # Strip whitespace from column names
    df_sa.columns = df_sa.columns.str.strip()
    # Normalize time to date-only format
    df_sa['time'] = pd.to_datetime(df_sa['time']).dt.date.astype(str)
    print(f"   Shape: {df_sa.shape}")
    print(f"   Columns: {list(df_sa.columns)}")
    
    # Load precipitation data
    print(f"\n2. Loading precipitation data: {PRECIP_CSV}")
    df_precip = pd.read_csv(PRECIP_CSV)
    # Normalize time to date-only format
    df_precip['time'] = pd.to_datetime(df_precip['time']).dt.date.astype(str)
    print(f"   Shape: {df_precip.shape}")
    print(f"   Columns: {list(df_precip.columns)}")
    
    # Rename precipitation column if needed
    if 'tp' in df_precip.columns:
        precip_var = 'tp'
    elif 'total_precipitation' in df_precip.columns:
        precip_var = 'total_precipitation'
    else:
        raise ValueError(f"Precipitation variable not found. Available: {df_precip.columns.tolist()}")
    
    # Standardize column name
    df_precip = df_precip.rename(columns={precip_var: 'tp'})
    
    # Merge on time, latitude, longitude
    print(f"\n3. Merging datasets on [time, latitude, longitude]...")
    df_merged = pd.merge(
        df_sa,
        df_precip[['time', 'latitude', 'longitude', 'tp']],
        on=['time', 'latitude', 'longitude'],
        how='left'
    )
    print(f"   Merged shape: {df_merged.shape}")
    
    # Check for missing values
    missing_tp = df_merged['tp'].isna().sum()
    if missing_tp > 0:
        print(f"   WARNING: {missing_tp} missing precipitation values")
        print(f"   Filling with 0 (assuming missing over land/ocean mask)")
        df_merged['tp'] = df_merged['tp'].fillna(0)
    
    # Compute anomalies and z-scores for ALL input variables
    print(f"\n4. Computing anomalies and z-scores for all variables...")
    
    variables_to_process = ['t2m', 'tp', 'msl', 'avg_iews', 'avg_inss']
    stats = {}
    
    for var in variables_to_process:
        print(f"\n   Processing {var}...")
        df_merged, mean_val, std_val = compute_anomalies_and_zscore(
            df_merged, var, train_end='2018-12'
        )
        stats[var] = {'mean': mean_val, 'std': std_val}
        print(f"     Train period (1980-2018): Mean={mean_val:.6f}, Std={std_val:.6f}")
    
    # Save merged dataset
    print(f"\n5. Saving merged dataset: {OUTPUT_CSV}")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} columns")
    
    # Summary statistics
    print(f"\n6. Final dataset summary:")
    print(f"   Time range: {df_merged['time'].min()} to {df_merged['time'].max()}")
    print(f"   Spatial extent: lat [{df_merged['latitude'].min()}, {df_merged['latitude'].max()}], "
          f"lon [{df_merged['longitude'].min()}, {df_merged['longitude'].max()}]")
    
    print(f"\n7. Variables with anomalies computed:")
    for var in variables_to_process:
        if f'{var}_anom_z' in df_merged.columns:
            print(f"   ✅ {var}: raw, _anom, _anom_z")
            z_scores = df_merged[f'{var}_anom_z']
            print(f"      Z-score range: [{z_scores.min():.3f}, {z_scores.max():.3f}], "
                  f"mean: {z_scores.mean():.3f}")
    
    print(f"\n8. Model-ready input channels (z-scored):")
    model_channels = [f'{v}_anom_z' for v in variables_to_process]
    available = [c for c in model_channels if c in df_merged.columns]
    print(f"   Total: {len(available)}/5 channels ready")
    for ch in available:
        print(f"   ✅ {ch}")
    
    print("\n" + "="*70)
    print("Merge complete! Dataset ready for dual-encoder CNN-TCN")
    print("="*70)
    print(f"Output: {OUTPUT_CSV}")
    print(f"5 input channels prepared: t2m, tp, msl, avg_iews, avg_inss")
    print("Next: Build input tensors (12, 18, 21, 5) for model training")
    print("="*70)


if __name__ == "__main__":
    main()
