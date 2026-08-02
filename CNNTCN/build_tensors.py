"""
Tensor Construction Pipeline for Dual-Encoder CNN-TCN ENSO Forecasting

This script transforms preprocessed CSV data into 5D tensors for model training.

Input Files:
    - Pacific domain: /data/combined_era5_oras5.csv (6 channels)
    - South Asia domain: /data/ERA5_single_levels_southasia/era5_southasia_with_precip_1980_2025.csv (5 channels)
    - ENSO target: /nina34.csv or /meiv2.csv

Output Tensors:
    - X_pacific.npy: (N, 12, 30, 100, 6) - Pacific spatial-temporal features
    - X_sa.npy: (N, 12, 18, 21, 5) - South Asia spatial-temporal features
    - y_enso.npy: (N,) - ENSO index at t+6 months
    - y_impact.npy: (N, 2, 378) - SA temperature & precipitation (JJAS average)
    - dates.npy: (N,) - Corresponding dates for each sample

Processing:
    1. Load and pivot CSV data to 4D arrays
    2. Create 12-month sliding windows
    3. Apply 6-month lead time for ENSO prediction
    4. Compute JJAS seasonal averages for SA impacts
    5. Validate data integrity at each step
    6. Save tensors with verification

Author: Dual-Encoder Pipeline
Date: 2026-08-02
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration parameters."""
    
    # Directory paths
    BASE_DIR = Path("/Users/raman/Elnino")
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "CNNTCN" / "tensors"
    
    # Input files
    PACIFIC_CSV = DATA_DIR / "combined_era5_oras5.csv"
    SA_CSV = DATA_DIR / "ERA5_single_levels_southasia" / "era5_southasia_with_precip_1980_2025.csv"
    ENSO_CSV = BASE_DIR / "nina34.csv"
    
    # Pacific domain parameters
    PACIFIC_LAT_MIN, PACIFIC_LAT_MAX = -30, 30
    PACIFIC_LON_MIN, PACIFIC_LON_MAX = 120, 280  # 120°E to 280°E (=80°W)
    PACIFIC_LAT_RES, PACIFIC_LON_RES = 2, 2
    PACIFIC_N_LAT = 31  # -30 to 30 in 2° steps = 31 points
    PACIFIC_N_LON = 81  # 120 to 280 in 2° steps = 81 points
    
    # South Asia domain parameters
    SA_LAT_MIN, SA_LAT_MAX = 5, 39  # 39°N (not 40°N) due to 2° grid alignment
    SA_LON_MIN, SA_LON_MAX = 60, 100
    SA_LAT_RES, SA_LON_RES = 2, 2
    SA_N_LAT = 18  # 5° to 39° in 2° steps = 18 points
    SA_N_LON = 21  # 60° to 100° in 2° steps = 21 points
    
    # Variable names (z-scored columns)
    PACIFIC_VARS = [
        'sst_anom_z',
        'msl_anom_z',
        'avg_iews_anom_z',
        'avg_inss_anom_z',
        'sohtc300_anom_z',
        'so20chgt_anom_z'
    ]
    
    SA_VARS = [
        't2m_anom_z',
        'tp_anom_z',
        'msl_anom_z',
        'avg_iews_anom_z',
        'avg_inss_anom_z'
    ]
    
    # Temporal parameters
    WINDOW_SIZE = 12  # months of input history
    LEAD_TIME = 6     # months ahead for ENSO prediction
    JJAS_MONTHS = [6, 7, 8, 9]  # June, July, August, September
    
    # Time range
    START_YEAR = 1980
    END_YEAR = 2025
    
    # Data splits
    TRAIN_END = "2018-12"
    VAL_START = "2019-01"
    VAL_END = "2022-12"
    TEST_START = "2023-01"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging():
    """Initialize logging with timestamps."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def create_output_directory():
    """Create output directory if it doesn't exist."""
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")


def validate_file_exists(filepath: Path, description: str):
    """Validate that required input file exists."""
    if not filepath.exists():
        logger.error(f"{description} not found: {filepath}")
        raise FileNotFoundError(f"Required file missing: {filepath}")
    logger.info(f"✓ Found {description}: {filepath.name}")


def print_array_stats(arr: np.ndarray, name: str):
    """Print detailed statistics for a numpy array."""
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Shape: {arr.shape}")
    logger.info(f"  Dtype: {arr.dtype}")
    logger.info(f"  Min: {arr.min():.6f}")
    logger.info(f"  Max: {arr.max():.6f}")
    logger.info(f"  Mean: {arr.mean():.6f}")
    logger.info(f"  Std: {arr.std():.6f}")
    logger.info(f"  NaN count: {np.isnan(arr).sum()}")
    logger.info(f"  Inf count: {np.isinf(arr).sum()}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_pacific_data() -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load Pacific domain data from CSV.
    
    Returns:
        data: Shape (T, lat, lon, channels) where T = time steps
        dates: DatetimeIndex of time steps
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING PACIFIC DOMAIN DATA")
    logger.info("="*70)
    
    validate_file_exists(Config.PACIFIC_CSV, "Pacific CSV")
    
    # Load CSV
    logger.info("Reading Pacific CSV...")
    df = pd.read_csv(Config.PACIFIC_CSV)
    logger.info(f"  Rows: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['time', 'latitude', 'longitude'] + Config.PACIFIC_VARS
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in Pacific CSV: {missing_cols}")
    logger.info(f"✓ All required columns present")
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Get unique dimensions
    unique_times = sorted(df['time'].unique())
    unique_lats = sorted(df['latitude'].unique(), reverse=True)  # North to South
    unique_lons = sorted(df['longitude'].unique())
    
    n_times = len(unique_times)
    n_lats = len(unique_lats)
    n_lons = len(unique_lons)
    n_channels = len(Config.PACIFIC_VARS)
    
    logger.info(f"\nDimensions:")
    logger.info(f"  Time steps: {n_times}")
    logger.info(f"  Latitudes: {n_lats} (from {unique_lats[0]}° to {unique_lats[-1]}°)")
    logger.info(f"  Longitudes: {n_lons} (from {unique_lons[0]}° to {unique_lons[-1]}°)")
    logger.info(f"  Channels: {n_channels}")
    
    # Initialize array
    data = np.zeros((n_times, n_lats, n_lons, n_channels), dtype=np.float32)
    
    # Create coordinate mapping
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    time_to_idx = {time: i for i, time in enumerate(unique_times)}
    
    # Fill array
    logger.info("\nFilling 4D array...")
    for _, row in df.iterrows():
        t_idx = time_to_idx[row['time']]
        lat_idx = lat_to_idx[row['latitude']]
        lon_idx = lon_to_idx[row['longitude']]
        
        for ch_idx, var in enumerate(Config.PACIFIC_VARS):
            data[t_idx, lat_idx, lon_idx, ch_idx] = row[var]
    
    logger.info(f"✓ Array filled successfully")
    
    # Validate
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        logger.warning(f"⚠ Found {nan_count} NaN values in Pacific data")
    
    print_array_stats(data, "Pacific Data")
    
    dates = pd.DatetimeIndex(unique_times)
    
    return data, dates


def load_south_asia_data() -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load South Asia domain data from CSV.
    
    Returns:
        data: Shape (T, lat, lon, channels) where T = time steps
        dates: DatetimeIndex of time steps
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING SOUTH ASIA DOMAIN DATA")
    logger.info("="*70)
    
    validate_file_exists(Config.SA_CSV, "South Asia CSV")
    
    # Load CSV
    logger.info("Reading South Asia CSV...")
    df = pd.read_csv(Config.SA_CSV)
    logger.info(f"  Rows: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['time', 'latitude', 'longitude'] + Config.SA_VARS
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in South Asia CSV: {missing_cols}")
    logger.info(f"✓ All required columns present")
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Get unique dimensions
    unique_times = sorted(df['time'].unique())
    unique_lats = sorted(df['latitude'].unique(), reverse=True)  # North to South
    unique_lons = sorted(df['longitude'].unique())
    
    n_times = len(unique_times)
    n_lats = len(unique_lats)
    n_lons = len(unique_lons)
    n_channels = len(Config.SA_VARS)
    
    logger.info(f"\nDimensions:")
    logger.info(f"  Time steps: {n_times}")
    logger.info(f"  Latitudes: {n_lats} (from {unique_lats[0]}° to {unique_lats[-1]}°)")
    logger.info(f"  Longitudes: {n_lons} (from {unique_lons[0]}° to {unique_lons[-1]}°)")
    logger.info(f"  Channels: {n_channels}")
    
    # Verify expected dimensions
    expected_lats = Config.SA_N_LAT
    expected_lons = Config.SA_N_LON
    if n_lats != expected_lats or n_lons != expected_lons:
        logger.warning(f"⚠ Dimension mismatch!")
        logger.warning(f"  Expected: {expected_lats} lat × {expected_lons} lon")
        logger.warning(f"  Got: {n_lats} lat × {n_lons} lon")
    
    # Initialize array
    data = np.zeros((n_times, n_lats, n_lons, n_channels), dtype=np.float32)
    
    # Create coordinate mapping
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    time_to_idx = {time: i for i, time in enumerate(unique_times)}
    
    # Fill array
    logger.info("\nFilling 4D array...")
    for _, row in df.iterrows():
        t_idx = time_to_idx[row['time']]
        lat_idx = lat_to_idx[row['latitude']]
        lon_idx = lon_to_idx[row['longitude']]
        
        for ch_idx, var in enumerate(Config.SA_VARS):
            data[t_idx, lat_idx, lon_idx, ch_idx] = row[var]
    
    logger.info(f"✓ Array filled successfully")
    
    # Validate
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        logger.warning(f"⚠ Found {nan_count} NaN values in South Asia data")
    
    print_array_stats(data, "South Asia Data")
    
    dates = pd.DatetimeIndex(unique_times)
    
    return data, dates


def load_enso_index() -> pd.Series:
    """
    Load ENSO index (Niño 3.4) from CSV.
    
    Returns:
        series: Time series of ENSO index with datetime index
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING ENSO INDEX")
    logger.info("="*70)
    
    validate_file_exists(Config.ENSO_CSV, "ENSO CSV")
    
    # Load CSV
    logger.info("Reading ENSO CSV...")
    df = pd.read_csv(Config.ENSO_CSV)
    logger.info(f"  Rows: {len(df):,}")
    
    # Identify columns (first is time, second is value)
    time_col = df.columns[0]
    value_col = df.columns[1]
    
    logger.info(f"  Time column: '{time_col}'")
    logger.info(f"  Value column: '{value_col[:50]}...'")  # Truncate long name
    
    # Create series
    df[time_col] = pd.to_datetime(df[time_col])
    series = pd.Series(df[value_col].values, index=df[time_col])
    
    # Replace missing value markers (-9999, -99.99, etc.) with NaN
    series = series.replace([-9999.0, -99.99, -99.9, -9999], np.nan)
    
    # Drop NaN values
    n_missing = series.isna().sum()
    if n_missing > 0:
        logger.warning(f"⚠ Found {n_missing} missing values, dropping them")
        series = series.dropna()
    
    series = series.sort_index()
    
    # Filter to our time range (1980-2025)
    start_date = pd.Timestamp(f"{Config.START_YEAR}-01-01")
    end_date = pd.Timestamp(f"{Config.END_YEAR}-12-31")
    series = series[(series.index >= start_date) & (series.index <= end_date)]
    
    logger.info(f"\nENSO Index:")
    logger.info(f"  Time range: {series.index[0]} to {series.index[-1]}")
    logger.info(f"  Length: {len(series)} months")
    logger.info(f"  Min: {series.min():.3f}°C")
    logger.info(f"  Max: {series.max():.3f}°C")
    logger.info(f"  Mean: {series.mean():.3f}°C")
    logger.info(f"  Std: {series.std():.3f}°C")
    
    if len(series) == 0:
        raise ValueError("No ENSO data available in the specified time range!")
    
    return series


# ============================================================================
# TENSOR CONSTRUCTION FUNCTIONS
# ============================================================================

def create_sliding_windows(
    pacific_data: np.ndarray,
    sa_data: np.ndarray,
    enso_index: pd.Series,
    dates: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows and targets with lead time.
    
    Args:
        pacific_data: (T, lat, lon, channels) Pacific features
        sa_data: (T, lat, lon, channels) South Asia features
        enso_index: ENSO time series
        dates: Datetime index for data
    
    Returns:
        X_pacific: (N, 12, lat, lon, channels)
        X_sa: (N, 12, lat, lon, channels)
        y_enso: (N,)
        y_impact: (N, 2, lat*lon)
        sample_dates: (N,) - end date of each window
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING SLIDING WINDOWS")
    logger.info("="*70)
    
    T = len(dates)
    window_size = Config.WINDOW_SIZE
    lead_time = Config.LEAD_TIME
    
    # Calculate number of valid samples
    # We need:
    # - window_size months for input (t-11 to t)
    # - lead_time months for ENSO prediction (t+6)
    # - lead_time + 3 months for JJAS end (t+9 for September)
    max_offset = lead_time + 3  # Need up to t+9 for JJAS
    
    # Earliest window: starts at index 0 (1980-01 to 1980-12)
    # Latest window: ends at index T - max_offset - 1
    n_samples = T - window_size - max_offset + 1
    
    logger.info(f"\nWindow parameters:")
    logger.info(f"  Total time steps: {T}")
    logger.info(f"  Window size: {window_size} months")
    logger.info(f"  Lead time (ENSO): {lead_time} months")
    logger.info(f"  Max offset needed: {max_offset} months (for JJAS end)")
    logger.info(f"  Number of samples: {n_samples}")
    
    if n_samples <= 0:
        raise ValueError(f"Not enough data! Need at least {window_size + max_offset} months")
    
    # Initialize arrays
    pacific_shape = pacific_data.shape[1:]  # (lat, lon, channels)
    sa_shape = sa_data.shape[1:]
    
    X_pacific = np.zeros((n_samples, window_size, *pacific_shape), dtype=np.float32)
    X_sa = np.zeros((n_samples, window_size, *sa_shape), dtype=np.float32)
    y_enso = np.zeros(n_samples, dtype=np.float32)
    y_impact = np.zeros((n_samples, 2, sa_shape[0] * sa_shape[1]), dtype=np.float32)
    sample_dates = []
    
    logger.info("\nConstructing windows...")
    
    # Create windows
    for i in range(n_samples):
        # Window end index
        window_end = i + window_size - 1
        window_start = i
        
        # Extract 12-month window
        X_pacific[i] = pacific_data[window_start:window_end+1]
        X_sa[i] = sa_data[window_start:window_end+1]
        
        # Target ENSO at t+6
        enso_target_idx = window_end + lead_time
        enso_date = dates[enso_target_idx]
        
        # Get ENSO value
        try:
            y_enso[i] = enso_index.loc[enso_date]
        except KeyError:
            logger.warning(f"⚠ ENSO value missing for {enso_date}, using NaN")
            y_enso[i] = np.nan
        
        # JJAS impact: average over June-September starting at t+6
        # JJAS months: t+6 (June), t+7 (July), t+8 (Aug), t+9 (Sep)
        jjas_start_idx = window_end + lead_time
        jjas_end_idx = jjas_start_idx + 3  # 4 months total
        
        # Average temperature and precipitation over JJAS
        sa_jjas = sa_data[jjas_start_idx:jjas_end_idx+1]  # Shape: (4, lat, lon, channels)
        
        # Extract temperature (channel 0) and precipitation (channel 1)
        temp_jjas = sa_jjas[:, :, :, 0].mean(axis=0)  # (lat, lon)
        precip_jjas = sa_jjas[:, :, :, 1].mean(axis=0)  # (lat, lon)
        
        # Flatten and stack
        y_impact[i, 0, :] = precip_jjas.flatten()
        y_impact[i, 1, :] = temp_jjas.flatten()
        
        # Store window end date
        sample_dates.append(dates[window_end])
        
        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{n_samples} samples...")
    
    logger.info(f"✓ Created {n_samples} samples")
    
    # Convert dates to array
    sample_dates = np.array(sample_dates, dtype='datetime64[ns]')
    
    # Print sample info
    logger.info(f"\nSample date range:")
    logger.info(f"  First window ends: {sample_dates[0]}")
    logger.info(f"  Last window ends: {sample_dates[-1]}")
    logger.info(f"  → Predicting ENSO from {sample_dates[0] + np.timedelta64(6, 'M')} to {sample_dates[-1] + np.timedelta64(6, 'M')}")
    
    # Validate outputs
    print_array_stats(X_pacific, "X_pacific")
    print_array_stats(X_sa, "X_sa")
    print_array_stats(y_enso, "y_enso")
    print_array_stats(y_impact, "y_impact")
    
    # Check for NaNs
    if np.isnan(y_enso).any():
        nan_count = np.isnan(y_enso).sum()
        logger.error(f"✗ Found {nan_count} NaN values in y_enso!")
        raise ValueError("ENSO targets contain NaN values")
    
    return X_pacific, X_sa, y_enso, y_impact, sample_dates


# ============================================================================
# DATA SPLIT FUNCTIONS
# ============================================================================

def get_data_splits(sample_dates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split data into train/val/test sets based on dates.
    
    Args:
        sample_dates: Array of datetime64 dates
    
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    logger.info("\n" + "="*70)
    logger.info("SPLITTING DATA")
    logger.info("="*70)
    
    # Convert to pandas datetime for easier comparison
    dates_pd = pd.to_datetime(sample_dates)
    
    # Define split boundaries
    train_end = pd.Timestamp(Config.TRAIN_END)
    val_start = pd.Timestamp(Config.VAL_START)
    val_end = pd.Timestamp(Config.VAL_END)
    test_start = pd.Timestamp(Config.TEST_START)
    
    # Get indices
    train_idx = np.where(dates_pd <= train_end)[0]
    val_idx = np.where((dates_pd >= val_start) & (dates_pd <= val_end))[0]
    test_idx = np.where(dates_pd >= test_start)[0]
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train_idx)} samples ({dates_pd[train_idx[0]]} to {dates_pd[train_idx[-1]]})")
    logger.info(f"  Val:   {len(val_idx)} samples ({dates_pd[val_idx[0]]} to {dates_pd[val_idx[-1]]})")
    logger.info(f"  Test:  {len(test_idx)} samples ({dates_pd[test_idx[0]]} to {dates_pd[test_idx[-1]]})")
    logger.info(f"  Total: {len(train_idx) + len(val_idx) + len(test_idx)} samples")
    
    # Validate no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
    logger.info(f"✓ No overlap between splits")
    
    return {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }


# ============================================================================
# SAVING FUNCTIONS
# ============================================================================

def save_tensors(
    X_pacific: np.ndarray,
    X_sa: np.ndarray,
    y_enso: np.ndarray,
    y_impact: np.ndarray,
    dates: np.ndarray
):
    """Save all tensors to disk with verification."""
    logger.info("\n" + "="*70)
    logger.info("SAVING TENSORS")
    logger.info("="*70)
    
    tensors = {
        'X_pacific.npy': X_pacific,
        'X_sa.npy': X_sa,
        'y_enso.npy': y_enso,
        'y_impact.npy': y_impact,
        'dates.npy': dates
    }
    
    for filename, data in tensors.items():
        filepath = Config.OUTPUT_DIR / filename
        logger.info(f"\nSaving {filename}...")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Dtype: {data.dtype}")
        logger.info(f"  Size: {data.nbytes / 1024 / 1024:.2f} MB")
        
        np.save(filepath, data)
        
        # Verify by loading
        loaded = np.load(filepath)
        if not np.array_equal(loaded, data, equal_nan=True):
            raise ValueError(f"Verification failed for {filename}!")
        
        logger.info(f"  ✓ Saved and verified: {filepath}")
    
    logger.info(f"\n✓ All tensors saved to: {Config.OUTPUT_DIR}")


def save_metadata(
    sample_dates: np.ndarray,
    splits: Dict[str, np.ndarray],
    pacific_shape: tuple,
    sa_shape: tuple
):
    """Save metadata about the dataset."""
    logger.info("\n" + "="*70)
    logger.info("SAVING METADATA")
    logger.info("="*70)
    
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'n_samples': len(sample_dates),
        'date_range': {
            'first': str(sample_dates[0]),
            'last': str(sample_dates[-1])
        },
        'shapes': {
            'X_pacific': list(pacific_shape),
            'X_sa': list(sa_shape),
        },
        'splits': {
            'train': {
                'n_samples': len(splits['train']),
                'date_range': [str(sample_dates[splits['train'][0]]), 
                              str(sample_dates[splits['train'][-1]])]
            },
            'val': {
                'n_samples': len(splits['val']),
                'date_range': [str(sample_dates[splits['val'][0]]), 
                              str(sample_dates[splits['val'][-1]])]
            },
            'test': {
                'n_samples': len(splits['test']),
                'date_range': [str(sample_dates[splits['test'][0]]), 
                              str(sample_dates[splits['test'][-1]])]
            }
        },
        'parameters': {
            'window_size': Config.WINDOW_SIZE,
            'lead_time': Config.LEAD_TIME,
            'jjas_months': Config.JJAS_MONTHS
        }
    }
    
    import json
    metadata_file = Config.OUTPUT_DIR / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata saved to: {metadata_file}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution."""
    logger.info("\n" + "="*70)
    logger.info("DUAL-ENCODER CNN-TCN TENSOR CONSTRUCTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup
        create_output_directory()
        
        # Step 1: Load data
        pacific_data, pacific_dates = load_pacific_data()
        sa_data, sa_dates = load_south_asia_data()
        enso_index = load_enso_index()
        
        # Verify date alignment
        logger.info("\n" + "="*70)
        logger.info("VERIFYING DATE ALIGNMENT")
        logger.info("="*70)
        
        if not pacific_dates.equals(sa_dates):
            logger.error("✗ Pacific and South Asia dates don't match!")
            logger.error(f"  Pacific: {len(pacific_dates)} months")
            logger.error(f"  South Asia: {len(sa_dates)} months")
            raise ValueError("Date mismatch between datasets")
        
        logger.info(f"✓ Dates aligned: {len(pacific_dates)} months")
        logger.info(f"  Range: {pacific_dates[0]} to {pacific_dates[-1]}")
        
        # Step 2: Create sliding windows
        X_pacific, X_sa, y_enso, y_impact, sample_dates = create_sliding_windows(
            pacific_data, sa_data, enso_index, pacific_dates
        )
        
        # Step 3: Get data splits
        splits = get_data_splits(sample_dates)
        
        # Step 4: Save everything
        save_tensors(X_pacific, X_sa, y_enso, y_impact, sample_dates)
        save_metadata(sample_dates, splits, X_pacific.shape, X_sa.shape)
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"\nCreated tensors:")
        logger.info(f"  X_pacific: {X_pacific.shape}")
        logger.info(f"  X_sa: {X_sa.shape}")
        logger.info(f"  y_enso: {y_enso.shape}")
        logger.info(f"  y_impact: {y_impact.shape}")
        logger.info(f"  dates: {sample_dates.shape}")
        logger.info(f"\nData splits:")
        logger.info(f"  Train: {len(splits['train'])} samples")
        logger.info(f"  Val: {len(splits['val'])} samples")
        logger.info(f"  Test: {len(splits['test'])} samples")
        logger.info(f"\nOutput directory: {Config.OUTPUT_DIR}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"PIPELINE FAILED!")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
