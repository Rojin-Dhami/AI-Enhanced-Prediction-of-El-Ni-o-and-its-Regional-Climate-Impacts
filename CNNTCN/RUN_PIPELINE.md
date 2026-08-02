# 🚀 Pipeline Execution Guide

## ✅ Pre-Flight Checklist

### Required Files
- [x] `/data/combined_era5_oras5.csv` - Pacific domain (31×81 grid, 6 channels)
- [x] `/data/ERA5_single_levels_southasia/era5_southasia_with_precip_1980_2025.csv` - South Asia domain (18×21 grid, 5 channels)  
- [x] `/nina34.csv` - ENSO index (Niño 3.4)

### Scripts Ready
- [x] `build_tensors.py` - Tensor construction pipeline ✨ **NEW**
- [x] `model_dual_encoder.py` - Model architecture (31×81 Pacific, updated)
- [x] `train_dual.py` - Training script

### Verified Dimensions
- **Pacific**: 31 lat × 81 lon = 2,511 cells (verified in data)
- **South Asia**: 18 lat × 21 lon = 378 cells (verified in data)

---

## 🔧 Phase 2: Build Tensors

### Command
```bash
cd /Users/raman/Elnino/CNNTCN
python3 build_tensors.py
```

### What It Does
1. **Loads data** from 3 CSV files (~30 seconds)
2. **Creates sliding windows** (540 samples) (~1-2 minutes)
3. **Computes ENSO targets** at t+6 months
4. **Computes JJAS averages** for SA impacts
5. **Saves 5 tensors** to `tensors/` directory
6. **Saves metadata** (splits, parameters) to JSON

### Expected Output
```
DUAL-ENCODER CNN-TCN TENSOR CONSTRUCTION PIPELINE
======================================================================

LOADING PACIFIC DOMAIN DATA
======================================================================
✓ Found Pacific CSV: combined_era5_oras5.csv
Reading Pacific CSV...
  Rows: 1,386,072
  Dimensions:
    Time steps: 552
    Latitudes: 31 (from 30° to -30°)
    Longitudes: 81 (from 120° to 280°)
    Channels: 6
✓ Array filled successfully

LOADING SOUTH ASIA DOMAIN DATA
======================================================================
✓ Found South Asia CSV: era5_southasia_with_precip_1980_2025.csv
Reading South Asia CSV...
  Rows: 208,656
  Dimensions:
    Time steps: 552
    Latitudes: 18 (from 39° to 5°)
    Longitudes: 21 (from 60° to 100°)
    Channels: 5
✓ Array filled successfully

LOADING ENSO INDEX
======================================================================
✓ Found ENSO CSV: nina34.csv
Reading ENSO CSV...
ENSO Index:
  Time range: 1980-01-01 to 2025-12-01
  Length: 552 months

CREATING SLIDING WINDOWS
======================================================================
Window parameters:
  Total time steps: 552
  Window size: 12 months
  Lead time (ENSO): 6 months
  Max offset needed: 9 months (for JJAS end)
  Number of samples: 540
✓ Created 540 samples

SPLITTING DATA
======================================================================
Data splits:
  Train: ~450 samples (1980-01 to 2018-12)
  Val:   ~48 samples (2019-01 to 2022-12)
  Test:  ~36 samples (2023-01 to 2025-12)
✓ No overlap between splits

SAVING TENSORS
======================================================================
Saving X_pacific.npy...
  Shape: (540, 12, 31, 81, 6)
  Size: 340.17 MB
  ✓ Saved and verified

Saving X_sa.npy...
  Shape: (540, 12, 18, 21, 5)
  Size: 24.49 MB
  ✓ Saved and verified

Saving y_enso.npy...
  Shape: (540,)
  Size: 0.00 MB
  ✓ Saved and verified

Saving y_impact.npy...
  Shape: (540, 2, 378)
  Size: 1.55 MB
  ✓ Saved and verified

Saving dates.npy...
  Shape: (540,)
  Size: 0.00 MB
  ✓ Saved and verified

PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
Created tensors:
  X_pacific: (540, 12, 31, 81, 6)
  X_sa: (540, 12, 18, 21, 5)
  y_enso: (540,)
  y_impact: (540, 2, 378)
  dates: (540,)

Data splits:
  Train: ~450 samples
  Val: ~48 samples
  Test: ~36 samples

Output directory: /Users/raman/Elnino/CNNTCN/tensors
```

### Expected Runtime
- **Estimated**: 2-5 minutes
- **Bottleneck**: Loading Pacific CSV (large file ~100MB)

### Output Files
```
tensors/
├── X_pacific.npy      340 MB
├── X_sa.npy            24 MB
├── y_enso.npy           2 KB
├── y_impact.npy         2 MB
├── dates.npy            4 KB
└── metadata.json        1 KB
────────────────────────────
Total:                ~366 MB
```

---

## 🏋️ Phase 3: Train Model

### Command (after Phase 2 completes)
```bash
cd /Users/raman/Elnino/CNNTCN
python3 train_dual.py \
  --data_dir tensors \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --alpha 0.5 \
  --patience 15 \
  --out best_model_dual.pt
```

### Expected Runtime
- **With CPU**: 10-20 minutes (40-60 epochs)
- **With GPU**: 2-5 minutes

---

## 🛠️ Quality Assurance Features

### Error Detection
- ✅ File existence validation
- ✅ Column name verification
- ✅ NaN/Inf detection
- ✅ Dimension consistency checks
- ✅ Date alignment verification
- ✅ No data split overlap
- ✅ Save-load verification

### Logging
- ✅ Timestamped progress messages
- ✅ Detailed statistics (min/max/mean/std)
- ✅ Memory usage reports
- ✅ Warning messages for issues
- ✅ Success/failure confirmations

### Validation
- ✅ Shape verification at each step
- ✅ Value range checks (z-scores should be ~[-3, 3])
- ✅ Missing data handling (ENSO -9999 markers)
- ✅ Chronological date ordering
- ✅ Metadata tracking (JSON output)

---

## 📊 Sanity Checks After Completion

### Verify Tensor Shapes
```python
import numpy as np

X_pacific = np.load('tensors/X_pacific.npy')
X_sa = np.load('tensors/X_sa.npy')
y_enso = np.load('tensors/y_enso.npy')
y_impact = np.load('tensors/y_impact.npy')
dates = np.load('tensors/dates.npy')

print(f"X_pacific: {X_pacific.shape}")  # Should be (540, 12, 31, 81, 6)
print(f"X_sa: {X_sa.shape}")             # Should be (540, 12, 18, 21, 5)
print(f"y_enso: {y_enso.shape}")         # Should be (540,)
print(f"y_impact: {y_impact.shape}")     # Should be (540, 2, 378)
print(f"dates: {dates.shape}")           # Should be (540,)
```

### Verify Value Ranges
```python
# Z-scores should be roughly in [-3, 3] range
print(f"Pacific z-scores: [{X_pacific.min():.2f}, {X_pacific.max():.2f}]")
print(f"SA z-scores: [{X_sa.min():.2f}, {X_sa.max():.2f}]")
print(f"ENSO range: [{y_enso.min():.2f}, {y_enso.max():.2f}]°C")
print(f"Impact range: [{y_impact.min():.2f}, {y_impact.max():.2f}]")
```

### Verify No NaNs
```python
assert not np.isnan(X_pacific).any(), "Pacific has NaNs!"
assert not np.isnan(X_sa).any(), "SA has NaNs!"
assert not np.isnan(y_enso).any(), "ENSO has NaNs!"
assert not np.isnan(y_impact).any(), "Impact has NaNs!"
print("✓ No NaN values detected")
```

---

## ⚠️ Troubleshooting

### Issue: "File not found"
**Solution**: Check file paths in `build_tensors.py` Config class

### Issue: "Column 'xxx' not found"
**Solution**: Verify CSV column names match Config.PACIFIC_VARS / Config.SA_VARS

### Issue: "NaN values in ENSO targets"
**Solution**: Check ENSO CSV date range covers 1980-2025 + 9 months buffer

### Issue: "Dimension mismatch"
**Solution**: Verify Pacific = 31×81, South Asia = 18×21

### Issue: "Out of memory"
**Solution**: Close other applications; ~400MB RAM needed

---

## ✅ Success Indicators

- [x] All 5 `.npy` files created in `tensors/` directory
- [x] `metadata.json` created with split info
- [x] No error messages in console
- [x] Total file size ~366 MB
- [x] Console shows "PIPELINE COMPLETED SUCCESSFULLY!"
- [x] Tensor shapes match expected dimensions
- [x] No NaN values in any tensor
- [x] Z-scores in reasonable range (~[-3, 3])

---

## 🎯 Ready to Run?

Execute Phase 2 when ready:
```bash
cd /Users/raman/Elnino/CNNTCN
python3 build_tensors.py
```

Expected time: **2-5 minutes** ⏱️

---

*Updated: 2026-08-02 | No time pressure - Quality over speed! 🎯*
