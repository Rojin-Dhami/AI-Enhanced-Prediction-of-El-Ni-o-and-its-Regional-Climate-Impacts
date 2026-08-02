# 🌊 Dual-Encoder CNN-TCN ENSO Forecasting Pipeline

**Complete workflow from raw data to forecasts**

---

## 📋 **WORKFLOW OVERVIEW**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: DATA PREPARATION                     │
│                           ✅ COMPLETED                               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: TENSOR CONSTRUCTION                     │
│                           ⏳ NEXT STEP                               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: MODEL TRAINING                       │
│                           ⏸️  PENDING                                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: EVALUATION                          │
│                           ⏸️  PENDING                                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 5: FORECASTING                          │
│                           ⏸️  PENDING                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 **PHASE 1: DATA PREPARATION** ✅

### Status: **COMPLETED**

### 1.1 Pacific Domain Data
**Location:** `/Users/raman/Elnino/data/combined_era5_oras5.csv`

**Variables (6 channels):**
- `sst_anom_z` - Sea surface temperature anomaly (ERA5)
- `msl_anom_z` - Mean sea level pressure anomaly (ERA5)
- `avg_iews_anom_z` - East-West wind stress anomaly (ERA5)
- `avg_inss_anom_z` - North-South wind stress anomaly (ERA5)
- `sohtc300_anom_z` - Ocean heat content 0-300m (ORAS5)
- `so20chgt_anom_z` - 20°C isotherm depth / thermocline depth (ORAS5)

**Domain:** 30°S-30°N, 120°E-80°W, 2°×2° grid (30 lat × 100 lon = 3,000 cells)

**Time Range:** 1980-01 to 2025-12 (552 months)

---

### 1.2 South Asia Domain Data
**Location:** `/Users/raman/Elnino/data/ERA5_single_levels_southasia/era5_southasia_with_precip_1980_2025.csv`

**Variables (5 channels):**
- `t2m_anom_z` - 2-meter temperature anomaly
- `tp_anom_z` - Total precipitation anomaly
- `msl_anom_z` - Mean sea level pressure anomaly
- `avg_iews_anom_z` - East-West wind stress anomaly
- `avg_inss_anom_z` - North-South wind stress anomaly

**Domain:** 5°N-40°N, 60°E-100°E, 2°×2° grid (18 lat × 21 lon = 378 cells)

**Time Range:** 1980-01 to 2025-12 (552 months)

**Data Quality:** ✅ Verified (0 nulls, zero precipitation values legitimate)

---

### 1.3 ENSO Target Data
**Location:** `/Users/raman/Elnino/nina34.csv` or `/Users/raman/Elnino/meiv2.csv`

**Variable:** Niño 3.4 index (monthly SST anomaly)

**Time Range:** 1980-01 to 2025-12

---

### 1.4 Data Processing Applied
✅ **Climatological anomalies:** Subtract monthly mean per grid cell (removes seasonal cycle)  
✅ **Z-score normalization:** Using **training period only** statistics (1980-2018)  
✅ **Order:** Raw → Anomaly → Z-score  
✅ **No data leakage:** Test set statistics not used in normalization

---

## 🔨 **PHASE 2: TENSOR CONSTRUCTION** ⏳

### Status: **NEXT STEP - TO IMPLEMENT**

### 2.1 Goal
Transform CSV files into 5D tensors for model training.

### 2.2 Input Files
```
/Users/raman/Elnino/data/combined_era5_oras5.csv          # Pacific data
/Users/raman/Elnino/data/ERA5_single_levels_southasia/
    era5_southasia_with_precip_1980_2025.csv              # South Asia data
/Users/raman/Elnino/nina34.csv                            # ENSO target
```

### 2.3 Output Tensors
```
/Users/raman/Elnino/CNNTCN/tensors/X_pacific.npy         # Shape: (N, 12, 30, 100, 6)
/Users/raman/Elnino/CNNTCN/tensors/X_sa.npy              # Shape: (N, 12, 18, 21, 5)
/Users/raman/Elnino/CNNTCN/tensors/y_enso.npy            # Shape: (N,)
/Users/raman/Elnino/CNNTCN/tensors/y_impact.npy          # Shape: (N, 2, 378)
/Users/raman/Elnino/CNNTCN/tensors/dates.npy             # Shape: (N,)
```

**Where N ≈ 540** (552 months - 12 window)

### 2.4 Processing Steps

#### Step 2.4.1: Create Sliding Windows
```python
# For each time step t:
# X_pacific[i] = Pacific data from (t-11) to t  (12 consecutive months)
# X_sa[i] = South Asia data from (t-11) to t
# y_enso[i] = Niño 3.4 at (t+6)  ← 6-month lead time
# y_impact[i] = JJAS average of temp & precip at (t+6, t+7, t+8, t+9)
```

**Example:**
- Window ending Jan 1985 → predicts Jul 1985 ENSO, JJAS 1985 impacts
- Window ending Dec 1990 → predicts Jun 1991 ENSO, JJAS 1991 impacts

#### Step 2.4.2: ENSO Target
```python
# Direct lookup 6 months ahead
y_enso[i] = nina34_index[t + 6]
```

#### Step 2.4.3: South Asia Impact Target
```python
# Average temperature and precipitation over JJAS season
# JJAS = June, July, August, September
# Starting 6 months after window end

# Get months t+6, t+7, t+8, t+9
temp_jjas = average(t2m_anom_z for months [t+6:t+10])  # Shape: (18, 21)
precip_jjas = average(tp_anom_z for months [t+6:t+10]) # Shape: (18, 21)

# Stack into (2, 378) tensor
y_impact[i, 0, :] = precip_jjas.flatten()  # Channel 0: precipitation
y_impact[i, 1, :] = temp_jjas.flatten()    # Channel 1: temperature
```

#### Step 2.4.4: Valid Sample Range
```python
# Earliest window: 1980-01 to 1980-12 → predicts 1981-06 ENSO, JJAS 1981
# Latest window: 2024-03 to 2025-02 → predicts 2025-08 ENSO, JJAS 2025
#                (need t+6 for ENSO, t+9 for JJAS end)

# Total samples N ≈ 540
```

### 2.5 Script to Create
**File:** `CNNTCN/build_tensors.py`

**Tasks:**
1. Load Pacific CSV (combined_era5_oras5.csv)
2. Load South Asia CSV (era5_southasia_with_precip_1980_2025.csv)
3. Load ENSO index (nina34.csv)
4. Pivot CSVs to 4D arrays (time, lat, lon, channels)
5. Create sliding windows
6. Apply 6-month lead time
7. Compute JJAS seasonal averages for impacts
8. Save all tensors as .npy files
9. Print data split statistics

**Estimated Runtime:** 2-5 minutes

---

## 🏋️ **PHASE 3: MODEL TRAINING** ⏸️

### Status: **PENDING (after tensor construction)**

### 3.1 Goal
Train dual-encoder CNN-TCN model to predict ENSO index and South Asia impacts.

### 3.2 Input
```
/Users/raman/Elnino/CNNTCN/tensors/*.npy
```

### 3.3 Training Script
**File:** `CNNTCN/train_dual.py` (already created ✅)

**Command:**
```bash
cd /Users/raman/Elnino/CNNTCN
python train_dual.py \
  --data_dir tensors \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --alpha 0.5 \
  --patience 15 \
  --out best_model_dual.pt
```

### 3.4 Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Epochs | 100 | Maximum iterations |
| Batch size | 16 | Good for ~450 samples |
| Learning rate | 0.001 | Conservative start |
| Alpha | 0.5 | Equal weight ENSO/impact loss |
| Patience | 15 | Early stopping threshold |
| Optimizer | Adam | Adaptive learning rate |
| LR Scheduler | ReduceLROnPlateau | Halves LR after plateau |
| Gradient clipping | 5.0 | Prevents exploding gradients |

### 3.5 Data Split
```
Train:      1980-2018  (~450 samples, 83%)
Validation: 2019-2022  (~48 samples,   9%)
Test:       2023-2025  (~36 samples,   8%)
```

### 3.6 Expected Outcomes
- **Training time:** 10-20 minutes (CPU) or 2-5 minutes (GPU)
- **Early stopping:** Likely at epoch 40-60
- **Model file:** `best_model_dual.pt` (~1 MB)
- **Logs:** Training/validation loss curves saved

### 3.7 Success Criteria
- Validation loss decreases to < 1.0
- No severe overfitting (train loss ≪ val loss)
- Model converges (loss stabilizes)

---

## 📊 **PHASE 4: EVALUATION** ⏸️

### Status: **PENDING (after training)**

### 4.1 Goal
Evaluate trained model on held-out test set (2023-2025).

### 4.2 Evaluation Script
**File:** `CNNTCN/evaluate.py` (to create)

**Command:**
```bash
python evaluate.py \
  --data_dir tensors \
  --model best_model_dual.pt \
  --out evaluation_results
```

### 4.3 Metrics

#### ENSO Prediction (scalar)
- **RMSE:** Root Mean Squared Error (target: < 0.8°C)
- **ACC:** Anomaly Correlation Coefficient (target: > 0.6)
- **SRMSE:** Standardized RMSE (RMSE / std_dev)

#### South Asia Impact (spatial)
- **Spatial RMSE:** Per-grid-cell error
- **Pattern correlation:** Spatial correlation with ground truth
- **Regional MAE:** Mean absolute error by region (Western/Central/Eastern India, Pakistan, Bangladesh)

### 4.4 Visualizations to Generate
1. **ENSO time series:** Predicted vs. actual Niño 3.4 (2023-2025)
2. **Spatial impact maps:** Predicted vs. actual temperature/precipitation anomalies
3. **Scatter plots:** Predicted vs. actual with R² values
4. **Error distributions:** Histograms of prediction errors
5. **Spatial error maps:** Where the model performs best/worst

### 4.5 Benchmark Comparison
Compare against:
- **Persistence model:** Use current month's value
- **Climatology:** Seasonal mean
- **Linear regression baseline:** From previous XGBoost work
- **Published benchmarks:** Literature values for 6-month lead ENSO prediction

### 4.6 Expected Performance
Based on literature (Ham et al. 2019, Cachay et al. 2021):
- ENSO ACC @ 6-month lead: **0.55-0.70** (good: > 0.60)
- ENSO RMSE: **0.7-0.9°C**
- Impact spatial correlation: **0.40-0.60**

---

## 🔮 **PHASE 5: FORECASTING** ⏸️

### Status: **PENDING (after evaluation)**

### 5.1 Goal
Generate operational forecasts for 2026-2027 using latest available data.

### 5.2 Forecast Script
**File:** `CNNTCN/forecast.py` (to create)

**Command:**
```bash
python forecast.py \
  --data_dir tensors \
  --model best_model_dual.pt \
  --forecast_months 12 \
  --out forecasts_2026
```

### 5.3 Forecast Strategy

#### Real-time Prediction (Current: Aug 2026)
```
Latest available data: Jan 2026 - Dec 2025 (12-month window)
                       ↓
Window: 2025-01 to 2025-12
                       ↓
Predict: Jul 2026 ENSO, JJAS 2026 SA impacts
```

#### Iterative Multi-Step Forecasting
```
Step 1: Use window [2025-01 to 2025-12] → predict Feb 2027 ENSO
Step 2: Use window [2025-02 to 2026-01] → predict Mar 2027 ENSO
...
Step N: Generate forecasts up to 12 months ahead
```

**Note:** Accuracy degrades with longer lead times (6-month target is sweet spot).

### 5.4 Outputs

#### 5.4.1 ENSO Forecast Table
```
Month         Predicted Niño 3.4    Category      Confidence
--------      ------------------    ----------    -----------
Jul 2026      +0.8°C                Weak El Niño  Medium
Aug 2026      +1.1°C                Moderate      High
Sep 2026      +1.3°C                Moderate      High
Oct 2026      +1.2°C                Moderate      Medium
Nov 2026      +0.9°C                Weak          Low
Dec 2026      +0.5°C                Neutral       Very Low
```

**ENSO Categories:**
- Strong El Niño: ≥ +2.0°C
- Moderate El Niño: +1.5 to +2.0°C
- Weak El Niño: +0.5 to +1.5°C
- Neutral: -0.5 to +0.5°C
- Weak La Niña: -1.5 to -0.5°C
- Moderate La Niña: -2.0 to -1.5°C
- Strong La Niña: ≤ -2.0°C

#### 5.4.2 South Asia Impact Maps
Generate spatial maps showing:
- **Precipitation anomaly** (mm/month)
- **Temperature anomaly** (°C)
- **Regional summaries** (Western/Central/Eastern India, Pakistan, Bangladesh)

#### 5.4.3 Uncertainty Quantification
- Use ensemble methods (dropout during inference, bootstrap)
- Provide confidence intervals (±1σ, ±2σ)
- Flag low-confidence predictions (e.g., forecasts > 6 months)

### 5.5 Operational Use
1. Update forecasts monthly as new data arrives
2. Compare predictions with persistence/climatology baselines
3. Archive forecasts for verification
4. Generate reports for stakeholders (agriculture, water management, disaster preparedness)

---

## 📂 **FILE STRUCTURE**

```
/Users/raman/Elnino/
├── data/
│   ├── combined_era5_oras5.csv                    ✅ Pacific data (6 channels)
│   ├── ERA5_single_levels_southasia/
│   │   └── era5_southasia_with_precip_1980_2025.csv  ✅ South Asia (5 channels)
│   ├── nina34.csv                                 ✅ ENSO index
│   └── meiv2.csv                                  ✅ Alternative ENSO index
│
├── CNNTCN/
│   ├── model_dual_encoder.py                      ✅ Model architecture (261K params)
│   ├── train_dual.py                              ✅ Training script
│   ├── build_tensors.py                           ⏳ TO CREATE (Phase 2)
│   ├── evaluate.py                                ⏸️  TO CREATE (Phase 4)
│   ├── forecast.py                                ⏸️  TO CREATE (Phase 5)
│   ├── README_DUAL_ENCODER.md                     ✅ Documentation
│   ├── WORKFLOW.md                                ✅ This file
│   │
│   ├── tensors/                                   ⏳ TO CREATE (Phase 2 output)
│   │   ├── X_pacific.npy
│   │   ├── X_sa.npy
│   │   ├── y_enso.npy
│   │   ├── y_impact.npy
│   │   └── dates.npy
│   │
│   ├── outputs/                                   ⏸️  TO CREATE (Phase 3-5 output)
│   │   ├── best_model_dual.pt                     # Trained model
│   │   ├── training_history.csv                   # Loss curves
│   │   ├── evaluation_results/                    # Test set results
│   │   └── forecasts_2026/                        # 2026-2027 forecasts
│   │
│   └── figures/                                   ⏸️  TO CREATE (visualizations)
│       ├── training_curves.png
│       ├── test_timeseries.png
│       ├── spatial_impacts_*.png
│       └── forecast_maps_*.png
│
└── scripts/
    ├── download_era5_precipitation_southasia.py   ✅ Downloaded precipitation
    └── merge_precipitation_southasia.py           ✅ Merged & processed data
```

---

## ⏱️ **ESTIMATED TIMELINE**

| Phase | Task | Time | Status |
|-------|------|------|--------|
| **1** | Data preparation | 3-4 hours | ✅ **DONE** |
| **2** | Tensor construction script | 30-45 min | ⏳ **NEXT** |
| **2** | Run tensor construction | 2-5 min | ⏳ **NEXT** |
| **3** | Model training | 10-20 min | ⏸️  Pending |
| **4** | Evaluation script | 20-30 min | ⏸️  Pending |
| **4** | Run evaluation | 2-5 min | ⏸️  Pending |
| **5** | Forecast script | 15-20 min | ⏸️  Pending |
| **5** | Generate forecasts | 1-2 min | ⏸️  Pending |
| **5** | Create visualizations | 10-15 min | ⏸️  Pending |
| | **TOTAL** | **~2-3 hours remaining** | |

---

## 🚀 **NEXT IMMEDIATE ACTION**

### Create `build_tensors.py` script

This script will:
1. Load Pacific and South Asia CSV files
2. Load ENSO target data (nina34.csv)
3. Pivot data into 4D arrays (time × lat × lon × channels)
4. Create 12-month sliding windows
5. Apply 6-month lead time for targets
6. Compute JJAS seasonal averages for SA impacts
7. Save tensors as .npy files
8. Print statistics and verify shapes

**Expected output:**
```
Loaded Pacific data: 552 months × 3000 cells × 6 channels
Loaded South Asia data: 552 months × 378 cells × 5 channels
Loaded ENSO index: 552 months

Created 540 training samples:
  X_pacific: (540, 12, 30, 100, 6)
  X_sa: (540, 12, 18, 21, 5)
  y_enso: (540,)
  y_impact: (540, 2, 378)
  dates: (540,)

Saved tensors to: /Users/raman/Elnino/CNNTCN/tensors/

Data splits:
  Train: 450 samples (1980-2018)
  Val: 48 samples (2019-2022)
  Test: 36 samples (2023-2025)
```

---

## 🎯 **SUCCESS CRITERIA FOR COMPLETE PIPELINE**

### Minimum Viable Product (MVP)
✅ Dual-encoder model trains without errors  
✅ Validation loss < training loss (no severe overfitting)  
✅ Test ENSO ACC > 0.50 (better than persistence)  
✅ Generates 2026 forecasts successfully  

### Stretch Goals
🎯 Test ENSO ACC > 0.60 (competitive with state-of-the-art)  
🎯 Test ENSO RMSE < 0.8°C  
🎯 South Asia impact spatial correlation > 0.45  
🎯 Outperforms linear regression baseline  

---

## 📞 **READY TO PROCEED?**

**Next step:** Create `build_tensors.py` script and run tensor construction.

Estimated time to complete: **35-50 minutes** (script creation + execution)

---

*Last Updated: 2026-08-02*
