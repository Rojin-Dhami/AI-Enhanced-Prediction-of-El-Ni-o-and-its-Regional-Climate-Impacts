"""
MIFS Feature Selection Module — Section 3.4.4

Mutual Information Feature Selection (Equations 3-8, 3-9 from proposal):

  Relevance (Eq 3-8):   I(X_ij; Y)
  MIFS Score (Eq 3-9):  S(X_ij) = I(X_ij; Y) - (1/|S|) * Σ I(X_ij; X_kl)
                         where S is the current selected feature set.

Pipeline:
  1. Compute relevance I(X_ij; Y) for all features (train only)
  2. Pre-filter to top N_CANDIDATES by relevance
  3. Greedy MIFS loop with redundancy penalty
  4. Sweep K → train XGBoost at each K → pick best K by val RMSE
  5. Save rankings and selected features
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output"

N_CANDIDATES = 300     # top features by raw relevance before greedy loop
MAX_K = 50             # maximum features to select in MIFS
MI_N_NEIGHBORS = 5     # k for Kraskov MI estimator
MI_RANDOM_STATE = 42


# ─── Step 1: Compute relevance I(X_ij; Y) ───────────────────────────────────

def compute_relevance(X_train, y_train, n_neighbors=MI_N_NEIGHBORS):
    """Compute I(X_ij; Y) for every feature using Kraskov k-NN estimator.
    
    Uses sklearn's mutual_info_regression which implements the Kraskov
    estimator (continuous-continuous MI, no discretization needed).
    
    Args:
        X_train: DataFrame or ndarray, (n_samples, n_features)
        y_train: array-like, (n_samples,)
    
    Returns:
        relevance: Series indexed by feature name, sorted descending.
    """
    print(f"[MIFS-1] Computing relevance I(X_ij; Y) for {X_train.shape[1]:,} features...")
    print(f"         Using {X_train.shape[0]} training samples, k={n_neighbors}")

    X_arr = X_train.values if hasattr(X_train, 'values') else X_train
    y_arr = np.asarray(y_train).ravel()

    # Replace any remaining NaN with 0 for MI computation
    if np.isnan(X_arr).any():
        n_nan = np.isnan(X_arr).sum()
        print(f"         Warning: {n_nan} NaN values found, replacing with 0")
        X_arr = np.nan_to_num(X_arr, nan=0.0)

    mi_scores = mutual_info_regression(
        X_arr, y_arr,
        n_neighbors=n_neighbors,
        random_state=MI_RANDOM_STATE,
    )

    feature_names = (X_train.columns.tolist() if hasattr(X_train, 'columns')
                     else [f"f{i}" for i in range(X_arr.shape[1])])
    relevance = pd.Series(mi_scores, index=feature_names, name="relevance")
    relevance.sort_values(ascending=False, inplace=True)

    print(f"         Top 10 by relevance:")
    for name, score in relevance.head(10).items():
        print(f"           {score:.4f}  {name}")

    return relevance


# ─── Step 2: Pre-filter top candidates ───────────────────────────────────────

def prefilter_candidates(relevance, n_candidates=N_CANDIDATES):
    """Take top N_CANDIDATES by raw relevance for the greedy MIFS loop.
    
    Flags if any variable type is entirely excluded from the candidate pool.
    """
    print(f"\n[MIFS-2] Pre-filtering to top {n_candidates} by relevance...")
    top = relevance.head(n_candidates)

    # Extract variable types from column names (e.g., 'sst_anom_z' from 'sst_anom_z_(5,160)_lag3')
    all_var_types = set(relevance.index.map(_extract_var_type))
    top_var_types = set(top.index.map(_extract_var_type))
    missing = all_var_types - top_var_types

    if missing:
        print(f"         ⚠ WARNING: These variable types are entirely excluded "
              f"from the top-{n_candidates} candidate pool:")
        for v in sorted(missing):
            best_rank = (relevance.index.map(_extract_var_type) == v).argmax()
            best_score = relevance.iloc[best_rank]
            print(f"           {v} (best rank: #{best_rank+1}, MI={best_score:.4f})")
        print(f"         This is a known limitation — these variables may be weak "
              f"predictors of Niño 3.4 at 6-month lead.")
    else:
        print(f"         All variable types represented in candidate pool.")

    # Report variable-type distribution in candidates
    var_counts = pd.Series(list(top.index.map(_extract_var_type))).value_counts()
    print(f"         Candidate pool by variable type:")
    for var, count in var_counts.items():
        print(f"           {var}: {count}")

    return top


def _extract_var_type(col_name):
    """Extract variable name from windowed column name.
    e.g., 'sst_anom_z_(-2.0,120.0)_lag3' → 'sst_anom_z'
    """
    # Split on '_(' to get everything before the coordinate
    parts = col_name.split("_(")
    return parts[0] if len(parts) > 1 else col_name


# ─── Step 3: Greedy MIFS loop ────────────────────────────────────────────────

def greedy_mifs(X_train, y_train, relevance_top, max_k=MAX_K,
                n_neighbors=MI_N_NEIGHBORS):
    """Greedy MIFS selection with redundancy penalty (Eq 3-9).
    
    S(X_ij) = I(X_ij; Y) - (1/|S|) * Σ_{X_kl ∈ S} I(X_ij; X_kl)
    
    Caches pairwise MI to avoid recomputation.
    
    Args:
        X_train: full training DataFrame (will be sliced to candidate columns)
        y_train: training target
        relevance_top: Series of top candidates with their relevance scores
        max_k: maximum number of features to select
    
    Returns:
        selected_order: list of (feature_name, mifs_score) in selection order
    """
    print(f"\n[MIFS-3] Greedy MIFS loop (max K={max_k})...")
    candidate_names = list(relevance_top.index)
    X_cand = X_train[candidate_names].values.copy()
    X_cand = np.nan_to_num(X_cand, nan=0.0).astype(np.float32)

    n_cand = len(candidate_names)
    rel = relevance_top.values.copy()  # relevance scores

    # Cache: pairwise MI between candidates, computed lazily
    # mi_cache[i, j] = I(X_i; X_j), filled on demand
    mi_cache = np.full((n_cand, n_cand), np.nan, dtype=np.float32)

    selected_indices = []
    selected_order = []
    remaining = set(range(n_cand))

    for step in range(min(max_k, n_cand)):
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            if len(selected_indices) == 0:
                score = rel[idx]
            else:
                # Compute redundancy: (1/|S|) * Σ I(X_idx; X_sel)
                redundancy = 0.0
                for sel_idx in selected_indices:
                    # Check cache
                    if np.isnan(mi_cache[idx, sel_idx]):
                        mi_val = mutual_info_regression(
                            X_cand[:, idx].reshape(-1, 1),
                            X_cand[:, sel_idx],
                            n_neighbors=n_neighbors,
                            random_state=MI_RANDOM_STATE,
                        )[0]
                        mi_cache[idx, sel_idx] = mi_val
                        mi_cache[sel_idx, idx] = mi_val
                    redundancy += mi_cache[idx, sel_idx]
                redundancy /= len(selected_indices)
                score = rel[idx] - redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        selected_order.append((candidate_names[best_idx], float(best_score)))

        if (step + 1) % 5 == 0 or step < 5:
            print(f"         Step {step+1:3d}: selected '{candidate_names[best_idx]}' "
                  f"(score={best_score:.4f})")

    print(f"         MIFS selected {len(selected_order)} features.")
    return selected_order


# ─── Step 4: K-sweep with XGBoost ────────────────────────────────────────────

def sweep_k_xgboost(X_train, y_train, X_val, y_val, selected_order,
                     k_step=2, max_k=MAX_K):
    """Train XGBoost at each K, evaluate validation RMSE, pick best K.
    
    Args:
        selected_order: list of (feature_name, score) from greedy_mifs
        k_step: step size for K sweep
    
    Returns:
        best_k: optimal number of features
        k_results: list of (K, val_rmse) tuples
    """
    print(f"\n[MIFS-4] K-sweep with XGBoost (step={k_step}, max={max_k})...")
    feature_names_ordered = [name for name, _ in selected_order]
    k_values = list(range(k_step, min(max_k, len(feature_names_ordered)) + 1, k_step))
    # Always include k=1
    if 1 not in k_values:
        k_values = [1] + k_values

    k_results = []
    best_rmse = np.inf
    best_k = k_values[0]

    for k in k_values:
        top_k_features = feature_names_ordered[:k]
        Xtr_k = X_train[top_k_features].values
        Xval_k = X_val[top_k_features].values

        # Replace NaN if any
        Xtr_k = np.nan_to_num(Xtr_k, nan=0.0)
        Xval_k = np.nan_to_num(Xval_k, nan=0.0)

        if HAS_XGBOOST:
            model = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                random_state=42, verbosity=0,
            )
        else:
            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=3, learning_rate=0.1,
                random_state=42,
            )
        model.fit(Xtr_k, y_train)
        y_pred = model.predict(Xval_k)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        k_results.append((k, rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k

        print(f"         K={k:3d}  val RMSE={rmse:.4f}"
              f"{'  ← best' if rmse == best_rmse else ''}")

    print(f"\n         Best K = {best_k} (val RMSE = {best_rmse:.4f})")
    return best_k, k_results


# ─── Step 5: Save results ────────────────────────────────────────────────────

def save_mifs_results(relevance, selected_order, best_k, k_results,
                      output_dir=OUTPUT_DIR):
    """Save full ranking, selected features, and K-sweep results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full relevance ranking
    relevance.to_csv(output_dir / "mifs_relevance_ranking.csv", header=True)
    print(f"\n[MIFS-5] Saved relevance ranking ({len(relevance)} features)")

    # MIFS-selected features in order
    sel_df = pd.DataFrame(selected_order, columns=["feature", "mifs_score"])
    sel_df.index.name = "rank"
    sel_df.to_csv(output_dir / "mifs_selected_features.csv")
    print(f"         Saved MIFS selected features ({len(selected_order)} features)")

    # K-sweep results
    k_df = pd.DataFrame(k_results, columns=["K", "val_rmse"])
    k_df.to_csv(output_dir / "mifs_k_sweep.csv", index=False)
    print(f"         Saved K-sweep results ({len(k_results)} evaluations)")

    # Best K and final feature list
    final_features = [name for name, _ in selected_order[:best_k]]
    result = {
        "best_k": best_k,
        "best_val_rmse": float(k_df.loc[k_df["K"] == best_k, "val_rmse"].iloc[0]),
        "selected_features": final_features,
    }
    with open(output_dir / "mifs_best_k.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"         Saved best K={best_k} and final feature list")

    return final_features


# ─── Step 6: Physical sanity check ───────────────────────────────────────────

def physical_sanity_check(selected_features):
    """Report which variable types and approximate regions dominate the
    selected set. Compare against proposal expectations:
      - Central/eastern Pacific SST
      - Walker Circulation wind stress
      - Western Pacific thermocline OHC
    """
    print(f"\n[MIFS-6] Physical sanity check on {len(selected_features)} selected features:")

    var_types = [_extract_var_type(f) for f in selected_features]
    lag_values = []
    lat_values = []
    lon_values = []
    for f in selected_features:
        # Parse lag
        lag_part = f.rsplit("_lag", 1)
        if len(lag_part) == 2:
            lag_values.append(int(lag_part[1]))
        # Parse coordinates from _(lat,lon)
        try:
            coord_part = f.split("_(")[1].split(")")[0]
            lat_str, lon_str = coord_part.split(",")
            lat_values.append(float(lat_str))
            lon_values.append(float(lon_str))
        except (IndexError, ValueError):
            pass

    # Variable distribution
    var_counts = pd.Series(var_types).value_counts()
    print(f"\n    Variable type distribution:")
    for var, count in var_counts.items():
        print(f"      {var}: {count}")

    # Lag distribution
    if lag_values:
        lag_counts = pd.Series(lag_values).value_counts().sort_index()
        print(f"\n    Lag distribution (0=most recent, 11=oldest):")
        for lag, count in lag_counts.items():
            print(f"      lag{lag}: {count}")

    # Spatial distribution
    if lat_values and lon_values:
        lat_arr = np.array(lat_values)
        lon_arr = np.array(lon_values)
        print(f"\n    Spatial extent:")
        print(f"      Latitude:  {lat_arr.min():.0f}° to {lat_arr.max():.0f}°")
        print(f"      Longitude: {lon_arr.min():.0f}° to {lon_arr.max():.0f}°")

        # Check for ENSO-relevant regions
        # Niño 3.4 region: 5°N–5°S, 170°W–120°W (190°E–240°E)
        nino34_mask = (np.abs(lat_arr) <= 5) & (lon_arr >= 190) & (lon_arr <= 240)
        n_nino34 = nino34_mask.sum()
        # Western Pacific warm pool: ~120°E–160°E (120–160)
        wp_mask = (np.abs(lat_arr) <= 10) & (lon_arr >= 120) & (lon_arr <= 160)
        n_wp = wp_mask.sum()
        print(f"\n    ENSO-relevant regions:")
        print(f"      Niño 3.4 box (5°N-5°S, 190°-240°E): {n_nino34} features")
        print(f"      Western Pacific (10°N-10°S, 120°-160°E): {n_wp} features")

    # Comparison with proposal expectations
    print(f"\n    Proposal expectation check:")
    has_sst = any("sst_anom_z" in v for v in var_types)
    has_wind = any("iews" in v or "inss" in v for v in var_types)
    has_ohc = any("sohtc300" in v or "so20chgt" in v for v in var_types)
    has_slp = any("msl_anom_z" in v for v in var_types)
    print(f"      SST present:        {'✓' if has_sst else '✗'}")
    print(f"      Wind stress present: {'✓' if has_wind else '✗'}")
    print(f"      OHC present:        {'✓' if has_ohc else '✗'}")
    print(f"      SLP present:        {'✓' if has_slp else '✗'}")


# ─── Main: run full MIFS pipeline ────────────────────────────────────────────

def run_mifs(splits):
    """Execute the complete MIFS pipeline (Step 2 from the .md).
    
    Args:
        splits: dict from data_pipeline.run_pipeline() with keys
                'train', 'val', 'test', each containing (X, y, times).
    
    Returns:
        final_features: list of best-K feature names
        best_k: optimal K
    """
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]

    # 1. Relevance
    relevance = compute_relevance(X_train, y_train)

    # 2. Pre-filter
    top_relevance = prefilter_candidates(relevance, n_candidates=N_CANDIDATES)

    # 3. Greedy MIFS
    selected_order = greedy_mifs(X_train, y_train, top_relevance, max_k=MAX_K)

    # 4. K-sweep
    best_k, k_results = sweep_k_xgboost(
        X_train, y_train, X_val, y_val, selected_order
    )

    # 5. Save
    final_features = save_mifs_results(
        relevance, selected_order, best_k, k_results
    )

    # 6. Sanity check
    physical_sanity_check(final_features)

    return final_features, best_k


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import run_pipeline

    print("=" * 70)
    print("STEP 1: Data Pipeline")
    print("=" * 70)
    splits, _ = run_pipeline(save=True)

    print("\n" + "=" * 70)
    print("STEP 2: MIFS Feature Selection")
    print("=" * 70)
    final_features, best_k = run_mifs(splits)

    print("\n" + "=" * 70)
    print(f"DONE — Selected {best_k} features via MIFS")
    print("=" * 70)
