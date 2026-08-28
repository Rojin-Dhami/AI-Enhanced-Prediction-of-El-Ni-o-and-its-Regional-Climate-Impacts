"""
El Nino Forecast Dashboard
==========================
Interactive Streamlit front-end for the CNN-TCN multi-task ENSO / South Asia
forecast. It only *reads* the artifacts written by the notebook's Section 9 into
`forecast_outputs/` -- it never loads PyTorch or retrains, so it starts instantly.

Run from the project root (Main/):
    streamlit run dashboard/app.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Paths (dashboard/ lives inside the project root) ---
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "forecast_outputs"
META = OUT / "forecast_meta.json"
CSV  = OUT / "ensemble_oni_2026.csv"
NPZ  = OUT / "ensemble_maps_2026.npz"
HIST_ONI_CSV = ROOT / "data" / "Combined" / "nino34_index_3month_running_mean_official_oni.csv"

st.set_page_config(page_title="El Nino Forecast", page_icon="🌊", layout="wide")

# --- Muted, low-saturation palette (kept small on purpose: informative, not "rainbow") ---
ACCENT      = "#4C8BB4"   # forecast line / primary accent (muted steel blue)
NEUTRAL_GRY = "#8a8f98"   # historical line / neutral text
TEMP_SCALE  = [[0, "#3b6fa0"], [0.5, "#f2f0ea"], [1, "#b3562f"]]   # muted blue -> sand -> muted orange
PRECIP_SCALE = [[0, "#a1752f"], [0.5, "#f2f0ea"], [1, "#3f7a5c"]]  # muted brown -> sand -> muted green
SPREAD_SCALE = [[0, "#f2f0ea"], [1, "#5b6470"]]                    # sand -> slate (single-hue = uncertainty)

# Approximate rectangular bounding boxes (lat_min, lat_max, lon_min, lon_max) — for
# demo-level, coarse-grid regional averaging only, NOT real political borders.
COUNTRY_BOXES = {
    "Nepal":      (26.0, 31.0, 80.0, 89.0),
    "India":      (8.0, 28.0, 68.0, 88.0),   # core/central-southern box to limit overlap
    "Bangladesh": (21.0, 27.0, 88.0, 93.0),
}

ENSO_STAGES = [   # (threshold, label, muted color) — checked high to low
    (2.0,  "Very Strong El Nino", "#8c2d04"),
    (1.5,  "Strong El Nino",      "#b3562f"),
    (1.0,  "Moderate El Nino",    "#c98a4f"),
    (0.5,  "Weak El Nino",        "#d9b98a"),
    (-0.5, "Neutral",             "#9a9a9a"),
    (-1.0, "Weak La Nina",        "#9db8cf"),
    (-1.5, "Moderate La Nina",    "#6f97b8"),
    (float("-inf"), "Strong La Nina", "#3b6fa0"),
]


# --- Loaders (cached so files are read once) ---
@st.cache_data
def load_meta():
    return json.loads(META.read_text())


@st.cache_data
def load_oni():
    return pd.read_csv(CSV, index_col=0)


@st.cache_data
def load_maps():
    d = np.load(NPZ, allow_pickle=True)
    return {k: d[k] for k in d.files}


@st.cache_data
def load_historical_oni():
    """Observed ONI history (through the model's input cutoff), or None if missing."""
    if not HIST_ONI_CSV.exists():
        return None
    df = pd.read_csv(HIST_ONI_CSV)
    df["time"] = pd.to_datetime(df["time"])
    return df[["time", "nino34_3m"]].dropna()


def enso_category(oni: float) -> tuple[str, str]:
    """Return (label, muted hex-color) for an ONI value using standard NOAA thresholds."""
    for thr, label, color in ENSO_STAGES:
        if oni >= thr:
            return label, color
    return ENSO_STAGES[-1][1], ENSO_STAGES[-1][2]


def region_mean(grid2d: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                box: tuple[float, float, float, float]) -> float:
    """Mean of a (H,W) anomaly grid inside a lat/lon bounding box."""
    lat_min, lat_max, lon_min, lon_max = box
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    sub = grid2d[np.ix_(lat_mask, lon_mask)]
    return float(np.nanmean(sub)) if sub.size else float("nan")


def classify_risk(t2m_z: float, tp_z: float) -> tuple[str, str]:
    """Simple severity rule: the larger of |t2m| / |tp| anomaly drives the risk level."""
    severity = max(abs(t2m_z), abs(tp_z))
    if severity >= 0.5:
        return "HIGH", "#a8502f"
    if severity >= 0.25:
        return "MEDIUM", "#c9932f"
    return "LOW", "#4d7a5c"


# --- Guard: artifacts must exist ---
if not META.exists():
    st.error(
        f"No forecast artifacts found in `{OUT}`.\n\n"
        "Run Section 9 of `notebooks/CNN_TCN_multitask_pacific.ipynb` first to "
        "generate `forecast_meta.json`, `ensemble_oni_2026.csv`, and "
        "`ensemble_maps_2026.npz`."
    )
    st.stop()

meta = load_meta()
oni_df = load_oni()
maps = load_maps()
season = meta["season_label"]

# --- Header ---
st.title("🌊 El Nino & South Asia Climate Forecast")
st.caption(
    f"CNN-TCN multi-task ensemble ({meta['n_members']} members) · "
    f"lead {meta['lead_months']} months · inputs through "
    f"{meta['pac_data_max']} · generated {meta['generated'][:10]}"
)

if meta["skipped"]:
    st.info(
        f"**{season} {2026}** shown. Month(s) skipped for lack of input data: "
        f"{', '.join(meta['skipped'])} "
        f"(needs Pacific observations {meta['lead_months']} months ahead of each target)."
    )

# --- Sidebar controls ---
st.sidebar.header("Controls")
month_options = list(maps["months"])
sel_month = st.sidebar.selectbox("Forecast month", month_options,
                                 index=len(month_options) - 1)
layer = st.sidebar.radio("Map layer", ["Ensemble mean", "Ensemble spread (std)"])
mi = month_options.index(sel_month)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Grid:** {meta['grid']['Hs']}×{meta['grid']['Ws']} cells  \n"
    f"**Lat:** {meta['grid']['lat_range'][0]}–{meta['grid']['lat_range'][1]}°N  \n"
    f"**Lon:** {meta['grid']['lon_range'][0]}–{meta['grid']['lon_range'][1]}°E"
)

# ======================================================================
# Section 1 — ENSO (Nino 3.4 ONI) ensemble forecast
# ======================================================================
st.header("1 · Nino 3.4 ENSO Index (ONI)")

seed_cols = [c for c in oni_df.columns if c.startswith("seed")]
row = oni_df.loc[sel_month] if sel_month in oni_df.index else oni_df.loc[f"{season} mean"]
sel_mean, sel_std = float(row["ens_mean"]), float(row["ens_std"])
label, color = enso_category(sel_mean)

c1, c2, c3 = st.columns(3)
c1.metric(f"{sel_month} ONI (ensemble mean)", f"{sel_mean:+.2f} °C", f"± {sel_std:.2f} spread")
c2.markdown(
    f"### Status\n<span style='color:{color};font-weight:700;font-size:1.4rem'>{label}</span>",
    unsafe_allow_html=True,
)
c3.metric("Ensemble members", meta["n_members"])

# Bar chart with error bars across all months
plot_rows = [m for m in oni_df.index]
means = oni_df.loc[plot_rows, "ens_mean"].values
stds = oni_df.loc[plot_rows, "ens_std"].values
bar_colors = [enso_category(v)[1] for v in means]

fig = go.Figure()
fig.add_bar(
    x=plot_rows, y=means,
    error_y=dict(type="data", array=stds, visible=True),
    marker_color=bar_colors,
    text=[f"{v:+.2f}" for v in means], textposition="outside",
)
fig.add_hline(y=0.5, line_dash="dot", line_color="red",
              annotation_text="El Nino (+0.5)", annotation_position="top left")
fig.add_hline(y=-0.5, line_dash="dot", line_color="blue",
              annotation_text="La Nina (-0.5)", annotation_position="bottom left")
fig.update_layout(
    yaxis_title="ONI (°C)", xaxis_title=None,
    height=380, margin=dict(t=30, b=10), showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Per-member ONI table"):
    st.dataframe(oni_df.style.format("{:.3f}"), use_container_width=True)

# ======================================================================
# Section 2 — South Asia anomaly maps
# ======================================================================
st.header(f"2 · South Asia Impact Maps — {sel_month}")

lats, lons = maps["lats"], maps["lons"]
is_std = layer.startswith("Ensemble spread")


def make_map(field_mean, field_std, title, diverging_scale):
    arr = (field_std if is_std else field_mean)[mi]
    if is_std:
        zmin, zmax, scale = 0.0, float(np.nanmax(field_std[mi])), "Viridis"
    else:
        v = float(np.nanmax(np.abs(field_mean[mi]))) or 1.0
        zmin, zmax, scale = -v, v, diverging_scale
    fig = go.Figure(go.Heatmap(
        z=arr, x=lons, y=lats, zmin=zmin, zmax=zmax,
        colorscale=scale, colorbar=dict(title="σ" if is_std else "anom"),
    ))
    fig.update_layout(
        title=f"{title} — {'spread (std)' if is_std else 'mean anomaly'}",
        xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        height=460, margin=dict(t=40, b=10),
    )
    fig.update_yaxes(scaleanchor=None)
    return fig


m1, m2 = st.columns(2)
m1.plotly_chart(make_map(maps["t2m_mean"], maps["t2m_std"],
                         "2 m Temperature (t2m)", "RdBu_r"),
                use_container_width=True)
m2.plotly_chart(make_map(maps["tp_mean"], maps["tp_std"],
                         "Total Precipitation (tp)", "BrBG"),
                use_container_width=True)

st.caption(
    "Values are standardized anomalies (z-scores) vs. the 1980–2018 training climatology. "
    "For the mean layer, red/brown = warmer/drier, blue/green = cooler/wetter. "
    "The spread (std) layer shows where the 5 ensemble members disagree most (forecast uncertainty)."
)

# ======================================================================
# Section 3 — Downloads
# ======================================================================
st.header("3 · Data")
d1, d2, d3 = st.columns(3)
d1.download_button("⬇ ONI CSV", CSV.read_bytes(), "ensemble_oni_2026.csv", "text/csv")
d2.download_button("⬇ Maps NPZ", NPZ.read_bytes(), "ensemble_maps_2026.npz")
d3.download_button("⬇ Metadata JSON", META.read_bytes(), "forecast_meta.json", "application/json")
