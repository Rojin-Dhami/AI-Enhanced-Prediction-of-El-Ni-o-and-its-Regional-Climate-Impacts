"""
Export the CNN-TCN ensemble forecast artifacts (forecast_outputs/) into a
single JSON file the Next.js dashboard can import directly (data/forecast.json),
plus copies of the raw artifacts into public/forecast/ for the download links.

Run from the project root (Main/):
    python scripts/export_forecast_data.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "forecast_outputs"
META = OUT / "forecast_meta.json"
CSV = OUT / "ensemble_oni_2026.csv"
NPZ = OUT / "ensemble_maps_2026.npz"
HIST_ONI_CSV = ROOT / "data" / "Combined" / "nino34_index_3month_running_mean_official_oni.csv"

DATA_OUT = ROOT / "data" / "forecast.json"
PUBLIC_OUT = ROOT / "public" / "forecast"

# Approximate rectangular bounding boxes (lat_min, lat_max, lon_min, lon_max) --
# demo-level, coarse-grid regional averaging only, NOT real political borders.
COUNTRY_BOXES = {
    "Nepal": (26.0, 31.0, 80.0, 89.0),
    "India": (8.0, 28.0, 68.0, 88.0),  # core/central-southern box to limit overlap
    "Bangladesh": (21.0, 27.0, 88.0, 93.0),
}

ENSO_STAGES = [  # (threshold, label, muted color) -- checked high to low
    (2.0, "Very Strong El Nino", "#8c2d04"),
    (1.5, "Strong El Nino", "#b3562f"),
    (1.0, "Moderate El Nino", "#c98a4f"),
    (0.5, "Weak El Nino", "#d9b98a"),
    (-0.5, "Neutral", "#9a9a9a"),
    (-1.0, "Weak La Nina", "#9db8cf"),
    (-1.5, "Moderate La Nina", "#6f97b8"),
    (float("-inf"), "Strong La Nina", "#3b6fa0"),
]


def enso_category(oni: float) -> tuple[str, str]:
    for thr, label, color in ENSO_STAGES:
        if oni >= thr:
            return label, color
    return ENSO_STAGES[-1][1], ENSO_STAGES[-1][2]


def region_mean(grid2d: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                box: tuple[float, float, float, float]) -> float:
    lat_min, lat_max, lon_min, lon_max = box
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    sub = grid2d[np.ix_(lat_mask, lon_mask)]
    return float(np.nanmean(sub)) if sub.size else float("nan")


def classify_risk(t2m_z: float, tp_z: float) -> tuple[str, str]:
    severity = max(abs(t2m_z), abs(tp_z))
    if severity >= 0.5:
        return "HIGH", "#a8502f"
    if severity >= 0.25:
        return "MEDIUM", "#c9932f"
    return "LOW", "#4d7a5c"


def build_insights(oni_df: pd.DataFrame, regional_risk: dict) -> list[str]:
    insights = []

    month_rows = [m for m in oni_df.index if m != "JJA mean"]
    first_mean = float(oni_df.loc[month_rows[0], "ens_mean"])
    last_mean = float(oni_df.loc[month_rows[-1], "ens_mean"])
    jja_mean = float(oni_df.loc["JJA mean", "ens_mean"]) if "JJA mean" in oni_df.index else last_mean
    label, _ = enso_category(jja_mean)

    delta = last_mean - first_mean
    if delta >= 0.3:
        trend = f"intensifying through the season (+{delta:.2f} \u00b0C from {month_rows[0]} to {month_rows[-1]})"
    elif delta <= -0.3:
        trend = f"weakening through the season ({delta:.2f} \u00b0C from {month_rows[0]} to {month_rows[-1]})"
    else:
        trend = f"holding roughly steady through the season ({delta:+.2f} \u00b0C from {month_rows[0]} to {month_rows[-1]})"

    insights.append(
        f"The ensemble projects a **{label}** JJA 2026 (ONI {jja_mean:+.2f} \u00b0C), {trend}."
    )

    jja_risk = regional_risk.get("JJA mean", [])
    for entry in jja_risk:
        if entry["risk"] == "LOW":
            continue
        t2m_dir = "warmer" if entry["t2mZ"] > 0 else "cooler"
        tp_dir = "drier" if entry["tpZ"] < 0 else "wetter"
        dominant = "temperature" if abs(entry["t2mZ"]) >= abs(entry["tpZ"]) else "precipitation"
        insights.append(
            f"**{entry['country']}** shows **{entry['risk']}** climate risk for JJA 2026, "
            f"driven mainly by {dominant} anomalies (**{t2m_dir}** by {entry['t2mZ']:+.2f}\u03c3, "
            f"**{tp_dir}** by {entry['tpZ']:+.2f}\u03c3)."
        )

    if len(insights) == 1:
        insights.append(
            "No region crosses the moderate risk threshold (\u2265 0.25\u03c3) for JJA 2026 in this forecast."
        )

    return insights


def main() -> None:
    if not META.exists():
        raise SystemExit(f"No forecast artifacts found in {OUT}. Run Section 9 of the notebook first.")

    meta = json.loads(META.read_text())
    oni_df = pd.read_csv(CSV, index_col=0)
    maps = np.load(NPZ, allow_pickle=True)

    lats = maps["lats"]
    lons = maps["lons"]
    months = list(maps["months"])
    t2m_mean, t2m_std = maps["t2m_mean"], maps["t2m_std"]
    tp_mean, tp_std = maps["tp_mean"], maps["tp_std"]

    seed_cols = [c for c in oni_df.columns if c.startswith("seed")]
    oni_rows = []
    for month in oni_df.index:
        row = oni_df.loc[month]
        oni_rows.append({
            "month": month,
            "seeds": [float(row[c]) for c in seed_cols],
            "ensMean": float(row["ens_mean"]),
            "ensStd": float(row["ens_std"]),
        })

    historical = []
    if HIST_ONI_CSV.exists():
        hist_df = pd.read_csv(HIST_ONI_CSV)
        hist_df["time"] = pd.to_datetime(hist_df["time"])
        hist_df = hist_df[["time", "nino34_3m"]].dropna()
        for _, r in hist_df.iterrows():
            historical.append({
                "time": r["time"].strftime("%Y-%m-%d"),
                "value": float(r["nino34_3m"]),
            })

    regional_risk: dict[str, list] = {}
    for mi, month in enumerate(months):
        entries = []
        for country, box in COUNTRY_BOXES.items():
            t2m_z = region_mean(t2m_mean[mi], lats, lons, box)
            tp_z = region_mean(tp_mean[mi], lats, lons, box)
            risk, color = classify_risk(t2m_z, tp_z)
            entries.append({
                "country": country,
                "t2mZ": round(t2m_z, 4),
                "tpZ": round(tp_z, 4),
                "risk": risk,
                "color": color,
            })
        regional_risk[month] = entries

    insights = build_insights(oni_df, regional_risk)

    def round_grid(arr: np.ndarray) -> list:
        return np.round(arr.astype(float), 4).tolist()

    payload = {
        "meta": {
            "generated": meta["generated"],
            "seasonLabel": meta["season_label"],
            "nMembers": meta["n_members"],
            "leadMonths": meta["lead_months"],
            "seqLen": meta["seq_len"],
            "pacDataMax": meta["pac_data_max"],
            "skipped": meta["skipped"],
            "grid": {
                "hs": meta["grid"]["Hs"],
                "ws": meta["grid"]["Ws"],
                "latRange": meta["grid"]["lat_range"],
                "lonRange": meta["grid"]["lon_range"],
            },
        },
        "oni": {"months": list(oni_df.index), "rows": oni_rows},
        "historical": historical,
        "maps": {
            "months": months,
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "t2mMean": round_grid(t2m_mean),
            "t2mStd": round_grid(t2m_std),
            "tpMean": round_grid(tp_mean),
            "tpStd": round_grid(tp_std),
        },
        "regionalRisk": regional_risk,
        "insights": insights,
    }

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {DATA_OUT} ({DATA_OUT.stat().st_size / 1024:.1f} KB)")

    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    for src in (CSV, NPZ, META):
        shutil.copy2(src, PUBLIC_OUT / src.name)
    print(f"Copied raw artifacts to {PUBLIC_OUT}")


if __name__ == "__main__":
    main()
