"use client";

import { useMemo } from "react";
import { forecastData } from "@/data/forecast";
import {
  divergingColor,
  percentileAbs,
  sequentialColor,
  SPREAD_SCALE,
} from "@/lib/colorScale";
import ChoroplethMap, { MAP_TEMP_SCALE, MAP_PRECIP_SCALE } from "./ChoroplethMap";

interface RegionalMapsProps {
  monthIndex: number;
  showSpread: boolean;
  layout?: "horizontal" | "vertical";
}

const ISO_BY_NAME: Record<string, string> = {
  Nepal: "524",
  India: "356",
  Bangladesh: "050",
};

export default function RegionalMaps({ monthIndex, showSpread, layout = "horizontal" }: RegionalMapsProps) {
  const { maps } = forecastData;

  const t2mGrid = showSpread ? maps.t2mStd[monthIndex] : maps.t2mMean[monthIndex];
  const tpGrid = showSpread ? maps.tpStd[monthIndex] : maps.tpMean[monthIndex];

  // Clip to the 92nd percentile so a few extreme cells don't wash out the rest of the map.
  const t2mBound = useMemo(() => percentileAbs(t2mGrid.flat(), 92), [t2mGrid]);
  const tpBound = useMemo(() => percentileAbs(tpGrid.flat(), 92), [tpGrid]);

  const t2mColor = (v: number) =>
    showSpread ? sequentialColor(v, t2mBound, SPREAD_SCALE) : divergingColor(v, t2mBound, MAP_TEMP_SCALE);
  const tpColor = (v: number) =>
    showSpread ? sequentialColor(v, tpBound, SPREAD_SCALE) : divergingColor(v, tpBound, MAP_PRECIP_SCALE);

  const risks = useMemo(() => forecastData.regionalRisk["JJA mean"] ?? [], []);

  const t2mValueOf = useMemo(() => {
    const map = new Map(risks.map((r) => [ISO_BY_NAME[r.country], r.t2mZ]));
    return (isoId: string) => map.get(isoId) ?? null;
  }, [risks]);

  const tpValueOf = useMemo(() => {
    const map = new Map(risks.map((r) => [ISO_BY_NAME[r.country], r.tpZ]));
    return (isoId: string) => map.get(isoId) ?? null;
  }, [risks]);

  return (
    <div className={`grid gap-8 ${layout === "vertical" ? "grid-cols-1" : "md:grid-cols-2"}`}>
      <div>
        <p className="mb-2 text-sm font-medium text-slate-300">
          2m Temperature {showSpread ? "\u2014 ensemble spread" : "\u2014 mean anomaly"}
        </p>
        <ChoroplethMap
          colorOf={t2mColor}
          valueOf={t2mValueOf}
          scaleStops={showSpread ? SPREAD_SCALE : MAP_TEMP_SCALE}
          scaleLabels={
            showSpread
              ? { low: "0", high: `${t2mBound.toFixed(2)}σ`, unit: "ensemble spread (σ)" }
              : { low: `−${t2mBound.toFixed(2)} cooler`, high: `+${t2mBound.toFixed(2)} warmer`, unit: "anomaly (z-score)" }
          }
        />
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-slate-300">
          Total Precipitation {showSpread ? "\u2014 ensemble spread" : "\u2014 mean anomaly"}
        </p>
        <ChoroplethMap
          colorOf={tpColor}
          valueOf={tpValueOf}
          scaleStops={showSpread ? SPREAD_SCALE : MAP_PRECIP_SCALE}
          scaleLabels={
            showSpread
              ? { low: "0", high: `${tpBound.toFixed(2)}σ`, unit: "ensemble spread (σ)" }
              : { low: `−${tpBound.toFixed(2)} drier`, high: `+${tpBound.toFixed(2)} wetter`, unit: "anomaly (z-score)" }
          }
        />
      </div>
      <p className="col-span-full -mt-2 text-[11px] text-slate-600">
        Countries without anomaly data are shown in neutral gray. Only Nepal (NP), India (IN), and Bangladesh (BD) have
        forecast data. Color scale is clipped to the 92nd percentile of anomalies to keep the typical range visible.
      </p>
    </div>
  );
}

