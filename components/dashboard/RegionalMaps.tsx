"use client";

import { useMemo } from "react";
import { forecastData } from "@/data/forecast";
import {
  ColorStop,
  divergingColor,
  percentileAbs,
  PRECIP_SCALE,
  sequentialColor,
  SPREAD_SCALE,
  TEMP_SCALE,
} from "@/lib/colorScale";
import { COUNTRY_BOXES } from "@/lib/countryBoxes";

interface RegionalMapsProps {
  monthIndex: number;
  showSpread: boolean;
}

const LAT_MIN = 5;
const LAT_MAX = 39;
const LON_MIN = 60;
const LON_MAX = 100;

function toX(lon: number) {
  return ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * 100;
}
function toY(lat: number) {
  // North (higher lat) is up, i.e. smaller SVG y.
  return 100 - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * 100;
}

function GridHeatmap({
  grid,
  lats,
  lons,
  colorOf,
  scaleStops,
  scaleLabels,
}: {
  grid: number[][];
  lats: number[];
  lons: number[];
  colorOf: (value: number) => string;
  scaleStops: ColorStop[];
  scaleLabels: { low: string; high: string; unit: string };
}) {
  const cellW = 100 / lons.length;
  const cellH = 100 / lats.length;
  const latTicks = lats.filter((_, i) => i % 3 === 0);
  const lonTicks = lons.filter((_, i) => i % 3 === 0);

  return (
    <div>
      <div className="flex gap-1.5">
        <div className="relative w-7 flex-none">
          {latTicks.map((lat) => (
            <span
              key={lat}
              className="absolute right-1 -translate-y-1/2 text-[9px] text-slate-500"
              style={{ top: `${toY(lat)}%` }}
            >
              {lat}°
            </span>
          ))}
        </div>
        <div className="relative flex-1">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-56 w-full rounded-lg border border-slate-800 sm:h-64">
            {grid.map((row, ri) =>
              row.map((value, ci) => {
                const title = `${lats[ri]}\u00b0N, ${lons[ci]}\u00b0E: ${value.toFixed(2)}`;
                return (
                  <rect
                    key={`${ri}-${ci}`}
                    x={ci * cellW}
                    y={(lats.length - 1 - ri) * cellH}
                    width={cellW + 0.15}
                    height={cellH + 0.15}
                    fill={colorOf(value)}
                  >
                    <title>{title}</title>
                  </rect>
                );
              })
            )}
            {COUNTRY_BOXES.map((box) => (
              <rect
                key={box.name}
                x={toX(box.lonMin)}
                y={toY(box.latMax)}
                width={toX(box.lonMax) - toX(box.lonMin)}
                height={toY(box.latMin) - toY(box.latMax)}
                fill="none"
                stroke="#e2e8f0"
                strokeOpacity={0.6}
                strokeWidth={0.4}
                strokeDasharray="1.5 1"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
          {COUNTRY_BOXES.map((box) => (
            <span
              key={box.name}
              className="pointer-events-none absolute rounded bg-slate-950/70 px-1 text-[9px] font-semibold text-slate-200"
              style={{ left: `${toX(box.lonMin)}%`, top: `${toY(box.latMax)}%` }}
            >
              {box.abbr}
            </span>
          ))}
          <div className="mt-1 flex justify-between">
            {lonTicks.map((lon) => (
              <span key={lon} className="text-[9px] text-slate-500">
                {lon}°
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2 pl-8">
        <span className="text-[11px] text-slate-500">{scaleLabels.low}</span>
        <div
          className="h-2 flex-1 rounded-full"
          style={{
            background: `linear-gradient(to right, ${scaleStops.map(([t, c]) => `${c} ${t * 100}%`).join(", ")})`,
          }}
        />
        <span className="text-[11px] text-slate-500">{scaleLabels.high}</span>
      </div>
      <p className="mt-1 pl-8 text-[10px] text-slate-600">{scaleLabels.unit}</p>
    </div>
  );
}

export default function RegionalMaps({ monthIndex, showSpread }: RegionalMapsProps) {
  const { maps } = forecastData;

  const t2mGrid = showSpread ? maps.t2mStd[monthIndex] : maps.t2mMean[monthIndex];
  const tpGrid = showSpread ? maps.tpStd[monthIndex] : maps.tpMean[monthIndex];

  // Clip to the 92nd percentile so a few extreme cells don't wash out the rest of the map.
  const t2mBound = useMemo(() => percentileAbs(t2mGrid.flat(), 92), [t2mGrid]);
  const tpBound = useMemo(() => percentileAbs(tpGrid.flat(), 92), [tpGrid]);

  const t2mColor = (v: number) =>
    showSpread ? sequentialColor(v, t2mBound, SPREAD_SCALE) : divergingColor(v, t2mBound, TEMP_SCALE);
  const tpColor = (v: number) =>
    showSpread ? sequentialColor(v, tpBound, SPREAD_SCALE) : divergingColor(v, tpBound, PRECIP_SCALE);

  return (
    <div className="grid gap-8 md:grid-cols-2">
      <div>
        <p className="mb-2 text-sm font-medium text-slate-300">
          2m Temperature {showSpread ? "\u2014 ensemble spread" : "\u2014 mean anomaly"}
        </p>
        <GridHeatmap
          grid={t2mGrid}
          lats={maps.lats}
          lons={maps.lons}
          colorOf={t2mColor}
          scaleStops={showSpread ? SPREAD_SCALE : TEMP_SCALE}
          scaleLabels={
            showSpread
              ? { low: "0", high: `${t2mBound.toFixed(2)}\u03c3`, unit: "ensemble spread (\u03c3)" }
              : { low: `\u2212${t2mBound.toFixed(2)} cooler`, high: `+${t2mBound.toFixed(2)} warmer`, unit: "anomaly (z-score)" }
          }
        />
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-slate-300">
          Total Precipitation {showSpread ? "\u2014 ensemble spread" : "\u2014 mean anomaly"}
        </p>
        <GridHeatmap
          grid={tpGrid}
          lats={maps.lats}
          lons={maps.lons}
          colorOf={tpColor}
          scaleStops={showSpread ? SPREAD_SCALE : PRECIP_SCALE}
          scaleLabels={
            showSpread
              ? { low: "0", high: `${tpBound.toFixed(2)}\u03c3`, unit: "ensemble spread (\u03c3)" }
              : { low: `\u2212${tpBound.toFixed(2)} drier`, high: `+${tpBound.toFixed(2)} wetter`, unit: "anomaly (z-score)" }
          }
        />
      </div>
      <p className="col-span-full -mt-2 text-[11px] text-slate-600">
        Dashed boxes mark the approximate Nepal (NP) / India (IN) / Bangladesh (BD) regions used for the risk cards
        below — not political borders. Color scale is clipped to the 92nd percentile of anomalies to keep the
        typical range visible.
      </p>
    </div>
  );
}

