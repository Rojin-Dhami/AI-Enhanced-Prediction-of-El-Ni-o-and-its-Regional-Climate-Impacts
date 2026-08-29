"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { forecastData } from "@/data/forecast";
import { ensoCategory } from "@/types/forecast";

interface ChartPoint {
  date: string;
  label: string;
  hist: number | null;
  forecast: number | null;
  band: [number, number] | null;
}

function buildChartData(): { data: ChartPoint[]; boundaryLabel: string } {
  const { historical, oni } = forecastData;
  const recentHistory = historical.slice(-24);
  const forecastRows = oni.rows.filter((r) => r.month !== "JJA mean");

  const points: ChartPoint[] = recentHistory.map((h, i) => {
    const isLast = i === recentHistory.length - 1;
    return {
      date: h.time,
      label: new Date(h.time).toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
      hist: h.value,
      forecast: isLast ? h.value : null,
      band: isLast ? [h.value, h.value] : null,
    };
  });

  for (const row of forecastRows) {
    const [mon, yr] = row.month.split(" ");
    points.push({
      date: `${yr}-${mon}`,
      label: row.month,
      hist: null,
      forecast: row.ensMean,
      band: [row.ensMean - row.ensStd, row.ensMean + row.ensStd],
    });
  }

  const boundaryLabel = recentHistory.length
    ? new Date(recentHistory[recentHistory.length - 1].time).toLocaleDateString("en-US", { month: "short", year: "2-digit" })
    : "";

  return { data: points, boundaryLabel };
}

function computeStats() {
  const { historical, oni, meta } = forecastData;
  const recentHistory = historical.slice(-24);
  const forecastRows = oni.rows.filter((r) => r.month !== "JJA mean");

  const lastHist = recentHistory[recentHistory.length - 1];
  const currentOni = lastHist?.value ?? 0;
  const currentMonth = lastHist
    ? new Date(lastHist.time).toLocaleDateString("en-US", { month: "short", year: "numeric" })
    : "";
  const phase = ensoCategory(currentOni);

  let peakVal = -Infinity;
  let peakMonth = "";
  for (const row of forecastRows) {
    if (row.ensMean > peakVal) {
      peakVal = row.ensMean;
      peakMonth = row.month;
    }
  }

  // Confidence: % of seeds in the same ENSO phase as the ensemble mean, worst case across months
  let minAgreement = 1;
  for (const row of forecastRows) {
    const meanPhase = ensoCategory(row.ensMean);
    let matchCount = 0;
    for (const seed of row.seeds) {
      if (ensoCategory(seed).label === meanPhase.label) matchCount++;
    }
    const agreement = matchCount / row.seeds.length;
    if (agreement < minAgreement) minAgreement = agreement;
  }

  return {
    currentOni,
    currentMonth,
    phase,
    peakVal,
    peakMonth,
    horizon: `${meta.leadMonths} Months`,
    confidence: Math.round(minAgreement * 100),
  };
}

const CHART_COLORS = {
  hist: "#8a8f98",
  forecast: "#4C8BB4",
  band: "#4C8BB4",
  elNino: "#b3562f",
  laNina: "#3b6fa0",
  grid: "#1e293b",
  tick: "#94a3b8",
  bg: "#0f172a",
  border: "#1e293b",
};

function StatStrip({ stats }: { stats: ReturnType<typeof computeStats> }) {
  const items = [
    { label: "Current ONI", value: `${stats.currentOni >= 0 ? "+" : ""}${stats.currentOni.toFixed(2)} °C`, sub: stats.currentMonth },
    { label: "Current Phase", value: stats.phase.label, dot: stats.phase.color },
    { label: "Forecast Peak", value: `${stats.peakVal >= 0 ? "+" : ""}${stats.peakVal.toFixed(2)} °C`, sub: stats.peakMonth },
    { label: "Forecast Horizon", value: stats.horizon },
    { label: "Confidence", value: `${stats.confidence}%` },
  ];

  return (
    <div className="mb-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-slate-800 bg-slate-950/80 px-2.5 py-2 text-center">
          <p className="text-[10px] leading-tight text-slate-500 sm:text-xs">{item.label}</p>
          <div className="mt-1 flex items-center justify-center gap-1.5">
            {"dot" in item && item.dot && (
              <span className="inline-block h-2 w-2 flex-none rounded-full" style={{ backgroundColor: item.dot }} />
            )}
            <p className="text-xs font-bold text-white sm:text-sm">{item.value}</p>
          </div>
          {"sub" in item && item.sub && (
            <p className="mt-0.5 text-[10px] text-slate-500">{item.sub}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function CustomLegend() {
  const items = [
    { label: "Historical (Observed)", style: { background: CHART_COLORS.hist, width: 20, height: 2, borderRadius: 1 } },
    { label: "Forecast (Ensemble Mean)", style: { background: `repeating-linear-gradient(to right, ${CHART_COLORS.forecast} 0px, ${CHART_COLORS.forecast} 5px, transparent 5px, transparent 9px)`, width: 20, height: 2 } },
    { label: "Uncertainty (± Spread)", style: { background: CHART_COLORS.band, width: 14, height: 10, borderRadius: 2, opacity: 0.3 } },
  ];

  return (
    <div className="mb-2 flex flex-wrap items-center gap-4 text-[11px] text-slate-400">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5">
          <span className="inline-block flex-none" style={item.style} />
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ name: string; value: number | [number, number] | null; dataKey: string }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  const histVal = payload.find((p) => p.dataKey === "hist")?.value;
  const forecastVal = payload.find((p) => p.dataKey === "forecast")?.value;
  const bandVal = payload.find((p) => p.dataKey === "band")?.value;

  return (
    <div style={{ background: CHART_COLORS.bg, border: `1px solid ${CHART_COLORS.border}`, borderRadius: 8, padding: "8px 12px" }}>
      <p className="mb-1 text-xs font-medium" style={{ color: "#e2e8f0" }}>{label}</p>
      {histVal != null && typeof histVal === "number" && (
        <p className="text-xs" style={{ color: CHART_COLORS.hist }}>
          Historical ONI: {histVal >= 0 ? "+" : ""}{histVal.toFixed(2)} °C
        </p>
      )}
      {forecastVal != null && typeof forecastVal === "number" && (
        <p className="text-xs" style={{ color: CHART_COLORS.forecast }}>
          Forecast Mean: {forecastVal >= 0 ? "+" : ""}{forecastVal.toFixed(2)} °C
        </p>
      )}
      {Array.isArray(bandVal) && typeof bandVal[0] === "number" && typeof bandVal[1] === "number" && (
        <p className="text-xs text-slate-400">
          Spread: {bandVal[0] >= 0 ? "+" : ""}{bandVal[0].toFixed(2)} to {bandVal[1] >= 0 ? "+" : ""}{bandVal[1].toFixed(2)} °C
        </p>
      )}
    </div>
  );
}

export default function EnsoTrendChart() {
  const stats = useMemo(() => computeStats(), []);
  const { data, boundaryLabel } = useMemo(() => buildChartData(), []);

  const yDomain = useMemo(() => {
    const forecasts = data.filter((d) => d.forecast != null).map((d) => d.forecast as number);
    const bandLows = data.filter((d) => d.band != null).map((d) => (d.band as [number, number])[0]);
    const bandHighs = data.filter((d) => d.band != null).map((d) => (d.band as [number, number])[1]);
    const allVals = [...forecasts, ...bandLows, ...bandHighs];
    if (!allVals.length) return [-1, 2];
    const maxVal = Math.max(...allVals);
    const minVal = Math.min(...allVals);
    return [Math.min(-1.0, minVal - 0.3), Math.max(maxVal + 0.4, 1.3)];
  }, [data]);

  return (
    <div className="h-full flex flex-col">
      <StatStrip stats={stats} />
      <CustomLegend />
      <div className="flex-1 min-h-0 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 30, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
            <XAxis
              dataKey="label"
              tick={{ fill: CHART_COLORS.tick, fontSize: 11 }}
              minTickGap={20}
            />
            <YAxis
              tick={{ fill: CHART_COLORS.tick, fontSize: 11 }}
              width={44}
              domain={yDomain}
              tickFormatter={(v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`}
              label={{
                value: "ONI (°C)",
                angle: -90,
                position: "insideLeft",
                offset: -10,
                style: { fill: CHART_COLORS.tick, fontSize: 11 },
              }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* El Niño / La Niña threshold lines */}
            <ReferenceLine
              y={0.5}
              stroke={CHART_COLORS.elNino}
              strokeDasharray="4 4"
              label={{
                value: "El Niño",
                fill: CHART_COLORS.elNino,
                fontSize: 10,
                position: "insideTopLeft",
                offset: 6,
              }}
            />
            <ReferenceLine
              y={-0.5}
              stroke={CHART_COLORS.laNina}
              strokeDasharray="4 4"
              label={{
                value: "La Niña",
                fill: CHART_COLORS.laNina,
                fontSize: 10,
                position: "insideBottomLeft",
                offset: 6,
              }}
            />

            {/* Zone labels */}
            <ReferenceLine y={1.0} stroke="none" label={{ value: "EL NIÑO", fill: "#b3562f", fontSize: 9, position: "insideTopRight", opacity: 0.4 }} />
            <ReferenceLine y={0} stroke="none" label={{ value: "NEUTRAL", fill: "#9a9a9a", fontSize: 9, position: "insideRight", opacity: 0.4 }} />
            <ReferenceLine y={-1.0} stroke="none" label={{ value: "LA NIÑA", fill: "#3b6fa0", fontSize: 9, position: "insideBottomRight", opacity: 0.4 }} />

            <ReferenceLine
              x={boundaryLabel}
              stroke="#475569"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={({ viewBox }: { viewBox: { x: number; y: number } }) => (
                <g>
                  <text x={viewBox.x - 8} y={viewBox.y + 12} textAnchor="end" fill="#94a3b8" fontSize={10}>
                    ← Historical
                  </text>
                  <text x={viewBox.x + 8} y={viewBox.y + 12} textAnchor="start" fill="#94a3b8" fontSize={10}>
                    Forecast →
                  </text>
                </g>
              )}
            />

            {/* Uncertainty band (forecast only) */}
            <Area
              dataKey="band"
              stroke="none"
              fill={CHART_COLORS.band}
              fillOpacity={0.15}
              isAnimationActive={false}
            />

            {/* Historical line */}
            <Line
              dataKey="hist"
              stroke={CHART_COLORS.hist}
              strokeWidth={2}
              dot={{ r: 2, fill: CHART_COLORS.hist, strokeWidth: 0 }}
              name="Historical"
              connectNulls={false}
              isAnimationActive={false}
            />

            {/* Forecast line */}
            <Line
              dataKey="forecast"
              stroke={CHART_COLORS.forecast}
              strokeWidth={2.5}
              strokeDasharray="6 4"
              dot={{ r: 3, fill: CHART_COLORS.forecast }}
              name="Forecast"
              connectNulls={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
