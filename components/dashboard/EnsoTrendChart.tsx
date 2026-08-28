"use client";

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

interface ChartPoint {
  date: string;
  label: string;
  hist: number | null;
  forecast: number | null;
  band: [number, number] | null;
}

function buildChartData(): ChartPoint[] {
  const { historical, oni } = forecastData;
  const recentHistory = historical.slice(-24);
  const forecastRows = oni.rows.filter((r) => r.month !== "JJA mean");

  const points: ChartPoint[] = recentHistory.map((h) => ({
    date: h.time,
    label: new Date(h.time).toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
    hist: h.value,
    forecast: null,
    band: null,
  }));

  const last = recentHistory[recentHistory.length - 1];
  if (last) {
    points.push({
      date: `${last.time}-bridge`,
      label: new Date(last.time).toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
      hist: last.value,
      forecast: last.value,
      band: [last.value, last.value],
    });
  }

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

  return points;
}

export default function EnsoTrendChart() {
  const data = buildChartData();

  return (
    <div className="h-72 w-full sm:h-80">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="label" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={40} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
            formatter={(value, name) => {
              if (Array.isArray(value)) return [`${Number(value[0]).toFixed(2)} to ${Number(value[1]).toFixed(2)}`, "Spread"];
              const num = Number(value);
              return [Number.isFinite(num) ? `${num.toFixed(2)} \u00b0C` : String(value), String(name)];
            }}
          />
          <ReferenceLine y={0.5} stroke="#b3562f" strokeDasharray="4 4" label={{ value: "El Ni\u00f1o", fill: "#b3562f", fontSize: 10, position: "insideTopLeft" }} />
          <ReferenceLine y={-0.5} stroke="#3b6fa0" strokeDasharray="4 4" label={{ value: "La Ni\u00f1a", fill: "#3b6fa0", fontSize: 10, position: "insideBottomLeft" }} />
          <Area dataKey="band" stroke="none" fill="#4C8BB4" fillOpacity={0.18} isAnimationActive={false} />
          <Line dataKey="hist" stroke="#8a8f98" strokeWidth={2} dot={false} name="Historical" isAnimationActive={false} />
          <Line dataKey="forecast" stroke="#4C8BB4" strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3, fill: "#4C8BB4" }} name="Forecast" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
