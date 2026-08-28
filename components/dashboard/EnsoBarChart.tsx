"use client";

import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { forecastData } from "@/data/forecast";
import { ensoCategory } from "@/types/forecast";

export default function EnsoBarChart() {
  const rows = forecastData.oni.rows;
  const data = rows.map((r) => ({ month: r.month, ensMean: r.ensMean, ensStd: r.ensStd }));

  return (
    <div className="h-72 w-full sm:h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={40} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
            formatter={(value) => [`${Number(value).toFixed(2)} \u00b0C`, "ONI"]}
          />
          <ReferenceLine y={0.5} stroke="#b3562f" strokeDasharray="4 4" />
          <ReferenceLine y={-0.5} stroke="#3b6fa0" strokeDasharray="4 4" />
          <Bar dataKey="ensMean" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            <ErrorBar dataKey="ensStd" stroke="#e2e8f0" strokeWidth={1.5} width={4} />
            {data.map((d) => (
              <Cell key={d.month} fill={ensoCategory(d.ensMean).color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
