"use client";

import { useMemo } from "react";
import { Activity, Thermometer, TrendingUp, Calendar, ShieldCheck } from "lucide-react";
import { forecastData } from "@/data/forecast";
import { ensoCategory } from "@/types/forecast";

function computeKpiStats() {
  const { historical, oni, meta } = forecastData;
  const recentHistory = historical.slice(-24);
  const forecastRows = oni.rows.filter((r) => r.month !== "JJA mean");

  // Current ONI
  const lastHist = recentHistory[recentHistory.length - 1];
  const currentOni = lastHist?.value ?? 0;
  const currentMonth = lastHist
    ? new Date(lastHist.time).toLocaleDateString("en-US", { month: "short", year: "numeric" })
    : "";

  // Current Phase
  const phase = ensoCategory(currentOni);

  // Forecast Peak
  let peakVal = -Infinity;
  let peakMonth = "";
  let peakMonthLast = "";
  for (const row of forecastRows) {
    if (row.ensMean > peakVal) {
      peakVal = row.ensMean;
      peakMonth = row.month;
    }
  }
  if (forecastRows.length > 0) {
    peakMonthLast = forecastRows[forecastRows.length - 1].month;
  }
  const peakRange = peakMonth === peakMonthLast ? peakMonth : `${peakMonth} \u2013 ${peakMonthLast}`;

  // Confidence
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
  const confidence = Math.round(minAgreement * 100);
  const confidenceLabel =
    confidence >= 80 ? "High" : confidence >= 50 ? "Moderate to High" : "Low";

  return {
    currentOni,
    currentMonth,
    phase,
    peakVal,
    peakRange,
    horizon: `${meta.leadMonths}`,
    confidence,
    confidenceLabel,
  };
}

export default function KpiStrip() {
  const stats = useMemo(() => computeKpiStats(), []);

  const cards = [
    {
      icon: Activity,
      label: "Current ENSO Status",
      value: stats.phase.label,
      color: stats.phase.color,
      caption: "ONI \u2265 +0.5\u00b0C",
    },
    {
      icon: Thermometer,
      label: "Current ONI",
      value: `${stats.currentOni >= 0 ? "+" : ""}${stats.currentOni.toFixed(2)}\u00b0C`,
      color: "#ffffff",
      caption: `Updated: ${stats.currentMonth}`,
    },
    {
      icon: TrendingUp,
      label: "Forecast Peak ONI",
      value: `${stats.peakVal >= 0 ? "+" : ""}${stats.peakVal.toFixed(2)}\u00b0C`,
      color: "#ffffff",
      caption: stats.peakRange,
    },
    {
      icon: Calendar,
      label: "Forecast Horizon",
      value: `${stats.horizon} Months`,
      color: "#ffffff",
      caption: "Jun \u2013 Aug 2026",
    },
    {
      icon: ShieldCheck,
      label: "Forecast Confidence",
      value: `${stats.confidence}%`,
      color: "#ffffff",
      caption: stats.confidenceLabel,
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      {cards.map((card) => (
        <div
          key={card.label}
          className="rounded-xl border border-slate-800 bg-slate-950/80 p-4"
        >
          <div className="flex items-center gap-1.5 text-slate-400">
            <card.icon size={14} />
            <p className="text-[10px] font-medium leading-tight sm:text-xs">{card.label}</p>
          </div>
          <p
            className="mt-2 text-xl font-bold tracking-tight sm:text-2xl"
            style={{ color: card.color }}
          >
            {card.value}
          </p>
          <p className="mt-1 text-[10px] text-slate-500 sm:text-xs">{card.caption}</p>
        </div>
      ))}
    </div>
  );
}
