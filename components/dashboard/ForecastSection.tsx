"use client";

import { useMemo, useState } from "react";
import { Download, Satellite, Sparkles } from "lucide-react";
import { forecastData } from "@/data/forecast";
import { ensoCategory } from "@/types/forecast";
import EnsoTrendChart from "./EnsoTrendChart";
import EnsoBarChart from "./EnsoBarChart";
import RegionalMaps from "./RegionalMaps";
import RegionalRiskCards from "./RegionalRiskCards";
import FormattedText from "./FormattedText";

export default function ForecastSection() {
  const { meta, oni, regionalRisk, insights } = forecastData;
  const months = oni.months;

  const [selectedMonth, setSelectedMonth] = useState(months[months.length - 1]);
  const [showSpread, setShowSpread] = useState(false);

  const monthIndex = forecastData.maps.months.indexOf(selectedMonth);
  const row = useMemo(() => oni.rows.find((r) => r.month === selectedMonth), [oni.rows, selectedMonth]);
  const { label, color } = ensoCategory(row?.ensMean ?? 0);
  const risks = regionalRisk[selectedMonth] ?? [];

  return (
    <section className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 sm:p-8 lg:p-10">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.18em] text-cyan-400 sm:text-sm">
            <Satellite size={18} />
            LIVE FORECAST
          </div>
          <h2 className="mt-2 text-2xl font-bold tracking-tight text-white sm:text-3xl">
            JJAS 2026 El Niño &amp; South Asia Forecast
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
            CNN-TCN multi-task ensemble ({meta.nMembers} members) &middot; lead {meta.leadMonths} months &middot;
            inputs through {meta.pacDataMax} &middot; generated {meta.generated.slice(0, 10)}
          </p>
        </div>

        <select
          value={selectedMonth}
          onChange={(e) => setSelectedMonth(e.target.value)}
          className="h-fit w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 sm:w-48"
        >
          {months.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      {meta.skipped.length > 0 && (
        <p className="mt-4 rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-2.5 text-xs text-slate-500">
          Month(s) skipped for lack of input data: {meta.skipped.join(", ")} (needs Pacific observations{" "}
          {meta.leadMonths} months ahead of each target).
        </p>
      )}

      {/* Metric cards */}
      <div className="mt-7 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
          <p className="text-sm text-slate-400">{selectedMonth} ONI (ensemble mean)</p>
          <p className="mt-2 text-3xl font-bold tracking-tight text-white">
            {row ? `${row.ensMean >= 0 ? "+" : ""}${row.ensMean.toFixed(2)} \u00b0C` : "\u2014"}
          </p>
          <p className="mt-1 text-sm text-slate-500">&plusmn; {row?.ensStd.toFixed(2)} spread</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
          <p className="text-sm text-slate-400">Status</p>
          <p className="mt-2 text-2xl font-bold tracking-tight" style={{ color }}>
            {label}
          </p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
          <p className="text-sm text-slate-400">Ensemble members</p>
          <p className="mt-2 text-3xl font-bold tracking-tight text-white">{meta.nMembers}</p>
          <p className="mt-1 text-sm text-slate-500">seq_len {meta.seqLen} &middot; lead {meta.leadMonths}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4 sm:p-5">
          <p className="mb-2 text-sm font-medium text-slate-300">Historical &rarr; Forecast ONI Trend</p>
          <EnsoTrendChart />
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-4 sm:p-5">
          <p className="mb-2 text-sm font-medium text-slate-300">Ensemble ONI by Month</p>
          <EnsoBarChart />
        </div>
      </div>

      {/* Maps */}
      <div className="mt-8 border-t border-slate-800 pt-7">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-xl font-semibold text-white">South Asia Impact Maps &mdash; {selectedMonth}</h3>
          <div className="flex gap-2 text-xs">
            <button
              onClick={() => setShowSpread(false)}
              className={`rounded-lg border px-3 py-1.5 font-medium transition-colors ${
                !showSpread ? "border-cyan-500 bg-cyan-500/10 text-cyan-300" : "border-slate-700 text-slate-400 hover:text-slate-200"
              }`}
            >
              Ensemble mean
            </button>
            <button
              onClick={() => setShowSpread(true)}
              className={`rounded-lg border px-3 py-1.5 font-medium transition-colors ${
                showSpread ? "border-cyan-500 bg-cyan-500/10 text-cyan-300" : "border-slate-700 text-slate-400 hover:text-slate-200"
              }`}
            >
              Ensemble spread (std)
            </button>
          </div>
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-500">
          Standardized anomalies (z-scores) vs. the 1980&ndash;2018 training climatology. Warmer/drier colors indicate
          positive temperature/negative precipitation anomalies; the spread layer shows where the {meta.nMembers}{" "}
          ensemble members disagree most.
        </p>
        <div className="mt-5">
          <RegionalMaps monthIndex={monthIndex} showSpread={showSpread} />
        </div>
      </div>

      {/* Regional risk */}
      <div className="mt-8 border-t border-slate-800 pt-7">
        <h3 className="text-xl font-semibold text-white">Regional Climate Risk &mdash; {selectedMonth}</h3>
        <p className="mt-2 text-sm text-slate-500">
          Coarse-grid regional averages over approximate bounding boxes, driven by the larger of the temperature or
          precipitation anomaly.
        </p>
        <div className="mt-5">
          <RegionalRiskCards risks={risks} />
        </div>
      </div>

      {/* Insights */}
      <div className="mt-8 border-t border-slate-800 pt-7">
        <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.18em] text-cyan-400 sm:text-sm">
          <Sparkles size={16} />
          KEY INSIGHTS
        </div>
        <ul className="mt-4 space-y-2.5 text-sm leading-6 text-slate-400">
          {insights.map((insight, i) => (
            <li key={i} className="flex gap-2">
              <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-cyan-400/70" />
              <span>
                <FormattedText text={insight} />
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* Downloads */}
      <div className="mt-8 border-t border-slate-800 pt-7">
        <h3 className="text-xl font-semibold text-white">Data</h3>
        <div className="mt-4 flex flex-wrap gap-3">
          <a
            href="/forecast/ensemble_oni_2026.csv"
            download
            className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500 hover:text-cyan-300"
          >
            <Download size={14} /> ONI CSV
          </a>
          <a
            href="/forecast/ensemble_maps_2026.npz"
            download
            className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500 hover:text-cyan-300"
          >
            <Download size={14} /> Maps NPZ
          </a>
          <a
            href="/forecast/forecast_meta.json"
            download
            className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500 hover:text-cyan-300"
          >
            <Download size={14} /> Metadata JSON
          </a>
        </div>
      </div>
    </section>
  );
}
