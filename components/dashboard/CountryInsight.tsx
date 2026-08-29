"use client";

import { useMemo, useState, useCallback } from "react";
import { Info, Thermometer, CloudRain, Download } from "lucide-react";
import { forecastData } from "@/data/forecast";
import type { CountryRisk, RiskLevel } from "@/types/forecast";
import { downloadJson } from "./downloadUtils";

const RISK_STYLES: Record<RiskLevel, string> = {
  HIGH: "border-[#a8502f]/50 bg-[#a8502f]/10 text-[#e59a80]",
  MEDIUM: "border-[#c9932f]/50 bg-[#c9932f]/10 text-[#e5c079]",
  LOW: "border-[#4d7a5c]/50 bg-[#4d7a5c]/10 text-[#8fbf9f]",
};

const FLAG_CODES: Record<string, string> = {
  Nepal: "np",
  India: "in",
  Bangladesh: "bd",
};

function compositeIndex(r: CountryRisk): number {
  return Math.min(1, Math.max(Math.abs(r.t2mZ), Math.abs(r.tpZ)) / 0.5);
}

function impactInterpretation(risk: RiskLevel, t2mZ: number, tpZ: number): string {
  const warmer = t2mZ > 0;
  const drier = tpZ < 0;

  if (risk === "HIGH") {
    const parts = [];
    if (warmer) parts.push("warmer");
    if (drier) parts.push("drier");
    const conditions = parts.length ? parts.join(" and ") : "anomalous";
    return `${conditions.charAt(0).toUpperCase() + conditions.slice(1)} conditions than normal may increase risk of drought, water stress, and impact on agriculture and ecosystems.`;
  }
  if (risk === "MEDIUM") {
    return "Mild anomalies from normal conditions may have moderate effects on agriculture and water resources.";
  }
  return "Conditions are near normal with minimal expected impact on regional climate-sensitive sectors.";
}

function GaugeArc({ score, risk }: { score: number; risk: RiskLevel }) {
  const startAngle = -90;
  const endAngle = 90;
  const sweep = endAngle - startAngle;
  const needleAngle = startAngle + sweep * score;

  const r = 44;
  const cx = 50;
  const cy = 50;

  function arcPath(start: number, end: number) {
    const s = (start * Math.PI) / 180;
    const e = (end * Math.PI) / 180;
    const x1 = cx + r * Math.cos(s);
    const y1 = cy + r * Math.sin(s);
    const x2 = cx + r * Math.cos(e);
    const y2 = cy + r * Math.sin(e);
    const large = end - start > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  }

  const needleRad = (needleAngle * Math.PI) / 180;
  const nx = cx + (r - 6) * Math.cos(needleRad);
  const ny = cy + (r - 6) * Math.sin(needleRad);

  const color =
    risk === "HIGH" ? "#a8502f" : risk === "MEDIUM" ? "#c9932f" : "#4d7a5c";

  return (
    <svg viewBox="0 0 100 65" className="w-full max-w-[180px]">
      {/* Background arc */}
      <path d={arcPath(startAngle, endAngle)} fill="none" stroke="#1e293b" strokeWidth="8" strokeLinecap="round" />
      {/* Green zone */}
      <path d={arcPath(startAngle, startAngle + sweep * 0.3)} fill="none" stroke="#4d7a5c" strokeWidth="8" strokeLinecap="round" opacity={0.4} />
      {/* Yellow zone */}
      <path d={arcPath(startAngle + sweep * 0.3, startAngle + sweep * 0.6)} fill="none" stroke="#c9932f" strokeWidth="8" strokeLinecap="round" opacity={0.4} />
      {/* Red zone */}
      <path d={arcPath(startAngle + sweep * 0.6, endAngle)} fill="none" stroke="#a8502f" strokeWidth="8" strokeLinecap="round" opacity={0.4} />
      {/* Filled arc up to score */}
      <path d={arcPath(startAngle, startAngle + sweep * score)} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round" />
      {/* Needle */}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#e2e8f0" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="2.5" fill="#e2e8f0" />
      {/* Score text */}
      <text x={cx} y={cy - 8} textAnchor="middle" className="fill-white text-[11px] font-bold">
        {risk}
      </text>
    </svg>
  );
}

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

export default function CountryInsight() {
  const risks = useMemo(() => forecastData.regionalRisk["JJA mean"] ?? [], []);
  const countries = useMemo(() => risks.map((r) => r.country), [risks]);
  const [selected, setSelected] = useState(countries[0] ?? "");

  const current = useMemo(() => risks.find((r) => r.country === selected) ?? risks[0], [risks, selected]);
  const ci = current ? compositeIndex(current) : 0;

  const handleDownload = useCallback(() => {
    if (!current) return;
    downloadJson({
      country: current.country,
      temperatureAnomaly: current.t2mZ,
      precipitationAnomaly: current.tpZ,
      compositeIndex: ci,
      riskLevel: current.risk,
    }, `country-insight-${current.country.toLowerCase()}-${Date.now()}.json`);
  }, [current, ci]);

  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
            COUNTRY INSIGHT
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`fi fi-${FLAG_CODES[selected] ?? ""} text-base`} />
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white outline-none transition-colors focus:border-cyan-500"
          >
            {countries.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <button className={ICON_BTN} title="Info">
            <Info size={14} />
          </button>
          <button className={ICON_BTN} title="Download country data" onClick={handleDownload}>
            <Download size={14} />
          </button>
        </div>
      </div>

      {current && (
        <>
          {/* Two-column stat block */}
          <div className="mt-4 grid grid-cols-2 gap-4">
            {/* Temperature */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
              <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
                <Thermometer size={12} />
                Temperature
              </div>
              <div className="mt-2 space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Anomaly</span>
                  <span className="font-medium text-white">
                    {current.t2mZ >= 0 ? "+" : ""}{current.t2mZ.toFixed(2)}&sigma;
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Impact</span>
                  <span className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${RISK_STYLES[current.risk]}`}>
                    {current.risk}
                  </span>
                </div>
              </div>
            </div>

            {/* Precipitation */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
              <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
                <CloudRain size={12} />
                Precipitation
              </div>
              <div className="mt-2 space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Anomaly</span>
                  <span className="font-medium text-white">
                    {current.tpZ >= 0 ? "+" : ""}{current.tpZ.toFixed(2)}&sigma;
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Impact</span>
                  <span className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${RISK_STYLES[current.risk]}`}>
                    {current.risk}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Overall Impact Gauge */}
          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Overall Impact
            </p>
            <div className="mt-2 flex items-center gap-4">
              <GaugeArc score={ci} risk={current.risk} />
              <p className="text-xs leading-5 text-slate-400">
                {impactInterpretation(current.risk, current.t2mZ, current.tpZ)}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
