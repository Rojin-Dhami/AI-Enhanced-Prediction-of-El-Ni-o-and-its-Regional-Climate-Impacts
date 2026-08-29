"use client";

import { useCallback } from "react";
import { Info, Download } from "lucide-react";
import { forecastData } from "@/data/forecast";
import type { CountryRisk, RiskLevel } from "@/types/forecast";
import Tooltip from "@/components/ui/Tooltip";
import { downloadCsv } from "./downloadUtils";

const FLAG_CODES: Record<string, string> = {
  Nepal: "np",
  India: "in",
  Bangladesh: "bd",
};

const RISK_STYLES: Record<RiskLevel, string> = {
  HIGH: "border-[#a8502f]/50 bg-[#a8502f]/10 text-[#e59a80]",
  MEDIUM: "border-[#c9932f]/50 bg-[#c9932f]/10 text-[#e5c079]",
  LOW: "border-[#4d7a5c]/50 bg-[#4d7a5c]/10 text-[#8fbf9f]",
};

function compositeIndex(r: CountryRisk): number {
  return Math.min(1, Math.max(Math.abs(r.t2mZ), Math.abs(r.tpZ)) / 0.5);
}

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

const INFO_CONTENT = (
  <div>
    <p className="font-medium text-white mb-1">Composite Climate Impact</p>
    <p className="text-slate-300 text-xs leading-relaxed">
      Normalized score (0&ndash;1) combining |temperature z-score| and |precipitation z-score|.
      Pill indicates risk level: HIGH / MEDIUM / LOW.
    </p>
  </div>
);

export default function CompositeImpactTable() {
  const risks = forecastData.regionalRisk["JJA mean"] ?? [];

  const handleDownload = useCallback(() => {
    const rows = risks.map((r) => ({
      Country: r.country,
      "Temp Anomaly (σ)": r.t2mZ,
      "Precip Anomaly (σ)": r.tpZ,
      "Composite Index": compositeIndex(r),
      "Impact Level": r.risk,
    }));
    downloadCsv(rows, `composite-impact-${Date.now()}.csv`);
  }, [risks]);

  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
            COMPOSITE CLIMATE IMPACT
          </p>
          <p className="mt-0.5 text-xs font-medium text-slate-400">
            Temperature + Precipitation
          </p>
        </div>
        <Tooltip content={INFO_CONTENT} position="left">
          <button className={ICON_BTN} aria-label="Panel info">
            <Info size={14} />
          </button>
        </Tooltip>
        <Tooltip content="Download data as CSV" position="left">
          <button className={ICON_BTN} aria-label="Download data" onClick={handleDownload}>
            <Download size={14} />
          </button>
        </Tooltip>
      </div>

      <div className="mt-4 overflow-x-auto max-w-full">
        <table className="w-full text-left text-xs">
          <colgroup>
            <col style={{ width: '22%' }} />
            <col style={{ width: '19%' }} />
            <col style={{ width: '19%' }} />
            <col style={{ width: '22%' }} />
            <col style={{ width: '18%' }} />
          </colgroup>
          <thead>
            <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
              <th className="pb-2 pr-1 font-medium">Country</th>
              <th className="pb-2 pr-1 font-medium whitespace-nowrap">Temp.<br />Anomaly</th>
              <th className="pb-2 pr-1 font-medium whitespace-nowrap">Precip.<br />Anomaly</th>
              <th className="pb-2 pr-1 font-medium whitespace-nowrap">Composite<br />Index</th>
              <th className="pb-2 font-medium whitespace-nowrap">Impact<br />Level</th>
            </tr>
          </thead>
          <tbody>
            {risks.map((r) => {
              const ci = compositeIndex(r);
              return (
                <tr key={r.country} className="border-b border-slate-800/50 last:border-0">
                  <td className="py-2.5 pr-1">
                    <div className="flex items-center gap-1.5">
                      <span className={`fi fi-${FLAG_CODES[r.country] ?? ""}`} />
                      <span className="font-medium text-white">{r.country}</span>
                    </div>
                  </td>
                  <td className="py-2.5 pr-1 text-slate-300 whitespace-nowrap">
                    {r.t2mZ >= 0 ? "+" : ""}{r.t2mZ.toFixed(2)}&sigma;
                  </td>
                  <td className="py-2.5 pr-1 text-slate-300 whitespace-nowrap">
                    {r.tpZ >= 0 ? "+" : ""}{r.tpZ.toFixed(2)}&sigma;
                  </td>
                  <td className="py-2.5 pr-1">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-10 overflow-hidden rounded-full bg-slate-800 flex-none">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${ci * 100}%`,
                            backgroundColor: r.color,
                          }}
                        />
                      </div>
                      <span className="text-xs text-slate-400">{ci.toFixed(2)}</span>
                    </div>
                  </td>
                  <td className="py-2.5">
                    <span className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${RISK_STYLES[r.risk]} whitespace-nowrap`}>
                      {r.risk}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[10px] leading-4 text-slate-600">
        Composite Impact Index is a normalized score (0&ndash;1) combining temperature and precipitation anomalies.
      </p>
    </div>
  );
}
