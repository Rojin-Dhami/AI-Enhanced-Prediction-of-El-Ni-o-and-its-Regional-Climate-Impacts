"use client";

import { useRef, useCallback } from "react";
import { Download, Info } from "lucide-react";
import KpiStrip from "@/components/dashboard/KpiStrip";
import EnsoTrendChart from "@/components/dashboard/EnsoTrendChart";
import Tooltip from "@/components/ui/Tooltip";
import { downloadPng } from "@/components/dashboard/downloadUtils";

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

export default function EnsoOniPage() {
  const chartRef = useRef<HTMLDivElement>(null);

  const handleDownload = useCallback(() => {
    if (chartRef.current) {
      downloadPng(chartRef.current, `oni-chart-${Date.now()}.png`);
    }
  }, []);

  return (
    <>
      <KpiStrip />
      <div className="grid grid-cols-1 min-h-[0]">
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 flex flex-col h-[450px]">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
                ONI (Ni&ntilde;o 3.4) INDEX
              </p>
              <p className="mt-1 text-[10px] font-medium text-slate-500">
                Historical & Forecast &mdash; Detailed View
              </p>
            </div>
            <div className="flex gap-1.5 flex-shrink-0">
              <Tooltip content="ONI = Oceanic Ni&ntilde;o Index. 3-month running mean SST anomaly in Ni&ntilde;o 3.4 region." position="left">
                <button className={ICON_BTN} aria-label="Panel info">
                  <Info size={14} />
                </button>
              </Tooltip>
              <Tooltip content="Download chart as PNG" position="left">
                <button className={ICON_BTN} aria-label="Download chart" onClick={handleDownload}>
                  <Download size={14} />
                </button>
              </Tooltip>
            </div>
          </div>
          <div className="flex-1 overflow-hidden" ref={chartRef}>
            <EnsoTrendChart />
          </div>
          <div className="flex items-start gap-6 text-[10px] text-slate-500 pt-3">
            <div>
              <p className="font-medium">How to read</p>
              <ul className="list-disc list-inside space-y-0.5">
                <li className="text-slate-400">ONI &ge; +0.5&deg;C = El Ni&ntilde;o</li>
                <li className="text-slate-400">&minus;0.5 to +0.5 = Neutral</li>
                <li className="text-slate-400">&le; &minus;0.5&deg;C = La Ni&ntilde;a</li>
              </ul>
            </div>
            <div>
              <p className="font-medium">About ONI</p>
              <p className="text-slate-400">3-month running mean SST anomaly in the Ni&ntilde;o 3.4 region.</p>
            </div>
            <div>
              <p className="font-medium">Forecast notes</p>
              <ul className="list-disc list-inside space-y-0.5">
                <li className="text-slate-400">Forecast is an ensemble mean</li>
                <li className="text-slate-400">Shaded band is &plusmn; spread</li>
                <li className="text-slate-400">Official classification uses 3-month running average</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
