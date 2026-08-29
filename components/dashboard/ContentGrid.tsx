"use client";

import { useRef, useCallback } from "react";
import { Info, Download } from "lucide-react";
import { forecastData } from "@/data/forecast";
import { toPng } from "html-to-image";
import EnsoTrendChart from "./EnsoTrendChart";
import RegionalMaps from "./RegionalMaps";
import CompositeImpactTable from "./CompositeImpactTable";
import KeyInsights from "./KeyInsights";
import CountryInsight from "./CountryInsight";
import Tooltip from "@/components/ui/Tooltip";

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

export default function ContentGrid() {
  const defaultMonthIndex = forecastData.maps.months.length - 1;
  const chartRef = useRef<HTMLDivElement>(null);
  const mapsRef = useRef<HTMLDivElement>(null);

  const downloadChart = useCallback(async () => {
    if (!chartRef.current) return;
    try {
      const dataUrl = await toPng(chartRef.current, {
        backgroundColor: "#0f172a",
        pixelRatio: 2,
      });
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `oni-chart-${Date.now()}.png`;
      link.click();
    } catch (err) {
      console.error("Failed to export chart:", err);
    }
  }, []);

  const downloadMaps = useCallback(async () => {
    if (!mapsRef.current) return;
    try {
      const dataUrl = await toPng(mapsRef.current, {
        backgroundColor: "#0f172a",
        pixelRatio: 2,
      });
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `south-asia-maps-${Date.now()}.png`;
      link.click();
    } catch (err) {
      console.error("Failed to export maps:", err);
    }
  }, []);

  return (
    <div className="flex flex-col gap-6">
      {/* Row A: Large chart + Map panels */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 xl:grid-cols-5 gap-6 min-h-[0]">
        {/* Left: ONI Trend Chart */}
        <div className="col-span-1 sm:col-span-2 md:col-span-3 lg:col-span-3 xl:col-span-3 rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 flex flex-col min-h-[320px]">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
                ONI (Ni&ntilde;o 3.4) INDEX
              </p>
              <p className="mt-1 text-[10px] font-medium text-slate-500">
                Historical & Forecast
              </p>
            </div>
            <div className="flex gap-1.5 flex-shrink-0">
              <Tooltip content="ONI = Oceanic Ni&ntilde;o Index. 3-month running mean SST anomaly in Ni&ntilde;o 3.4 region. Solid = observed, dashed = ensemble forecast mean, shaded = &plusmn;1&sigma; spread." position="left">
                <button className={ICON_BTN} aria-label="Panel info">
                  <Info size={14} />
                </button>
              </Tooltip>
              <Tooltip content="Download chart as PNG" position="left">
                <button className={ICON_BTN} aria-label="Download chart" onClick={() => downloadChart()}>
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
                <li className="text-slate-400">ONI ≥ +0.5°C = El Niño</li>
                <li className="text-slate-400">−0.5 to +0.5 = Neutral</li>
                <li className="text-slate-400">≤ −0.5°C = La Niña</li>
              </ul>
            </div>
            <div>
              <p className="font-medium">About ONI</p>
              <p className="text-slate-400">3-month running mean SST anomaly in the Niño 3.4 region.</p>
            </div>
            <div>
              <p className="font-medium">Forecast notes</p>
              <ul className="list-disc list-inside space-y-0.5">
                <li className="text-slate-400">Forecast is an ensemble mean</li>
                <li className="text-slate-400">Shaded band is ± spread</li>
                <li className="text-slate-400">Official classification uses 3-month running average</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Right: South Asia Impact Maps */}
        <div className="col-span-1 sm:col-span-2 md:col-span-2 lg:col-span-2 xl:col-span-2 rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
                SOUTH ASIA IMPACT MAPS
              </p>
              <p className="mt-1 text-[10px] font-medium text-slate-500">
                Anomaly (z-score) | Next 3 Months
              </p>
            </div>
            <div className="flex gap-1.5 flex-shrink-0">
              <Tooltip content="Temperature/precipitation anomaly (z-score) for JJA mean. Only Nepal, India, Bangladesh have forecast data." position="left">
                <button className={ICON_BTN} aria-label="Panel info">
                  <Info size={14} />
                </button>
              </Tooltip>
              <Tooltip content="Download maps as PNG" position="left">
                <button className={ICON_BTN} aria-label="Download maps" onClick={downloadMaps}>
                  <Download size={14} />
                </button>
              </Tooltip>
            </div>
          </div>
          <div className="mt-4" ref={mapsRef}>
            <RegionalMaps
              monthIndex={defaultMonthIndex}
              showSpread={false}
              layout="vertical"
            />
          </div>
        </div>
      </div>

      {/* Row B: Three-column layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <CompositeImpactTable />
        <KeyInsights />
        <CountryInsight />
      </div>
    </div>
  );
}