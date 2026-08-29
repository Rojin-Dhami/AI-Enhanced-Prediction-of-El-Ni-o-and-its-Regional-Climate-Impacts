"use client";

import { useRef, useCallback } from "react";
import { Download, Info } from "lucide-react";
import { forecastData } from "@/data/forecast";
import RegionalMaps from "@/components/dashboard/RegionalMaps";
import Tooltip from "@/components/ui/Tooltip";
import { downloadPng } from "@/components/dashboard/downloadUtils";

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

export default function MapsPage() {
  const defaultMonthIndex = forecastData.maps.months.length - 1;
  const mapsRef = useRef<HTMLDivElement>(null);

  const handleDownload = useCallback(() => {
    if (mapsRef.current) {
      downloadPng(mapsRef.current, `south-asia-maps-${Date.now()}.png`);
    }
  }, []);

  return (
    <>
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10">
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
              SOUTH ASIA IMPACT MAPS
            </p>
            <p className="mt-1 text-[10px] font-medium text-slate-500">
              Temperature &amp; Precipitation Anomalies (z-score) | {forecastData.maps.months[defaultMonthIndex]}
            </p>
          </div>
          <div className="flex gap-1.5 flex-shrink-0">
            <Tooltip content="Temperature/precipitation anomaly (z-score) for JJA mean. Only Nepal, India, Bangladesh have forecast data." position="left">
              <button className={ICON_BTN} aria-label="Panel info">
                <Info size={14} />
              </button>
            </Tooltip>
            <Tooltip content="Download maps as PNG" position="left">
              <button className={ICON_BTN} aria-label="Download maps" onClick={handleDownload}>
                <Download size={14} />
              </button>
            </Tooltip>
          </div>
        </div>
        <div className="mt-4" ref={mapsRef}>
          <RegionalMaps
            monthIndex={defaultMonthIndex}
            showSpread={false}
            layout="horizontal"
          />
        </div>
      </div>
    </>
  );
}
