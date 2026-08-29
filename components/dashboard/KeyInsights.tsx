"use client";

import { useCallback } from "react";
import { Thermometer, CloudRain, Activity, Shield, Sparkles, Info, Download } from "lucide-react";
import { forecastData } from "@/data/forecast";
import Tooltip from "@/components/ui/Tooltip";
import { downloadJson } from "./downloadUtils";

const ICON_BTN = "flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300";

const KEY_INSIGHTS_INFO = (
  <div>
    <p className="font-medium text-white mb-1">Key Climate Insights</p>
    <p className="text-slate-300 text-xs leading-relaxed">
      Auto-generated from forecast data: temperature/precipitation anomalies and ENSO phase.
    </p>
  </div>
);

function pickIcon(text: string) {
  const t = text.toLowerCase();
  if (t.includes("temperature") || t.includes("warmer") || t.includes("cooler")) return Thermometer;
  if (t.includes("precipitation") || t.includes("drier") || t.includes("wetter")) return CloudRain;
  if (t.includes("risk")) return Shield;
  if (t.includes("el ni") || t.includes("la ni") || t.includes("enso") || t.includes("oni")) return Activity;
  return Sparkles;
}

export default function KeyInsights() {
  const insights = forecastData.insights;

  const handleDownload = useCallback(() => {
    downloadJson({ insights }, `key-insights-${Date.now()}.json`);
  }, [insights]);

  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
            KEY CLIMATE INSIGHTS
          </p>
          <p className="mt-1 text-sm font-medium text-slate-300">
            Auto-generated from forecast data
          </p>
        </div>
        <Tooltip content={KEY_INSIGHTS_INFO} position="left">
          <button className={ICON_BTN} aria-label="Panel info">
            <Info size={14} />
          </button>
        </Tooltip>
        <Tooltip content="Download insights as JSON" position="left">
          <button className={ICON_BTN} aria-label="Download insights" onClick={handleDownload}>
            <Download size={14} />
          </button>
        </Tooltip>
      </div>

      <ul className="mt-4 space-y-3">
        {insights.map((insight, i) => {
          const Icon = pickIcon(insight);
          return (
            <li key={i} className="flex gap-2.5">
              <span className="mt-0.5 flex-none text-cyan-400/70">
                <Icon size={14} />
              </span>
              <span className="text-sm leading-5 text-slate-400">
                {renderFormattedText(insight)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function renderFormattedText(text: string) {
  const BOLD_RE = /\*\*(.+?)\*\*/g;
  const parts: Array<{ text: string; bold: boolean }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = BOLD_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ text: text.slice(lastIndex, match.index), bold: false });
    }
    parts.push({ text: match[1], bold: true });
    lastIndex = BOLD_RE.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push({ text: text.slice(lastIndex), bold: false });
  }

  return parts.map((part, i) =>
    part.bold ? (
      <strong key={i} className="font-semibold text-white">{part.text}</strong>
    ) : (
      <span key={i}>{part.text}</span>
    )
  );
}
