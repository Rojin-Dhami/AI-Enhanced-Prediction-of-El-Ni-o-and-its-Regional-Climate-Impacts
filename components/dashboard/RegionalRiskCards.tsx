import { AlertTriangle, CloudRain, Thermometer } from "lucide-react";
import { CountryRisk } from "@/types/forecast";

const RISK_STYLES: Record<CountryRisk["risk"], string> = {
  HIGH: "border-[#a8502f]/50 bg-[#a8502f]/10 text-[#e59a80]",
  MEDIUM: "border-[#c9932f]/50 bg-[#c9932f]/10 text-[#e5c079]",
  LOW: "border-[#4d7a5c]/50 bg-[#4d7a5c]/10 text-[#8fbf9f]",
};

export default function RegionalRiskCards({ risks }: { risks: CountryRisk[] }) {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {risks.map((r) => (
        <div key={r.country} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 sm:p-5">
          <div className="flex items-center justify-between gap-2">
            <p className="font-semibold text-white">{r.country}</p>
            <span className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${RISK_STYLES[r.risk]}`}>
              <AlertTriangle size={12} />
              {r.risk}
            </span>
          </div>
          <div className="mt-3 space-y-1.5 text-sm text-slate-400">
            <p className="flex items-center gap-2">
              <Thermometer size={14} className="text-slate-500" />
              Temperature: <span className="font-medium text-slate-200">{`${r.t2mZ >= 0 ? "+" : ""}${r.t2mZ.toFixed(2)}\u03c3`}</span>
            </p>
            <p className="flex items-center gap-2">
              <CloudRain size={14} className="text-slate-500" />
              Precipitation: <span className="font-medium text-slate-200">{`${r.tpZ >= 0 ? "+" : ""}${r.tpZ.toFixed(2)}\u03c3`}</span>
            </p>
          </div>
          <p className="mt-2 text-[11px] leading-4 text-slate-600">Approximate regional average, not a political boundary.</p>
        </div>
      ))}
    </div>
  );
}
