export interface ForecastMeta {
  generated: string;
  seasonLabel: string;
  nMembers: number;
  leadMonths: number;
  seqLen: number;
  pacDataMax: string;
  skipped: string[];
  grid: {
    hs: number;
    ws: number;
    latRange: [number, number];
    lonRange: [number, number];
  };
}

export interface OniRow {
  month: string;
  seeds: number[];
  ensMean: number;
  ensStd: number;
}

export interface HistoricalOniPoint {
  time: string;
  value: number;
}

export interface MapsData {
  months: string[];
  lats: number[];
  lons: number[];
  t2mMean: number[][][];
  t2mStd: number[][][];
  tpMean: number[][][];
  tpStd: number[][][];
}

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";

export interface CountryRisk {
  country: string;
  t2mZ: number;
  tpZ: number;
  risk: RiskLevel;
  color: string;
}

export interface ForecastData {
  meta: ForecastMeta;
  oni: { months: string[]; rows: OniRow[] };
  historical: HistoricalOniPoint[];
  maps: MapsData;
  regionalRisk: Record<string, CountryRisk[]>;
  insights: string[];
}

export const ENSO_STAGES: Array<{ threshold: number; label: string; color: string }> = [
  { threshold: 2.0, label: "Very Strong El Nino", color: "#8c2d04" },
  { threshold: 1.5, label: "Strong El Nino", color: "#b3562f" },
  { threshold: 1.0, label: "Moderate El Nino", color: "#c98a4f" },
  { threshold: 0.5, label: "Weak El Nino", color: "#d9b98a" },
  { threshold: -0.5, label: "Neutral", color: "#9a9a9a" },
  { threshold: -1.0, label: "Weak La Nina", color: "#9db8cf" },
  { threshold: -1.5, label: "Moderate La Nina", color: "#6f97b8" },
  { threshold: -Infinity, label: "Strong La Nina", color: "#3b6fa0" },
];

export function ensoCategory(oni: number): { label: string; color: string } {
  const stage = ENSO_STAGES.find((s) => oni >= s.threshold) ?? ENSO_STAGES[ENSO_STAGES.length - 1];
  return { label: stage.label, color: stage.color };
}
