export type ColorStop = [number, string];

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Interpolate a hex color across a multi-stop scale, t in [0, 1]. */
export function scaleColor(stops: ColorStop[], t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const localT = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
      const [r0, g0, b0] = hexToRgb(c0);
      const [r1, g1, b1] = hexToRgb(c1);
      const r = Math.round(lerp(r0, r1, localT));
      const g = Math.round(lerp(g0, g1, localT));
      const b = Math.round(lerp(b0, b1, localT));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return stops[stops.length - 1][1];
}

// Muted, low-saturation scales matching the reference forecast dashboard.
export const TEMP_SCALE: ColorStop[] = [
  [0, "#3b6fa0"],
  [0.5, "#f2f0ea"],
  [1, "#b3562f"],
];
export const PRECIP_SCALE: ColorStop[] = [
  [0, "#a1752f"],
  [0.5, "#f2f0ea"],
  [1, "#3f7a5c"],
];
export const SPREAD_SCALE: ColorStop[] = [
  [0, "#f2f0ea"],
  [1, "#5b6470"],
];

/** Map a diverging value in [-max, max] to a color on a 3-stop scale. */
export function divergingColor(value: number, maxAbs: number, stops: ColorStop[]): string {
  if (!maxAbs) return scaleColor(stops, 0.5);
  const t = (value + maxAbs) / (2 * maxAbs);
  return scaleColor(stops, t);
}

/** Map a non-negative value in [0, max] to a color on a sequential scale. */
export function sequentialColor(value: number, maxVal: number, stops: ColorStop[]): string {
  if (!maxVal) return scaleColor(stops, 0);
  return scaleColor(stops, value / maxVal);
}

/** 0..100 percentile of the absolute values, used to clip outliers so the bulk of the
 * data isn't washed out near the neutral color by a few extreme cells. */
export function percentileAbs(values: number[], p: number): number {
  const abs = values.map(Math.abs).sort((a, b) => a - b);
  if (!abs.length) return 1e-6;
  const idx = Math.min(abs.length - 1, Math.floor((p / 100) * abs.length));
  return Math.max(abs[idx], 1e-6);
}

