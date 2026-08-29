"use client";

import { useMemo, useCallback, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import type { GeoJsonProperties } from "geojson";
import type { ColorStop } from "@/lib/colorScale";

import worldAtlasData from "world-atlas/countries-50m.json";

const worldAtlasUrl = worldAtlasData as typeof worldAtlasData & {
  objects: { countries: { geometries: Array<{ id: string; properties?: GeoJsonProperties }> } };
};

export const MAP_TEMP_SCALE: ColorStop[] = [
  [0, "#2563eb"],
  [0.5, "#f1f5f9"],
  [1, "#dc2626"],
];

export const MAP_PRECIP_SCALE: ColorStop[] = [
  [0, "#d97706"],
  [0.5, "#f1f5f9"],
  [1, "#16a34a"],
];

const SOUTH_ASIA_IDS = new Set([
  "050", // Bangladesh
  "064", // Bhutan
  "356", // India
  "524", // Nepal
  "586", // Pakistan
  "144", // Sri Lanka
]);

const DATA_IDS = new Set([
  "050", // Bangladesh
  "356", // India
  "524", // Nepal
]);

const COUNTRY_NAMES: Record<string, string> = {
  "050": "Bangladesh",
  "356": "India",
  "524": "Nepal",
};

interface ChoroplethProps {
  colorOf: (value: number) => string;
  valueOf: (geoId: string) => number | null;
  scaleStops: ColorStop[];
  scaleLabels: { low: string; high: string; unit: string };
}

export default function ChoroplethMap({
  colorOf,
  valueOf,
  scaleStops,
  scaleLabels,
}: ChoroplethProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  const geographies = useMemo(() => {
    const geos = worldAtlasUrl.objects.countries.geometries as Array<{
      id: string;
      properties?: GeoJsonProperties;
    }>;
    return geos;
  }, []);

  const handleMouseEnter = useCallback(
    (geo: { id?: string }, event: React.MouseEvent) => {
      const id = geo.id ?? "";
      setHovered(id);
      const value = valueOf(id);
      if (value !== null) {
        const name = COUNTRY_NAMES[id] ?? geographies.find((g) => g.id === id)?.properties?.name ?? id;
        setTooltip({
          x: event.clientX,
          y: event.clientY,
          text: `${name}: ${value >= 0 ? "+" : ""}${value.toFixed(2)}σ`,
        });
      }
    },
    [geographies, valueOf]
  );

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    setTooltip((prev) => (prev ? { ...prev, x: event.clientX, y: event.clientY } : null));
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHovered(null);
    setTooltip(null);
  }, []);

  const dataGeos = useMemo(
    () => geographies.filter((g) => DATA_IDS.has(g.id)),
    [geographies]
  );

  return (
    <div>
      <div className="relative">
        <ComposableMap
          projection="geoMercator"
          projectionConfig={{
            scale: 900,
            center: [80, 22],
            parallels: [20, 30],
          }}
          className="h-56 w-full rounded-lg border border-slate-800 sm:h-64"
          viewBox="0 0 800 600"
        >
          <Geographies geography={worldAtlasUrl}>
            {({ geographies: geoList }) =>
              geoList.map((geo) => {
                const id = geo.id ?? "";
                const isSouthAsia = SOUTH_ASIA_IDS.has(id);
                const isData = DATA_IDS.has(id);
                const value = isData ? valueOf(id) : null;
                const fillColor = isData && value !== null ? colorOf(value) : "#0f172a";
                const strokeColor = isSouthAsia ? "#475569" : "#1e293b";
                const isHovered = hovered === id;

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fillColor}
                    stroke={isHovered ? "#e2e8f0" : strokeColor}
                    strokeWidth={isSouthAsia ? 0.7 : 0.4}
                    style={{
                      default: { transition: "fill 0.2s, stroke 0.2s" },
                      hover: { transition: "fill 0.2s, stroke 0.2s" },
                      pressed: { transition: "fill 0.2s, stroke 0.2s" },
                    }}
                    onMouseEnter={(e) => handleMouseEnter(geo, e)}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                  />
                );
              })
            }
          </Geographies>

          {/* Name + Value labels for data countries */}
          {dataGeos.map((g) => {
            const value = valueOf(g.id);
            if (value === null) return null;
            const countryName = COUNTRY_NAMES[g.id] ?? "";
            const labelPositions: Record<string, { x: number; y: number }> = {
              "050": { x: 610, y: 310 },
              "356": { x: 400, y: 380 },
              "524": { x: 460, y: 225 },
            };
            const pos = labelPositions[g.id];
            if (!pos) return null;
            return (
              <g key={`label-${g.id}`}>
                <rect
                  x={pos.x - 50}
                  y={pos.y - 10}
                  width={100}
                  height={20}
                  rx={4}
                  fill="#0f172a"
                  fillOpacity={0.9}
                  stroke="#334155"
                  strokeWidth={0.5}
                />
                <text
                  x={pos.x - 44}
                  y={pos.y + 4}
                  className="fill-slate-200"
                  style={{ fontSize: "9px", fontWeight: 600 }}
                >
                  {countryName}
                </text>
                <text
                  x={pos.x + 44}
                  y={pos.y + 4}
                  textAnchor="end"
                  className="fill-cyan-300"
                  style={{ fontSize: "9px", fontWeight: 700 }}
                >
                  {value >= 0 ? "+" : ""}{value.toFixed(2)}σ
                </text>
              </g>
            );
          })}
        </ComposableMap>

        {/* Tooltip */}
        {tooltip && (
          <div
            className="pointer-events-none fixed z-50 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white shadow-lg"
            style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}
          >
            {tooltip.text}
          </div>
        )}
      </div>

      {/* Color scale legend */}
      <div className="mt-2 flex items-center gap-2 pl-8">
        <span className="text-[11px] text-slate-500">{scaleLabels.low}</span>
        <div
          className="h-2 flex-1 rounded-full"
          style={{
            background: `linear-gradient(to right, ${scaleStops.map(([t, c]) => `${c} ${t * 100}%`).join(", ")})`,
          }}
        />
        <span className="text-[11px] text-slate-500">{scaleLabels.high}</span>
      </div>
      <p className="mt-1 pl-8 text-[10px] text-slate-600">{scaleLabels.unit}</p>
    </div>
  );
}
