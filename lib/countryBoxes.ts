// Mirrors COUNTRY_BOXES in scripts/export_forecast_data.py -- keep in sync.
// Approximate rectangular bounding boxes (lat_min, lat_max, lon_min, lon_max),
// for coarse-grid regional averaging/overlay only, NOT real political borders.
export interface CountryBox {
  name: string;
  abbr: string;
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}

export const COUNTRY_BOXES: CountryBox[] = [
  { name: "Nepal", abbr: "NP", latMin: 26.0, latMax: 31.0, lonMin: 80.0, lonMax: 89.0 },
  { name: "India", abbr: "IN", latMin: 8.0, latMax: 28.0, lonMin: 68.0, lonMax: 88.0 },
  { name: "Bangladesh", abbr: "BD", latMin: 21.0, latMax: 27.0, lonMin: 88.0, lonMax: 93.0 },
];
