# AI-Enhanced Prediction of El Nino and Its Impacts on South Asian Monsoon Precipitation and Temperature

AI-powered climate forecasting dashboard that monitors El Nino / La Nina conditions (ENSO) and provides deterministic regional climate impact predictions for South Asia.

Built with **Next.js 16**, **React 19**, **TypeScript**, **Tailwind CSS 4**, **Recharts**, and a **CNN-TCN** deep learning model trained on ERA5 + ORAS5 reanalysis data (1980-2025).

---

## Overview

Most existing ENSO forecasting models predict only the Nino 3.4 index itself, without translating that signal into regional impacts. No unified framework currently couples ENSO forecasting with deterministic South Asian precipitation and temperature impact assessment in a single pipeline.

This project builds that missing link -- forecasting the Nino 3.4 index and using it to condition a regional impact assessment module for South Asia.

### Key Results

| Metric | Value |
|--------|-------|
| 3-Month Lead Correlation | **0.92** |
| 3-Month Lead RMSE | **0.51 C** |
| 6-Month Lead Correlation | **0.82** |
| Spatial Correlation (Temperature) | **0.30** |

The model correctly captured phase transitions of major historical events -- the 2020-2022 triple-dip La Nina and the 2023-2024 Super El Nino -- without dampening peak amplitude.

---

## Dashboard Routes

The app runs at `http://localhost:3000` and redirects to `/dashboard`. Eight routes are available via the sidebar:

| Route | Page | Description |
|-------|------|-------------|
| `/dashboard` | Overview | KPI strip (5 cards), ONI trend chart, South Asia maps, composite risk table, key insights, country detail |
| `/dashboard/enso-oni` | ENSO / ONI | Detailed historical & forecast ONI line chart with uncertainty band, stat strip, and legend |
| `/dashboard/south-asia-impact` | South Asia Impact | Composite impact table (per-country risk) + auto-generated climate insights |
| `/dashboard/maps` | Maps | Full-size temperature and precipitation anomaly choropleth maps for South Asia |
| `/dashboard/country-insights` | Country Insights | Per-country detail cards with gauge arc, anomaly stats, and impact interpretation |
| `/dashboard/data-sources` | Data & Sources | Downloadable datasets: ONI historical (CSV), ensemble forecast (CSV), temperature/precipitation grids (JSON), regional risk (CSV) |
| `/dashboard/model-info` | Model Info | CNN-TCN architecture details, performance metrics, training curves, and verified result images |
| `/dashboard/about` | About | Project description, objectives, methodology, key results, limitations, and team information |

---

## Getting Started

### Prerequisites

- **Node.js 18+** (for the Next.js dashboard)
- **Python 3.10+** with `numpy`, `pandas` (only needed for the data pipeline)

### Install and Run

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open in browser
open http://localhost:3000
```

### Other Commands

```bash
npm run build    # Production build
npm run start    # Start production server
npm run lint     # Run ESLint
```

---

## Data Pipeline

The dashboard consumes a pre-generated `data/forecast.json` file. To regenerate it from raw model outputs:

```bash
# 1. Place raw artifacts in forecast_outputs/
#    - forecast_meta.json
#    - ensemble_oni_2026.csv
#    - ensemble_maps_2026.npz

# 2. Run the export script
python scripts/export_forecast_data.py
```

This reads the raw NumPy/CSV artifacts, computes regional risk scores for Nepal, India, and Bangladesh, generates narrative insights, and outputs:

- `data/forecast.json` -- the single JSON payload consumed by all dashboard components
- `public/forecast/` -- copies of raw artifacts for direct download links

---

## Project Structure

```
.
├── app/
│   ├── layout.tsx                 # Root layout (Geist fonts, metadata, dark theme)
│   ├── page.tsx                   # Redirects to /dashboard
│   ├── globals.css                # Tailwind v4 + dark theme styles
│   └── dashboard/
│       ├── layout.tsx             # Dashboard shell (sidebar + top header)
│       ├── page.tsx               # Overview: KPIs + ContentGrid
│       ├── enso-oni/page.tsx      # Detailed ONI chart
│       ├── south-asia-impact/     # Impact table + insights
│       ├── maps/page.tsx          # Full-size regional maps
│       ├── country-insights/      # Per-country detail cards
│       ├── data-sources/page.tsx  # Dataset downloads
│       ├── model-info/page.tsx    # Model architecture + results
│       └── about/page.tsx         # Project description + team
│
├── components/
│   ├── dashboard/
│   │   ├── Sidebar.tsx            # Responsive nav (mobile + desktop)
│   │   ├── TopHeader.tsx          # Sticky header with hamburger
│   │   ├── KpiStrip.tsx           # 5-card KPI row
│   │   ├── ContentGrid.tsx        # Main dashboard grid layout
│   │   ├── EnsoTrendChart.tsx     # Historical-to-forecast ONI line chart
│   │   ├── EnsoBarChart.tsx       # Ensemble ONI bar chart
│   │   ├── RegionalMaps.tsx       # Side-by-side temp/precip maps
│   │   ├── ChoroplethMap.tsx      # Interactive SVG choropleth
│   │   ├── CompositeImpactTable.tsx  # Per-country risk table
│   │   ├── KeyInsights.tsx        # Auto-generated insight bullets
│   │   ├── CountryInsight.tsx     # Country detail card with gauge
│   │   ├── RegionalRiskCards.tsx  # 3-country risk card grid
│   │   ├── ForecastSection.tsx    # Full forecast view
│   │   ├── ProjectHero.tsx        # Hero section with stats
│   │   ├── ResultsExplorer.tsx    # Interactive result browser
│   │   ├── ResultVisualization.tsx # Single result image viewer
│   │   ├── FormattedText.tsx      # Safe bold-text renderer
│   │   └── downloadUtils.ts       # PNG/CSV/JSON export helpers
│   └── ui/
│       └── Tooltip.tsx            # Portal-based tooltip component
│
├── data/
│   ├── forecast.json              # Core forecast payload (~9K lines)
│   ├── forecast.ts                # TypeScript re-export
│   ├── models.ts                  # Model metadata
│   └── results.ts                 # Verified result image catalog
│
├── lib/
│   ├── colorScale.ts              # Diverging/sequential color interpolation
│   └── countryBoxes.ts            # Lat/lon bounding boxes for regional averaging
│
├── scripts/
│   └── export_forecast_data.py    # Python pipeline: raw artifacts -> forecast.json
│
├── forecast_outputs/              # Raw ML pipeline outputs (.npz, .csv, .json)
├── public/
│   ├── forecast/                  # Downloadable forecast files
│   └── results/cnn-tcn/           # Verified model result images
│
└── types/
    ├── forecast.ts                # ForecastData, OniRow, MapsData, etc.
    └── results.ts                 # OutputType, OutputOption
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | Next.js 16.3.2 | App router, static generation, file-based routing |
| UI | React 19.2.8 | Component rendering |
| Language | TypeScript 5 | Type safety |
| Styling | Tailwind CSS 4 | Utility-first CSS, dark theme |
| Charts | Recharts 3.10.1 | Line, bar, area, and composed charts |
| Maps | react-simple-maps + world-atlas | SVG choropleth maps |
| Icons | lucide-react | Icon library |
| Flags | flag-icons | Country flag emojis |
| Export | html-to-image | PNG screenshot export |
| Data pipeline | Python (NumPy, Pandas) | Converts model outputs to dashboard JSON |

---

## Model Details

- **Architecture:** CNN-TCN (Convolutional Neural Network + Temporal Convolutional Network) multi-task ensemble
- **Ensemble:** 5 members (seeds 0-4)
- **Training data:** ERA5 + ORAS5 reanalysis, 1980-2018 climatology
- **Input:** 4D tensors (12 months x 30 lat x 100 lon x 6 channels)
- **Forecast lead:** 3 months (with 6-month evaluation)
- **Regional coverage:** Nepal, India, Bangladesh (coarse-grid bounding box averaging)
- **Compared against:** XGBoost with ANOVA-based feature selection

---

## Team & Supervision

**Team:** Biraj Adhikari, Raman Shrestha, Rojin Dhami, Sandeep Khadka -- Computer Engineering, Thapathali Campus, IOE, Tribhuvan University

**Supervised by:** Asst. Prof. Kobid Karkee, Department of Electronics and Computer Engineering

Submitted August 2026, as a minor project for the Bachelor's degree.
