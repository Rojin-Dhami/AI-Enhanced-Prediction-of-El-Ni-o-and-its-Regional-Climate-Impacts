import raw from "./forecast.json";
import type { ForecastData } from "@/types/forecast";

export const forecastData = raw as unknown as ForecastData;
