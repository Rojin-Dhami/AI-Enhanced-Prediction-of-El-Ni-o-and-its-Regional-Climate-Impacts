"use client";

import {useState} from "react";
import {
    CloudRain,
    ThermometerSun,
    MapPin,
} from "lucide-react";

export default function ClimateOutlook() {
    const [activeView, setActiveView] = useState<
    "precipitation" | "temperature"
    >("precipitation");

    return(
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
                <div>
                    <p className="text-sm font-medium tracking-wider text-slate-400">
                        SOUTH ASIA CLIMATE OUTLOOK
                    </p>
                    <h2 className="mt-1 text-2xl font-bold text-white">
                        Predicted Climate Anomalies
                    </h2>
                    <p className="mt-2 text-sm text-slate-400">
                        Explore projected precipitation and temperature patterns.
                    </p>
                </div>

                <div className="flex rounded-xl border border-slate-700 bg-slate-950 p-1">
                    <button onClick={()=>setActiveView("precipitation")} className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm transition ${
                        activeView === "precipitation"
                        ? "bg-slate-800 text-white"
                        : "text-slate-400 hover:text-white"
                    }`}>
                        <CloudRain size={16}/>
                        Precipitation
                    </button>

                    <button onClick={()=>setActiveView("temperature")} className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm transition ${
                        activeView === "temperature"
                        ? "bg-slate-800 text-white"
                        : "text-slate-400 hover:text-white"
                    }`}>
                        <ThermometerSun size={16}/>
                        Temperature
                    </button>
                </div>

                <div className="mt-8 flex min-h-[420px] items-center justify-center rounded-2xl border border-dashed border-slate-700 bg-slate-950/50 p-6">
                    <div className="text-center">
                        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-800">
                            <MapPin className="text-slate-300" size={28}/>
                        </div>

                        <h3 className="mt-5 text-xl font-semibold text-white">
                            {activeView === "precipitation" ? "Precipitation Anomaly Outlook" : "Temperature Anomaly Outlook"}
                        </h3>

                        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-40">
                            The South Asia climate anomaly visualization will appear here.
                            This section will eventually display the model output spatially.
                        </p>
                    </div>
                </div>

                <div className="mt-6 flex items-center justify-between text-sm text-slate-400">
                    <span>
                        Forecast Region: South Asia
                    </span>

                    <span>
                        Forecast Period: JJAS
                    </span>
                </div>
            </div>
        </section>
    );
};