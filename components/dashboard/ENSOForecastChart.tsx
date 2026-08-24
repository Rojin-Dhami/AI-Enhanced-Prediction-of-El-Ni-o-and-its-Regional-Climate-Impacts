"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

import {ensoData} from "@/data/enso-data";

export default function ENSOForecastChart() {
    return(
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
                <div>
                    <p className="text-sm text-slate-400">
                        ENSO Forecast
                    </p>
                    <h2 className="mt-1 text-2xl font-bold text-white">
                        Niño 3.4 Forecast
                    </h2>
                    <p className="mt-2 text-sm text-slate-400">
                        Historical observations and AI-based forecast
                    </p>
                </div>

                <div className="rounded-lg border border-slate-700 bg-slate-950 px-4 py-2 text-sm text-slate-400">
                    Forecast Horizon:{" "}
                    <span className="font-semibold text-white">6 months</span>
                </div>
            </div>

            <div className="mt-8 h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={ensoData} margin={{top:10, right:20, left:-10, bottom: 10}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
                        <XAxis dataKey="date" stroke="#94a3b8" tick={{fontSize: 12}}/>
                        <YAxis stroke="#94a3b8" tick={{fontSize: 12}} domain={[-1.5,1.5]} 
                            label={{
                                value: "Niño 3.4 Anomaly (°C)",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#94a3b8",
                                fontSize: 12,
                            }}
                        />
                        <Tooltip contentStyle={{backgroundColor: "#020617", border: "1px solid #334155", borderRadius: "10px"}} labelStyle={{color:"#e2e9f0"}}/>
                        <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4"/>
                        <ReferenceLine x="Jun 2025" stroke="#64748b" strokeDasharray="5 5"
                            label={{
                                value: "Forecast Start",
                                position: "top",
                                fill: "#94a3b8",
                                fontSize: 12,
                            }}
                        />
                        <Line type="monotone" dataKey="observed" stroke="#38bdf8" strokeWidth={3} dot={false} connectNulls={false} name="Observed"/>
                        <Line type="monotone" dataKey="predicted" stroke="#f59e0b" strokeWidth={3} strokeDasharray="8 5" dot={false} connectNulls name="Predicted"/>
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap gap-6 text-sm text-slate-400">
                <div className="flex items-center gap-2">
                    <span className="h-[3px] w-8 rounded-full bg-sky-400"/>
                    Historical Observation
                </div>
                
                <div className="flex items-center gap-2">
                    <span className="h-[3px] w-8 rounded-full bg-amber-400"/>
                    AI Forecast
                </div>

                <div className="flex items-center gap-2">
                    <span className="h-[2px] w-8 border-t border-dashed border-slate-500"/>
                    Neutral Reference
                </div>
            </div>
        </section>
    );
}