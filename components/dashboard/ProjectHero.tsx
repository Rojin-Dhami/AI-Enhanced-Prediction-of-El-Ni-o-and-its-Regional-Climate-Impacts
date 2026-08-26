import { BrainCircuit, TrendingDown, ChartNoAxesCombined } from "lucide-react";

export default function ProjectHero() {
    return (
        <section className="relative overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 sm:p-8 lg:p-10">
            <div className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-cyan-400/5 blur-3xl" />
            <div className="relative grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(25rem,0.85fr)] lg:items-end lg:gap-12">
                <div className="max-w-3xl">
                    <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.18em] text-cyan-400 sm:text-sm">
                        <BrainCircuit size={18}/>
                        VERIFIED PROJECT RESULTS
                    </div>

                    <h1 className="mt-4 max-w-3xl text-3xl font-bold leading-[1.08] tracking-tight text-white sm:text-4xl lg:text-5xl">
                        ENSO Forecasting and South Asia Climate Prediction
                    </h1>

                    <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
                        Comparing XGBoost baselines with a multi-task CNN-TCN model for Niño 3.4 forecasting and regional climate anomaly prediction.
                    </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
                <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 sm:p-6">
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                        <TrendingDown size={18}/>
                        Best 3-Month Test RMSE
                    </div>

                    <p className="mt-3 text-3xl font-bold tracking-tight text-white sm:text-4xl">
                        0.3781°C
                    </p>

                    <p className="mt-1 text-sm text-slate-500">
                        CNN-TCN
                    </p>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 sm:p-6">
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                        <ChartNoAxesCombined size={18}/>
                        Test Correlation
                    </div>

                    <p className="mt-3 text-3xl font-bold tracking-tight text-white sm:text-4xl">
                        0.9474
                    </p>

                    <p className="mt-1 text-sm text-slate-500">
                        CNN-TCN · 3-Month Lead
                    </p>
                </div>
                </div>
            </div>
        </section>
    );
}