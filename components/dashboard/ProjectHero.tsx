import { BrainCircuit, TrendingDown, ChartNoAxesCombined } from "lucide-react";

export default function ProjectHero() {
    return(
        <section className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div className="max-w-3xl">
                <div className="flex items-center gap-2 text-sm font-medium tracking-widest text-sky-400">
                    <BrainCircuit size={18}/>
                    VERIFIED PROJECT RESULTS
                </div>

                <h1 className="mt-4 text-3xl font-bold tracking-tight text-white md:text-5xl">
                    ENSO Forecasting and South Asia Climate Prediction
                </h1>

                <p className="mt-4 max-w-2xl leading-7 text-slate-400"> 
                    Comparing XGBoost baselines with a multi-task CNN-TCN model for Niño 3.4 forecasting and regional climate anomaly prediction.
                </p>
            </div>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                    <div className="flex items-center gap-2 text-slate-400">
                        <TrendingDown size={18}/>
                        Best 3-Month Test RMSE
                    </div>

                    <p className="mt-3 text-3xl font-bold text-white">
                        0.3781°C
                    </p>

                    <p className="mt-1 text-sm text-slate-500">
                        CNN-TCN
                    </p>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                    <div className="flex items-center gap-2 text-slate-400">
                        <ChartNoAxesCombined size={18}/>
                        Test Correlation
                    </div>

                    <p className="mt-3 text-3xl font-bold text-white">
                        0.9474
                    </p>

                    <p className="mt-1 text-sm text-slate-500">
                        CNN-TCN . 3-Month Lead
                    </p>
                </div>
            </div>
        </section>
    );
}