const metrics = [
    {
        label: "Test RMSE",
        value: "0.53°C",
        description: "Prediction error",
    },
    {
        label: "Test Correlation",
        value: "0.92",
        description: "Observed vs Predicted relationship",
    },
    {
        label: "Evaluation",
        value: "Historical",
        description: "Chronological holdout testing",
    },
];

export default function ModelPerformance() {
    return(
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div>
                <p className="text-sm font-medium tracking-wider text-slate-400">
                    MODEL VALIDATION
                </p>

                <h2 className="mt-1 text-2xl font-bold text-white">
                    Baseline Performance
                </h2>

                <p className="mt-2 text-sm text-slate-400">
                    Performance metrics from historical model evaluation.
                </p>
            </div>

            <div className="mt-8 grid gap-5 md:grid-cols-3">
                {metrics.map((metric)=>(
                    <div key={metric.label} className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                        <p className="text-sm text-slate-400">{metric.label}</p>
                        <p className="mt-3 text-3xl font-bold text-white">{metric.value}</p>
                        <p className="mt-2 text-sm text-slate-500">{metric.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}