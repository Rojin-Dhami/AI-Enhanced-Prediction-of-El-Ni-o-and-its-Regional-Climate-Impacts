export default function ENSOForecastChart() {
    return(
        <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <div>
                <p className="text-sm text-slate-400">
                    ENSO Forecast
                </p>
                <h2 className="mt-1 text-2xl font-bold text-white">
                    Niño 3.4 Forecast
                </h2>
            </div>

            <div className="mt-6 flex h-80 items-center justify-center rounded-xl border border-dashed border-slate-700">
                <p className="text-slate-500">
                    Forecast visualization will appear here
                </p>
            </div>
        </section>
    );
}