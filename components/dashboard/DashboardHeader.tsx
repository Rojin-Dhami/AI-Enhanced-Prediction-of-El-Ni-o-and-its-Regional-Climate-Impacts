export default function DashboardHeader() {
    return (
        <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-4 sm:px-6 sm:py-5 lg:px-8">
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <div className="flex items-center gap-3">
                            <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10 text-sm font-bold text-cyan-300">EI</span>
                            <h1 className="text-lg font-bold tracking-tight text-white sm:text-xl">ENSO Insight</h1>
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-xs text-slate-400 sm:text-sm">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                            Forecast System
                        </div>
                    </div>

                    <p className="max-w-52 text-right text-xs leading-5 text-slate-400 sm:max-w-none sm:text-sm">
                        AI-Powered Climate Forecasting for South Asia
                    </p>
                </div>
            </div>
        </header>
    );
}