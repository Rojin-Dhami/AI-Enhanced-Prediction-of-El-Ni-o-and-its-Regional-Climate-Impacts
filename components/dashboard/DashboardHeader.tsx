export default function DashboardHeader() {
    return(
        <header className="border-b border-slate-800 bg-slate-950">
            <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

                <div>
                    <h1 className="text-xl font-bold tracking-tight text-white">ENSO Insight</h1>
                </div>

                <p className="text-sm text-slate-400">AI-Powered Climate Forecasting for South Asia</p>
            </div>

            <div className="hidden items-center gap-2 text-sm text-slate-400 md:flex">
                <span className="h-2 w-2 rounded-full bg-green-400"></span>
                Forecast System
            </div>
        </header>
    );
}