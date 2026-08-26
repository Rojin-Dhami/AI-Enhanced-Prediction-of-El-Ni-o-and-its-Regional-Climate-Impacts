import DashboardHeader from "@/components/dashboard/DashboardHeader";


export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950">

      <DashboardHeader />

      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">

        <section className="mb-10">
          <p className="text-sm font-medium tracking-widest text-sky-400">
            CLIMATE INTELLIGENCE
          </p>

          <h1 className="mt-3 max-w-3xl text-4xl font-bold tracking-tight text-white md:text-5xl">
            AI-Powered ENSO Forecasting for South Asia
          </h1>

          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400">
            Monitoring ENSO conditions and forecasting potential climate impacts on South Asian precipitation and temperature.
          </p>
        </section>

        <div className="space-y-8">

          {/* <StatusCards/>

          <ENSOForecastChart/>

          <ClimateOutlook/>

          <ClimateSummary/>

          <ModelPerformance/>

          <HowItWorks/> */}

        </div>

      </div>

    </main>
  );
}