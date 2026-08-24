import DashboardHeader from "@/components/dashboard/DashboardHeader";
import StatusCards from "@/components/dashboard/StatusCards";
import ENSOForecastChart from "@/components/dashboard/ENSOForecastChart";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950">

      <DashboardHeader />

      <section className="mx-auto max-w-7xl px-6 py-10">

        <div className="mb-10">
          <h2 className="text-3xl font-bold text-white">
          Climate Forecast Dashboard
          </h2>

          <p className="mt-2 text-slate-400">
            Monitoring ENSO conditions and forecasting potential climate impacts across South Asia.
          </p>
        </div>

        <StatusCards />

        <ENSOForecastChart />

      </section>

    </main>
  );
}