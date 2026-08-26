import DashboardHeader from "@/components/dashboard/DashboardHeader";
import ProjectHero from "@/components/dashboard/ProjectHero";
import ResultsExplorer from "@/components/dashboard/ResultsExplorer";

export default function Home() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-slate-950">

      <DashboardHeader />

      <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 sm:py-10 lg:px-8">

        <ProjectHero />

        <ResultsExplorer />

      </div>

    </main>
  );
}