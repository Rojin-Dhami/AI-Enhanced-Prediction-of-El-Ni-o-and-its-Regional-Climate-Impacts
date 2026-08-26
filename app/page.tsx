import DashboardHeader from "@/components/dashboard/DashboardHeader";
import ProjectHero from "@/components/dashboard/ProjectHero";
import ResultsExplorer from "@/components/dashboard/ResultsExplorer";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950">

      <DashboardHeader />

      <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">

        <ProjectHero/>

        <ResultsExplorer/>

      </div>

    </main>
  );
}