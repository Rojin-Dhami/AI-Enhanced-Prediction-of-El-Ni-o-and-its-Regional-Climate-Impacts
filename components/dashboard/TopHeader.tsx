import { Menu } from "lucide-react";

interface TopHeaderProps {
  onMenuToggle: () => void;
}

export default function TopHeader({ onMenuToggle }: TopHeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur">
      <div className="flex items-center gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <button
          onClick={onMenuToggle}
          aria-label="Open navigation menu"
          className="md:hidden flex h-8 w-8 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300"
        >
          <Menu size={16} />
        </button>
        <div>
          <h1 className="text-sm font-bold tracking-tight text-white sm:text-base">
            EL NI&Ntilde;O &amp; SOUTH ASIA CLIMATE FORECAST
          </h1>
          <p className="mt-0.5 text-xs text-slate-400">
            Monitoring ENSO conditions and detailed climate impact for South Asia
          </p>
        </div>
      </div>
    </header>
  );
}
