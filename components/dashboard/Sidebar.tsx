"use client";

import { useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutDashboard,
  Activity,
  Map,
  Globe,
  MapPin,
  FileText,
  Info,
  Users,
  X,
} from "lucide-react";
import { forecastData } from "@/data/forecast";

const NAV_ITEMS = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/dashboard" },
  { icon: Activity, label: "ENSO / ONI", path: "/dashboard/enso-oni" },
  { icon: Map, label: "South Asia Impact", path: "/dashboard/south-asia-impact" },
  { icon: Globe, label: "Maps", path: "/dashboard/maps" },
  { icon: MapPin, label: "Country Insights", path: "/dashboard/country-insights" },
  { icon: FileText, label: "Data & Sources", path: "/dashboard/data-sources" },
  { icon: Info, label: "Model Info", path: "/dashboard/model-info" },
  { icon: Users, label: "About", path: "/dashboard/about" },
] as const;

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

function SidebarNav({ onNavigate, showLabels = true }: { onNavigate: (path: string) => void; showLabels?: boolean }) {
  const pathname = usePathname();

  return (
    <ul className="space-y-0.5">
      {NAV_ITEMS.map(({ icon: Icon, label, path }) => {
        const isActive = pathname === path || pathname.startsWith(path + "/");
        return (
          <li key={label}>
            <button
              onClick={() => onNavigate(path)}
              className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                isActive
                  ? "bg-cyan-500/10 font-medium text-cyan-300"
                  : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
              }`}
            >
              <Icon size={16} strokeWidth={isActive ? 2 : 1.5} />
              <span className={showLabels ? "hidden lg:inline" : ""}>{label}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const router = useRouter();
  const sidebarRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  const generated = forecastData.meta.generated;
  const lastUpdated = generated
    ? new Date(generated).toLocaleDateString("en-US", { month: "short", year: "numeric" })
    : "";

  const handleNavigate = (path: string) => {
    router.push(path);
    onClose();
  };

  // Focus trap for mobile sidebar
  useEffect(() => {
    if (!isOpen) return;

    closeButtonRef.current?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }

      if (e.key === "Tab" && sidebarRef.current) {
        const focusable = sidebarRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Mobile sidebar (fixed overlay) */}
      <aside
        ref={sidebarRef}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation sidebar"
        className={`md:hidden fixed inset-y-0 left-0 z-50 flex w-[240px] flex-col border-r border-slate-800 bg-slate-950 transform transition-transform duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Logo / branding + close button */}
        <div className="flex items-center justify-between px-4 pt-5 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10 text-xs font-bold text-cyan-300">
              EI
            </span>
            <div>
              <p className="text-sm font-bold text-white leading-tight">El Niño</p>
              <p className="text-[10px] text-slate-500 leading-tight">South Asia Climate Forecast</p>
            </div>
          </div>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            aria-label="Close navigation sidebar"
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-400 transition-colors hover:border-cyan-500 hover:text-cyan-300"
          >
            <X size={14} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-1">
          <SidebarNav onNavigate={handleNavigate} showLabels={false} />
        </nav>

        {/* Last updated */}
        <div className="border-t border-slate-800 px-4 py-3">
          <p className="text-[10px] leading-4 text-slate-600">
            Last Updated
          </p>
          <p className="text-xs text-slate-400">{lastUpdated}</p>
        </div>
      </aside>

      {/* Desktop sidebar (static, existing behavior) */}
      <aside className="hidden md:flex w-[60px] lg:w-[200px] flex-none flex-col border-r border-slate-800 bg-slate-950 h-screen sticky top-0">
        {/* Logo / branding */}
        <div className="px-4 pt-5 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10 text-xs font-bold text-cyan-300">
              EI
            </span>
            <div className="hidden lg:block">
              <p className="text-sm font-bold text-white leading-tight">El Niño</p>
              <p className="text-[10px] text-slate-500 leading-tight">South Asia Climate Forecast</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-1">
          <SidebarNav onNavigate={handleNavigate} />
        </nav>

        {/* Last updated */}
        <div className="border-t border-slate-800 px-4 py-3 hidden lg:block">
          <p className="text-[10px] leading-4 text-slate-600">
            Last Updated
          </p>
          <p className="text-xs text-slate-400">{lastUpdated}</p>
        </div>
      </aside>
    </>
  );
}
