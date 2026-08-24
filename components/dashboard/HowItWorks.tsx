import {
    Database,
    BrainCircuit,
    ChartNoAxesCombined,
    Map,
    Icon,
} from "lucide-react";

const steps = [
    {
        title: "Climate Data",
        description: "ERA5 and ORAS5 climate datasets",
        icon: Database,
    },
    {
        title: "AI Analysis",
        description: "Spatio-temporal forecasting model",
        icon: BrainCircuit,
    },
    {
        title: "ENSO Forecast",
        description: "Niño 3.4 prediction",
        icon: ChartNoAxesCombined,
    },
    {
        title: "Regional Outlook",
        description: "South Asia climate anomalies",
        icon: Map,
    },
];

export default function HowItWorks() {
    return(
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div>
                <p className="text-sm font-medium tracking-wider text-slate-400">
                    HOW IT WORKS
                </p>

                <h2 className="mt-1 text-2xl font-bold text-white">
                    From Climate Data to Regional Outlook
                </h2>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                {steps.map((step,index)=>{
                    const Icon = step.icon;
                    return(
                        <div key={step.title} className="relative">
                            <div className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800">
                                    <Icon size={20} className="text-slate-200"/>
                                </div>

                                <p className="mt-5 font-semibold text-white">
                                    {step.title}
                                </p>

                                <p className="mt-2 text-sm leading-6 text-slate-500">
                                    {step.description}
                                </p>
                            </div>

                            {index < steps.length - 1 && (
                                <div className="absolute -right-3 top-1/2 hidden h-px w-6 bg-slate-700 lg:block"/>
                            )}
                        </div>
                    );
                })}
            </div>
        </section>
    );
}