import {
    Activity,
    Sparkles,
    CalendarDays,
} from "lucide-react";

const cards = [
    {
        title: "Current ENSO State",
        value: "Neutral",
        description: "Niño 3.4: +0.12 °C",
        icon: Activity,
    },
    {
        title: "AI Forecast",
        value: "El Niño",
        description: "Predicted ENSO condition",
        icon: Sparkles,
    },
    {
        title: "Forecast Horizon",
        value: "6 months",
        description: "Seasonal Outlook",
        icon: CalendarDays,
    },
];

export default function StatusCards() {
    return(
        <section className="grid gap-6 md:grid-cols-3">
            {cards.map((card) => {
                const Icon = card.icon;
                return(
                    <div key={card.title} className="group rounded-2xl border border-slate-800 bg-slate-900 p-6 transition duration-300 hover:-translate-y-1 hover:border-slate-700">
                        <div className="flex items-start justify-between">
                            <p className="text-sm font-medium text-slate-400">{card.title}</p>
                            <div className="rounded-xl bg-slate-800 p-2">
                                <Icon size={20} className="text-slate-300"/>
                            </div>
                        </div>
                        <h3 className="mt-6 text-3xl font-bold tracking-tight text-white">{card.value}</h3>
                        <p className="mt-2 text-sm text-slate-500">{card.description}</p>
                    </div>
                );
            })}
        </section>
    );
}