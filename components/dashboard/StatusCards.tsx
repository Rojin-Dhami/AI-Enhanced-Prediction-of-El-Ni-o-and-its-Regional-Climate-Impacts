const cards = [
    {
        title: "Current ENSO State",
        value: "Neutral",
        description: "Niño 3.4: +0.12 °C",
    },
    {
        title: "AI Forecast",
        value: "El Niño",
        description: "Predicted ENSO condition",
    },
    {
        title: "Forecast Horizon",
        value: "6 months",
        description: "Seasonal Outlook",
    },
];

export default function StatusCards() {
    return(
        <section className="grid gap-6 md:grid-cols-3">
            {cards.map((card) => (
                <div key={card.title} className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                    <p className="text-sm text-slate-400">{card.title}</p>
                    <h3 className="mt-3 text-2xl font-bold text-white">{card.value}</h3>
                    <p className="mt-2 text-sm text-slate-500">{card.description}</p>
                </div>
            ))}
        </section>
    );
}