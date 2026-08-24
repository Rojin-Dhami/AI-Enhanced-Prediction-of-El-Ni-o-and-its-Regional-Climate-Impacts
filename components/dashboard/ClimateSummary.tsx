import {
    CloudSun,
    Droplets,
    Thermometer,
} from "lucide-react";

export default function ClimateSummary() {
    return(
        <section className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
                <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-slate-800 p-2">
                        <CloudSun size={22} className="text-slate-200"/>
                    </div>

                    <div>
                        <p className="text-sm font-medium tracking-wider text-slate-400">
                            AI Climate Summary
                        </p>

                        <h2 className="text-xl font-bold text-white">
                            What does this mean?
                        </h2>
                    </div>
                </div>

                <p className="mt-6 text-lg leading-8 text-slate-300">
                    The model indicates a developing warm ENSO signal over the forecast period.
                    Parts of South Asia may experience warmer-than-normal conditions, while precipitation patterns could vary across the region.
                </p>

                <p className="mt-4 text-sm leading-6 text-slate-500">
                    This is a model-based seasonal outlook and should be interpreted as a predicted climate anomaly rather than a guaranteed local event.
                </p>
            </div>

            <div className="grid gap-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                    <div className="flex items-center gap-3">
                        <Droplets size={22} className="text-slate-300"/>
                        <div>
                            <p className="font-semibold text-white">
                                Precipitation
                            </p>

                            <p className="mt-1 text-sm text-slate-400">
                                Spatial variations expected across South Asia
                            </p>
                        </div>
                    </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                    <div className="flex items-center gap-3">
                        <Thermometer size={22} className="text-slate-300"/>
                        
                        <div>
                            <p className="font-semibold text-white">
                                Temperature
                            </p>

                            <p className="mt-1 text-sm text-slate-400">
                                Warmer-than-normal conditions indicated in parts of the region.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}