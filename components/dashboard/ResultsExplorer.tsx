"use client";

import { useMemo, useState } from "react";
import { results } from "@/data/results";
import {
    outputOptions,
    OutputType,
} from "@/types/results";
import ResultVisualization from "./ResultVisualization";

const MODEL_ID = "cnn-tcn" as const;

export default function ResultsExplorer() {
    const [selectedOutput, setSelectedOutput] = useState<OutputType>("enso-forecast");

    const availableOutputs = useMemo(() => {
        return outputOptions.filter((output) =>
        results.some((r) => r.model === MODEL_ID && r.output === output.id));
    },[]);

    const availableVariants = useMemo<string[]>(() => {
        const matchingResults = results.filter(
            (result) => result.model === MODEL_ID
            && result.output === selectedOutput
        );
        return Array.from(new Set(matchingResults.map((result)=>result.variant).filter((variant): variant is string => typeof variant === "string")));
    }, [selectedOutput]);

    const defaultVariant = useMemo(() => {
        if (availableVariants.includes("3 Months")) {
            return "3 Months";
        }
        return availableVariants[0] ?? "";
    }, [availableVariants]);

    const [selectedVariant, setSelectedVariant] = useState<string>("3 Months");

    const currentVariant = availableVariants.includes(selectedVariant) ? selectedVariant : defaultVariant;

    const selectedResult = useMemo(() => {
        return results.find((result) => {
            const baseMatch = result.model === MODEL_ID && result.output === selectedOutput;
            if (!baseMatch) return false;
            if (!availableVariants.length) return true;
            return result.variant === currentVariant;
        });
    }, [selectedOutput, currentVariant, availableVariants.length]);

    return (
        <section className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-slate-950/30 sm:p-8 lg:p-10">
            <div>
                <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400 sm:text-sm">
                    RESULTS EXPLORER
                </p>

                <h2 className="mt-2 text-2xl font-bold tracking-tight text-white sm:text-3xl">
                    Explore Verified Model Outputs
                </h2>

                <p className="mt-2 text-sm text-slate-400">
                    Model: CNN-TCN Ensemble &mdash; select an output to explore verified results.
                </p>
            </div>

            <div className="mt-7 rounded-xl border border-slate-800 bg-slate-950/45 p-4 sm:p-5">
                <div className="grid gap-4 md:grid-cols-2">
                <div>
                    <label htmlFor="output" className="mb-2 block text-sm font-medium text-slate-300">Output</label>
                    <select id="output" value={selectedOutput}
                        onChange={(event) => setSelectedOutput(event.target.value as OutputType)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none transition-colors focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
                    >
                        {availableOutputs.map((output) => (
                            <option key={output.id} value={output.id}>{output.label}</option>
                        ))}
                    </select>
                </div>

                {availableVariants.length > 0 && (
                <div>
                    <label htmlFor="variant" className="mb-2 block text-sm font-medium text-slate-300">{getVariantLabel(selectedOutput)}</label>
                    <select id="variant" value={currentVariant}
                        onChange={(event) => setSelectedVariant(event.target.value)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none transition-colors focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
                    >
                        {availableVariants.map((variant) => (
                            <option key={variant} value={variant}>{variant}</option>
                        ))}
                    </select>
                </div>
            )}
                </div>
            </div>

            <div className="mt-8">
                <ResultVisualization result={selectedResult}/>
            </div>
        </section>
    );
}

function getVariantLabel(output: OutputType) {
    if(output === "enso-forecast" || output === "training-curve") return "Lead Time";
    if(output === "temperature-anomaly" || output === "precipitation-anomaly") return "Data";
    return "View";
}