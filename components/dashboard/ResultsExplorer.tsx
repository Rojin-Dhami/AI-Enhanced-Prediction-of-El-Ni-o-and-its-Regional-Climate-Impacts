"use client";

import { useMemo, useState } from "react";
import { models, ModelId } from "@/data/models";
import { results } from "@/data/results";
import {
    outputOptions,
    OutputType,
} from "@/types/results";
import ResultVisualization from "./ResultVisualization";

export default function ResultsExplorer() {
    const [selectedModel, setSelectedModel] = useState<ModelId>("cnn-tcn");
    const [selectedOutput, setSelectedOutput] = useState<OutputType>("enso-forecast");

    const selectedModelInfo = useMemo(
        () => models.find((model) => model.id === selectedModel), [selectedModel]
    );

    const availableOutputs = useMemo(() => {
        return outputOptions.filter((output) =>
        selectedModelInfo?.outputs.includes(output.id));
    },[selectedModelInfo]);

    const availableVariants = useMemo<string[]>(() => {
        const matchingResults = results.filter(
            (result) => result.model === selectedModel
            && result.output === selectedOutput
        );
        return Array.from(new Set(matchingResults.map((result)=>result.variant).filter((variant): variant is string => typeof variant === "string")));
    }, [selectedModel, selectedOutput]);

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
            const baseMatch = result.model === selectedModel && result.output === selectedOutput;
            if (!baseMatch) return false;
            if (!availableVariants.length) return true;
            return result.variant === currentVariant;
        });
    }, [selectedModel, selectedOutput, currentVariant, availableVariants.length,]);

    function handleModelChange(modelId: ModelId) {
        const model = models.find((item) => item.id === modelId);
        setSelectedModel(modelId);
        if (!model) return;
        const firstOutput = model.outputs[0] as OutputType;
        setSelectedOutput(firstOutput);
        const firstVariant = results.find((result) => result.model === modelId && result.output === firstOutput)?.variant;
        setSelectedVariant(firstVariant ?? "");
    }

    function handleOutputChange(output: OutputType) {
        setSelectedOutput(output);
        const firstVariant = results.find((result) => result.model === selectedModel && result.output === output)?.variant;
        setSelectedVariant(firstVariant ?? "");
    }

    return(
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
            <div>
                <p className="text-sm font-medium tracking-widest text-sky-400">
                    RESULTS EXPLORER
                </p>

                <h2 className="mt-2 text-2xl font-bold text-white">
                    Explore Verified Model Outputs
                </h2>

                <p className="mt-2 text-sm text-slate-400">
                    Select a model and output to explore verified results.
                </p>
            </div>

            <div className="mt-8 grid gap-5 md:grid-cols-3">
                <div>
                    <label htmlFor="model" className="mb-2 block text-sm font-medium text-slate-300">Model</label>
                    <select id="model" value={selectedModel} 
                        onChange={(event) => handleModelChange(event.target.value as ModelId)}
                        className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-sky-500"
                    >
                        {models.map((model) => (
                            <option key={model.id} value={model.id}>{model.name}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label htmlFor="output" className="mb-2 block text-sm font-medium text-slate-300">Output</label>
                    <select id="output" value={selectedOutput}
                        onChange={(event) => setSelectedOutput(event.target.value as OutputType)}
                        className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-sky-500"
                    >
                        {availableOutputs.map((output) => (
                            <option key={output.id} value={output.id}>{output.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            {availableVariants.length > 0 && (
                <div>
                    <label htmlFor="variant" className="mb-2 block text-sm font-medium text-slate-300">{getVariantLabel(selectedModel, selectedOutput)}</label>
                    <select id="variant" value={currentVariant}
                        onChange={(event) => setSelectedVariant(event.target.value)}
                        className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-sky-500"
                    >
                        {availableVariants.map((variant) => (
                            <option key={variant} value={variant}>{variant}</option>
                        ))}
                    </select>
                </div>
            )}

            <div className="mt-8">
                <p className="text-sm text-slate-400">Selected Model</p>

                <h3 className="mt-1 text-xl font-semibold text-white">{selectedModelInfo?.name}</h3>

                <p className="mt-2 text-sm leading-6 text-slate-500">{selectedModelInfo?.description}</p>
            </div>

            <div className="mt-6">
                <ResultVisualization result={selectedResult}/>
            </div>
        </section>
    );
}

function getVariantLabel(model: ModelId, output: OutputType) {
    if (model === "cnn-tcn") {
        if(output === "enso-forecast" || output === "training-curve") return "Lead Time";
        if(output === "temperature-anomaly" || output === "precipitation-anomaly") return "Data";
    }
    if (model === "xgboost-method-1" && (output === "enso-forecast" || output === "feature-importance")) return "Configuration";
    return "View";
}