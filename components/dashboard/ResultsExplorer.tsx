"use client";

import { useMemo, useState } from "react";
import { models, ModelId } from "@/data/models";
import {
    outputOptions,
    OutputType,
} from "@/types/results";

export default function ResultsExplorer() {
    const [selectedModel, setSelectedModel] = useState<ModelId>("cnn-tcn");
    const [selectedOutput, setSelectedOutput] = useState<OutputType>("enso-forecast");

    const selectedModelInfo = useMemo(
        () => models.find((model) => model.id === selectedModel), [selectedModel]
    );

    const availableOutputs = outputOptions.filter(
        (output) => selectedModelInfo?.outputs.includes(output.id)
    );

    function handleModelChange(modelId: ModelId) {
        const model = models.find((item) => item.id === modelId);
        setSelectedModel(modelId);
        if(model && !model.outputs.includes(selectedOutput)) {
            setSelectedOutput(model.outputs[0] as OutputType);
        }
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
                    Select a model and output type to explore results.
                </p>
            </div>

            <div className="mt-8 grid gap-5 md:grid-cols-2">
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

            <div className="mt-8 rounded-xl border border-slate-800 bg-slate-950 p-5">
                <p className="text-sm text-slate-400">Selected Model</p>

                <h3 className="mt-1 text-xl font-semibold text-white">{selectedModelInfo?.name}</h3>

                <p className="mt-2 text-sm leading-6 text-slate-500">{selectedModelInfo?.description}</p>

                <div className="mt-6 rounded-lg border border-dashed border-slate-700 p-10 text-center">
                    <p className="text-sm text-slate-500">
                        Selected output:
                    </p>

                    <p className="mt-2 text-lg font-semibold text-white">
                        {
                            outputOptions.find(
                                (output) => output.id === selectedOutput
                            )?.label
                        }
                    </p>

                    <p className="mt-4 text-sm text-slate-500">
                        The corresponding verified visualization will appear here.
                    </p>
                </div>
            </div>
        </section>
    );
}