import Image from "next/image";
import { ResultItem } from "@/data/results";

interface ResultVisualizationProps {
    result?: ResultItem;
}

export default function ResultVisualization({result,}: ResultVisualizationProps) {
    if (!result) {
        return(
            <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950 p-12 text-center">
                <p className="text-lg font-semibold text-white">Result not available</p>
                <p className="mt-2 text-sm text-slate-500">No verified visualization is available for this selection.</p>
            </div>
        );
    }

    return(
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
            <div className="border-b border-slate-800 p-5">
                <h3 className="text-xl font-semibold text-white">
                    {result.title}
                </h3>

                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                    {result.description}
                </p>
            </div>

            {result.image && (
                <div className="relative w-full bg-white">
                    <Image src={result.image} alt={result.title} width={1600} height={900} className="h-auto w-full object-contain" priority/>
                </div>
            )}
        </div>
    );
}