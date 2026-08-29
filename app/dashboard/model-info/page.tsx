import { forecastData } from "@/data/forecast";
import { models } from "@/data/models";
import { results } from "@/data/results";

export default function ModelInfoPage() {
  const { meta } = forecastData;
  const model = models[0];

  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 sm:p-8">
      <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
        MODEL INFO
      </p>
      <h2 className="mt-3 text-xl font-bold tracking-tight text-white">
        {model.name}
      </h2>
      <p className="mt-2 text-sm text-slate-400">
        {model.description}
      </p>

      <div className="mt-6 space-y-6 text-sm leading-6 text-slate-400">
        {/* Architecture */}
        <section>
          <h3 className="text-base font-semibold text-white">Architecture</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              <span className="text-slate-300">Type:</span> Multi-task spatio-temporal neural network
            </li>
            <li>
              <span className="text-slate-300">Components:</span> Convolutional Neural Network (CNN) + Temporal Convolutional Network (TCN)
            </li>
            <li>
              <span className="text-slate-300">Tasks:</span> ENSO forecasting + South Asia temperature/precipitation anomaly prediction
            </li>
            <li>
              <span className="text-slate-300">Ensemble:</span> {meta.nMembers} members (seeds 0&ndash;{meta.nMembers - 1})
            </li>
          </ul>
        </section>

        {/* Performance */}
        <section>
          <h3 className="text-base font-semibold text-white">Performance</h3>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">Best 3-Month Test RMSE</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">
                0.3781&deg;C
              </p>
              <p className="mt-1 text-sm text-slate-500">CNN-TCN</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">Test Correlation</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">
                0.9474
              </p>
              <p className="mt-1 text-sm text-slate-500">CNN-TCN &middot; 3-Month Lead</p>
            </div>
          </div>
        </section>

        {/* Training Details */}
        <section>
          <h3 className="text-base font-semibold text-white">Training Details</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              <span className="text-slate-300">Training climatology:</span> 1980&ndash;2018 baseline period
            </li>
            <li>
              <span className="text-slate-300">Input sequence length:</span> {meta.seqLen} months
            </li>
            <li>
              <span className="text-slate-300">Forecast lead:</span> {meta.leadMonths} months
            </li>
            <li>
              <span className="text-slate-300">Region:</span> South Asia (lat {meta.grid.latRange[0]}&deg;&ndash;{meta.grid.latRange[1]}&deg;N, lon {meta.grid.lonRange[0]}&deg;&ndash;{meta.grid.lonRange[1]}&deg;E)
            </li>
            <li>
              <span className="text-slate-300">Compared against:</span> XGBoost baselines
            </li>
          </ul>
        </section>

        {/* Model Results */}
        <section>
          <h3 className="text-base font-semibold text-white">Model Results</h3>
          <p className="mt-1 text-xs text-slate-500">
            Actual vs. predicted outputs and training curves from the CNN-TCN model.
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {results.filter((r) => r.image).map((result) => (
              <div key={result.id} className="rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden">
                <div className="p-3">
                  <p className="text-xs font-medium text-white">{result.title}</p>
                  <p className="mt-0.5 text-[10px] text-slate-500">{result.description}</p>
                  {result.variant && (
                    <span className="mt-1.5 inline-block rounded-full border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400">
                      {result.variant}
                    </span>
                  )}
                </div>
                <img
                  src={result.image}
                  alt={result.title}
                  className="w-full border-t border-slate-800"
                  loading="lazy"
                />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
