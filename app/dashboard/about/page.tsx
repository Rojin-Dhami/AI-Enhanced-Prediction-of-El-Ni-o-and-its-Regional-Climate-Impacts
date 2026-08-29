export default function AboutPage() {
  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/90 p-5 shadow-2xl shadow-cyan-950/10 sm:p-8">
      <p className="text-xs font-semibold tracking-[0.18em] text-cyan-400">
        ABOUT
      </p>
      <h2 className="mt-3 text-xl font-bold tracking-tight text-white">
        AI-Enhanced Prediction of El Ni&ntilde;o and Its Impacts on South Asian Monsoon Precipitation and Temperature
      </h2>
      <p className="mt-2 text-sm text-slate-400">
        AI-Powered Climate Forecasting for South Asia
      </p>

      <div className="mt-6 space-y-6 text-sm leading-6 text-slate-400">
        {/* Section 1 — Project Description */}
        <section>
          <h3 className="text-base font-semibold text-white">Project Description</h3>
          <p className="mt-2">
            Most existing ENSO forecasting models predict only the Ni&ntilde;o 3.4 index itself, without
            translating that signal into regional impacts. No unified framework currently couples ENSO
            forecasting with deterministic South Asian precipitation and temperature impact assessment
            in a single pipeline. This project builds that missing link &mdash; forecasting the
            Ni&ntilde;o 3.4 index and using it to condition a regional impact assessment module for
            South Asia.
          </p>
        </section>

        {/* Section 2 — Objectives */}
        <section>
          <h3 className="text-base font-semibold text-white">Objectives</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              Develop a <span className="text-slate-300">CNN-TCN model</span> to forecast the Ni&ntilde;o 3.4 index
              at 3&ndash;6 month lead time using ERA5 + ORAS5 data (1980&ndash;2025).
            </li>
            <li>
              Develop a <span className="text-slate-300">coupled impact assessment module</span> that produces
              deterministic JJAS precipitation and temperature anomaly forecasts over South Asia,
              conditioned on the predicted Ni&ntilde;o 3.4 index.
            </li>
          </ul>
        </section>

        {/* Section 3 — Methodology */}
        <section>
          <h3 className="text-base font-semibold text-white">Methodology</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              <span className="text-slate-300">Input:</span> 4D tensors (12 months &times; 30 lat &times; 100
              lon &times; 6 channels), preserving spatial topology.
            </li>
            <li>
              <span className="text-slate-300">Feature selection:</span> Two independent MIFS (Mutual Information
              Feature Selection) methods both converged on Ocean Heat Content and central-to-eastern
              equatorial Pacific SST as the dominant predictors, consistent with established climate
              science.
            </li>
            <li>
              <span className="text-slate-300">Architecture:</span> A shared spatiotemporal encoder feeding two
              task-specific heads &mdash; an ENSO Predictor Head and a South Asia Impact Head &mdash;
              trained jointly.
            </li>
            <li>
              <span className="text-slate-300">Impact module:</span> PCA target compression combined with a custom
              spatial correlation loss, designed to handle the 340-grid-cell South Asian domain.
            </li>
            <li>
              <span className="text-slate-300">Baseline comparison:</span> XGBoost with ANOVA-based feature
              selection.
            </li>
          </ul>
        </section>

        {/* Section 4 — Key Results */}
        <section>
          <h3 className="text-base font-semibold text-white">Key Results</h3>
          <p className="mt-1 text-xs text-slate-500">
            See <a href="/dashboard/model-info" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">Model Info</a> for full performance metrics, training details, and result visualizations.
          </p>

          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">3-Month Lead Correlation</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">0.92</p>
              <p className="mt-1 text-sm text-slate-500">5-seed ensemble mean</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">3-Month Lead RMSE</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">0.51&deg;C</p>
              <p className="mt-1 text-sm text-slate-500">5-seed ensemble mean</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">6-Month Lead Correlation</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">0.82</p>
              <p className="mt-1 text-sm text-slate-500">5-seed ensemble mean</p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
              <p className="text-sm text-slate-400">Spatial Correlation (Temp)</p>
              <p className="mt-2 text-3xl font-bold tracking-tight text-white">0.30</p>
              <p className="mt-1 text-sm text-slate-500">Regional anomaly skill</p>
            </div>
          </div>

          <p className="mt-3">
            The model correctly captured phase transitions of major historical events &mdash; the
            2020&ndash;2022 triple-dip La Ni&ntilde;a and the 2023&ndash;2024 Super El Ni&ntilde;o
            &mdash; without dampening peak amplitude, a known weakness of tree-based baselines.
            Regional impact skill shows a spatial correlation of 0.30 for temperature anomalies
            versus 0.11 for precipitation anomalies, indicating temperature teleconnections are
            notably stronger and more learnable than precipitation.
          </p>
        </section>

        {/* Section 5 — Limitations & Future Scope */}
        <section>
          <h3 className="text-base font-semibold text-white">Limitations &amp; Future Scope</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              Monthly-averaged input data smooths out sub-monthly triggers such as westerly wind
              bursts.
            </li>
            <li>
              Precipitation forecast skill is more modest due to its chaotic, non-linear nature;
              probabilistic or diffusion-based approaches could help improve this.
            </li>
            <li>
              The model does not yet incorporate other climate drivers such as the Indian Ocean
              Dipole or the Arctic Oscillation.
            </li>
          </ul>
        </section>

        {/* Section 6 — Team & Supervision */}
        <section>
          <h3 className="text-base font-semibold text-white">Team &amp; Supervision</h3>
          <ul className="mt-2 space-y-1.5 list-disc list-inside">
            <li>
              <span className="text-slate-300">Team:</span> Biraj Adhikari, Raman Shrestha, Rojin
              Dhami, Sandeep Khadka &mdash; Computer Engineering, Thapathali Campus, IOE,
              Tribhuvan University.
            </li>
            <li>
              <span className="text-slate-300">Supervised by</span> Asst. Prof. Kobid Karkee,
              Department of Electronics and Computer Engineering.
            </li>
            <li>
              Submitted August 2026, as a minor project for the Bachelor&rsquo;s degree.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
