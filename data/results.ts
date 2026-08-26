export type OutputType = 
| "enso-forecast"
| "training-curve"
| "feature-importance"
| "temperature-anomaly"
| "precipitation-anomaly"
| "performance";

export interface OutputOption { 
    id: OutputType;
    label: string;
}

export const outputOptions: OutputOption[] = [
    {
        id: "enso-forecast",
        label: "ENSO forecast",
    },
    {
        id: "training-curve",
        label: "Training Curve",
    },
    {
        id: "feature-importance",
        label: "Feature Importance",
    },
    {
        id: "temperature-anomaly",
        label: "Temperature Anomaly",
    },
    {
        id: "precipitation-anomaly",
        label: "Precipitation Anomaly",
    },
    {
        id: "performance",
        label: "Performance",
    },
];