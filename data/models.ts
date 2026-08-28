export type ModelId = "cnn-tcn";

export interface ModelInfo {
    id: ModelId;
    name: string;
    shortName: string;
    description: string;
    outputs: string[];
}

export const models: ModelInfo[] = [
    {
        id: "cnn-tcn",
        name: "CNN-TCN Ensemble",
        shortName: "CNN-TCN",
        description: "Multi-task spatio-temporal model for ENSO forecasting and South Asia climate anomaly prediction.",
        outputs: [
            "enso-forecast",
            "training-curve",
            "temperature-anomaly",
            "precipitation-anomaly",
            "performance",
        ],
    },
];
