export type ModelId = | "cnn-tcn" | "xgboost-method-1" | "xgboost-method-2";

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
    {
        id: "xgboost-method-1",
        name: "XGBoost Method 1",
        shortName: "XGBoost M1",
        description: "MIFS-selected feature baseline for Niño 3.4 forecasting.",
        outputs: [
            "enso-forecast",
            "feature-importance",
            "performance",
        ],
    },
    {
        id: "xgboost-method-2",
        name: "XGBoost Method 2",
        shortName: "XGBoost M2",
        description: "Broad approximate feature-selection baseline and strongest XGBoost benchmark.",
        outputs: [
            "enso-forecast",
            "performance",
        ],
    },
];