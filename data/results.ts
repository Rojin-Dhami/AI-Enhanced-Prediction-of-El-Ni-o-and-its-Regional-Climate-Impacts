import { ModelId } from "./models";
import { OutputType } from "@/types/results";

export interface ResultItem {
    id: string;
    model: ModelId;
    output: OutputType;
    title: string;
    description: string;
    image?: string;
    variant?: string;
}

export const results: ResultItem[] = [
    // XGBoost method 1
    {
        id: "method-1-mifs-k-sweep",
        model: "xgboost-method-1",
        output: "feature-selection",
        title: "MIFS Feature Selection - Method 1",
        description: "Validation RMSE across different numbers of MIFS-selected features. The optimal configuration selected K = 18 features.",
        image: "/results/xgboost-method-1/method-1-mifs-k-sweep-validation-rmse.png",
    },
    {
        id: "method-1-baseline-forecast",
        model: "xgboost-method-1",
        output: "enso-forecast",
        variant: "Baseline",
        title: "XGBoost Method 1 - Baseline Forecast",
        description: "Actual  Niño 3.4 observation compared with baseline XGBoost predictions.",
        image: "/results/xgboost-method-1/method-1-baseline-enso-forecast.png",
    },
    {
        id: "method-1-baseline-forecast-zoomed",
        model: "xgboost-method-1",
        output: "enso-forecast",
        variant: "Baseline Zoomed",
        title: "XGBoost Method 1 - Baseline Forecast (Zoomed)",
        description: "Zoomed view of the out-of-sample Niño 3.4 forecast.",
        image: "/results/xgboost-method-1/method-1-baseline-enso-forecast-zoomed.png",
    },
    {
        id: "method-1-tuned-forecast",
        model: "xgboost-method-1",
        output: "enso-forecast",
        variant: "Tuned",
        title: "XGBoost Method 1 - Tuned Forecast",
        description: "Actual Niño 3.4 observations compared with predictions from the tuned XGBoost configuration.",
        image: "/results/xgboost-method-1/method-1-tuned-enso-forecast.png",
    },
    {
        id: "method-1-regularized-forecast",
        model: "xgboost-method-1",
        output: "enso-forecast",
        variant: "Regularized",
        title: "XGBoost Method 1 - Regularized Forecast",
        description: "Actual Niño 3.4 observations compared with predictions from the regularized XGBoost configuration.",
        image: "/results/xgboost-method-1/method-1-regularized-enso-forecast.png",
    },
    {
        id: "method-1-tuned-feature-importance",
        model: "xgboost-method-1",
        output: "feature-importance",
        variant: "Tuned",
        title: "XGBoost Method 1 - Tuned Feature Importance",
        description: "Relative importance od features selected by the tuned XGBoost model.",
        image: "/results/xgboost-method-1/method-1-tuned-feature-importance.png",
    },
    {
        id: "method-1-regularized-feature-importance",
        model: "xgboost-method-1",
        output: "feature-importance",
        variant: "Regularized",
        title: "XGBoost Method 1 - Regularized Feature Importance",
        description: "Relative importance of features in the regularized XGBoost configuration.",
        image: "/results/xgboost-method-1/method-1-regularized-feature-importance.png",
    },
    // XGBoost Method 2
    {
        id: "cnn-tcn-training-3-month",
        model: "cnn-tcn",
        output: "training-curve",
        variant: "3 Months",
        title: "CNN-TCN Training Curve - 3 Months Lead",
        description: "Training and validation performance during CNN-TCN optimization for the 3-month forecast lead.",
        image: "/results/cnn-tcn/cnn-tcn-training-3-month.png",
    },
    {
        id: "cnn-tcn-forecast-3-month",
        model: "cnn-tcn",
        output: "enso-forecast",
        variant: "3 Months",
        title: "CNN-TCN ENSO Forecast - 3 Months Lead",
        description: "Actual versus predicted Niño 3.4 values for the CNN-TCN model using a 3-month forecast lead.",
        image: "/results/cnn-tcn/cnn-tcn-enso-forecast-3-month.png",
    },
    {
        id: "cnn-tcn-training-6-month",
        model: "cnn-tcn",
        output: "training-curve",
        variant: "6 Months",
        title: "CNN-TCN Training Curve - 6 Month Lead",
        description: "Training and validation performance during CNN-TCN optimization for the 6-month forecast lead.",
        image: "/results/cnn-tcn/cnn-tcn-training-6-month.png",
    },
    {
        id: "cnn-tcn-forecast-6-month",
        model: "cnn-tcn",
        output: "enso-forecast",
        variant: "6 Months",
        title: "CNN-TCN ENSO Forecast - 6 Month Lead",
        description: "Actual vs Predicted Niño 3.4 values for the CNN-TCN model using a 6-month forecast lead.",
        image: "/results/cnn-tcn/cnn-tcn-enso-forecast-6-month.png",
    },
    {
        id: "cnn-tcn-temperature-actual",
        model: "cnn-tcn",
        output: "temperature-anomaly",
        variant: "Actual",
        title: "South Asia Temperature Anomaly - Actual",
        description: "Observed South Asia temperature anomaly for the evaluated forecast period.",
        image: "/results/cnn-tcn/south-asia/south-asia-temperature-actual-dec-2025.png",
    },
    {
        id: "cnn-tcn-temperature-predicted",
        model: "cnn-tcn",
        output: "temperature-anomaly",
        variant: "Predicted",
        title: "South Asia Temperature Anomaly - Predicted",
        description: "CNN-TCN predicted South Asia temperature anomaly for the evaluated forecast period.",
        image: "/results/cnn-tcn/south-asia/south-asia-temperature-predicted-dec-2025.png",
    },
    {
        id: "cnn-tcn-precipitation-actual",
        model: "cnn-tcn",
        output: "precipitation-anomaly",
        variant: "Actual",
        title: "South Asia Precipitation Anomaly - Actual",
        description: "Observed South Asia precipitation anomaly for the evaluated forecast period.",
        image: "/results/cnn-tcn/south-asia/south-asia-precipitation-actual-dec-2025.png",
    },
    {
        id: "cnn-tcn-precipitation-predicted",
        model: "cnn-tcn",
        output: "precipitation-anomaly",
        variant: "Predicted",
        title: "South Asia Precipitation Anomaly - Predicted",
        description: "CNN-TCN predicted South Asia precipitation anomaly for the evaluated forecast period.",
        image: "/results/cnn-tcn/south-asia/south-asia-precipitation-predicted-dec-2025.png",
    },
];