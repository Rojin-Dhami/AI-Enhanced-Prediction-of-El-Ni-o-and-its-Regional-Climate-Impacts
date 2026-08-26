export interface PerformanceMetric {
    model: string;
    label: string;
    rmse: number;
    correlation: number;
}

export const performanceMetrics: PerformanceMetric[] = [
    {
        model: "xgboost-method-1",
        label: "XGBoost Method 1",
        rmse: 0.8554,
        correlation: 0.5608,
    },
    {
        model: "xgboost-method-2",
        label: "XGBoost Mehod 2",
        rmse: 0.5299,
        correlation: 0.9207,
    },
    {
        model: "cnn-tcn",
        label: "CNN-TCN",
        rmse: 0.3781,
        correlation: 0.9474,
    },
];

export const cnnTcnSixMonthPerformance = {
    model: "cnn-tcn",
    label: "CNN-TCN 6 Month Lead",
    rmse: 0.6806,
    correlation: 0.8256
};