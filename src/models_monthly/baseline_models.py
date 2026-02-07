import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# 1. Caminhos
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "monthly_series_31.parquet"

# -------------------------------------------------------------------
# 2. Parâmetros
# -------------------------------------------------------------------

HORIZONS = [1, 3, 6]
SEASONAL_LAG = 12
MIN_TRAIN_SIZE = 24  # mínimo de observações para começar

# -------------------------------------------------------------------
# 3. Métricas
# -------------------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -------------------------------------------------------------------
# 4. Funções de previsão baseline
# -------------------------------------------------------------------

def naive_forecast(train_series, h):
    return np.repeat(train_series.iloc[-1], h)

def seasonal_naive_forecast(train_series, h, seasonal_lag):
    forecasts = []
    for i in range(h):
        forecasts.append(train_series.iloc[-seasonal_lag + i])
    return np.array(forecasts)

# -------------------------------------------------------------------
# 5. Rolling evaluation
# -------------------------------------------------------------------

series = pd.read_parquet(DATA_FILE)["y"]

results = []

for h in HORIZONS:
    naive_errors = []
    seasonal_errors = []

    for t in range(MIN_TRAIN_SIZE, len(series) - h):
        train = series.iloc[:t]
        test = series.iloc[t:t + h]

        # Naive
        naive_pred = naive_forecast(train, h)
        naive_errors.append({
            "mae": mae(test.values, naive_pred),
            "rmse": rmse(test.values, naive_pred)
            })

        # Seasonal naive
        if len(train) >= SEASONAL_LAG:
            seasonal_pred = seasonal_naive_forecast(train, h, SEASONAL_LAG)
            seasonal_errors.append({
                "mae": mae(test.values, seasonal_pred),
                "rmse": rmse(test.values, seasonal_pred)
                })

    results.append({
        "horizon": h,
        "naive_mae": np.mean([e["mae"] for e in naive_errors]),
        "naive_rmse": np.mean([e["rmse"] for e in naive_errors]),
        "seasonal_naive_mae": np.mean([e["mae"] for e in seasonal_errors]),
        "seasonal_naive_rmse": np.mean([e["rmse"] for e in seasonal_errors]),
        })

# -------------------------------------------------------------------
# 6. Resultados finais
# -------------------------------------------------------------------

results_df = pd.DataFrame(results)
print("\nBaseline results (rolling-origin evaluation):\n")
print(results_df.round(2))
