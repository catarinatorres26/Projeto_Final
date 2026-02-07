import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -------------------------------------------------------------------
# 1. Caminhos
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "monthly_series_31.parquet"

# -------------------------------------------------------------------
# 2. Parâmetros
# -------------------------------------------------------------------

HORIZONS = [1, 3, 6]
SEASONAL_PERIODS = 12
MIN_TRAIN_SIZE = 24

# -------------------------------------------------------------------
# 3. Métricas
# -------------------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -------------------------------------------------------------------
# 4. Avaliação rolling-origin
# -------------------------------------------------------------------

series = pd.read_parquet(DATA_FILE)["y"]

results = []

for h in HORIZONS:
    mae_errors = []
    rmse_errors = []

    for t in range(MIN_TRAIN_SIZE, len(series) - h):
        train = series.iloc[:t]
        test = series.iloc[t:t + h]

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIODS,
            initialization_method="estimated"
        )

        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(h)

        mae_errors.append(mae(test.values, forecast.values))
        rmse_errors.append(rmse(test.values, forecast.values))

    results.append({
        "horizon": h,
        "holt_winters_mae": np.mean(mae_errors),
        "holt_winters_rmse": np.mean(rmse_errors)
    })

# -------------------------------------------------------------------
# 5. Resultados finais
# -------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nHolt-Winters (additive) results – rolling origin:\n")
print(results_df.round(2))
