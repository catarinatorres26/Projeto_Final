import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

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

# Modelos SARIMA candidatos
MODELS = {
    "SARIMA(1,0,0)(1,0,0)[12]": ((1,0,0), (1,0,0,12)),
    "SARIMA(0,0,1)(0,0,1)[12]": ((0,0,1), (0,0,1,12)),
    "SARIMA(1,0,1)(1,0,1)[12]": ((1,0,1), (1,0,1,12)),
}

# -------------------------------------------------------------------
# 3. Métricas
# -------------------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -------------------------------------------------------------------
# 4. Rolling evaluation
# -------------------------------------------------------------------

series = pd.read_parquet(DATA_FILE)["y"].asfreq("MS")

results = []

for model_name, (order, seasonal_order) in MODELS.items():
    for h in HORIZONS:
        mae_errors = []
        rmse_errors = []

        for t in range(MIN_TRAIN_SIZE, len(series) - h):
            train = series.iloc[:t]
            test = series.iloc[t:t + h]

            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted = model.fit(disp=False)
            forecast = fitted.forecast(h)

            mae_errors.append(mae(test.values, forecast.values))
            rmse_errors.append(rmse(test.values, forecast.values))

        results.append({
            "model": model_name,
            "horizon": h,
            "mae": np.mean(mae_errors),
            "rmse": np.mean(rmse_errors)
        })

# -------------------------------------------------------------------
# 5. Resultados finais
# -------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nSARIMA results – rolling origin:\n")
print(results_df.round(2))
