import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# carregar dados
df = pd.read_parquet("data/processed/monthly_all_centers.parquet")

results = []

for center_id, g in df.groupby("CFGCENTROID"):
    g = g.sort_values("MONTH")
    y = g["y"].values

    # segurança: precisamos de histórico suficiente
    if len(y) < 24:   # 2 anos mínimo
        continue

    train, test = y[:-1], y[-1:]

    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        ).fit()

        y_hat = model.forecast(1)[0]

        mae = abs(test[0] - y_hat)
        rmse = np.sqrt((test[0] - y_hat) ** 2)

        results.append({
            "CFGCENTROID": center_id,
            "real": test[0],
            "previsto": round(y_hat),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2)
        })

    except Exception as e:
        # se algum centro falhar, não rebenta o loop
        results.append({
            "CFGCENTROID": center_id,
            "real": test[0],
            "previsto": np.nan,
            "mae": np.nan,
            "rmse": np.nan
        })

results_df = pd.DataFrame(results)

print("Backtest 1 mês – resumo")
print(results_df.describe())
