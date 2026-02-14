import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# carregar série mensal
df = pd.read_parquet("data/processed/monthly_all_centers.parquet")

results = []

for center_id, g in df.groupby("CFGCENTROID"):

    g = g.sort_values("MONTH")
    y = g["y"].values

    if len(y) < 24:
        continue

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    ).fit()

    # previsões
    forecast = model.forecast(12)

    last_date = g["MONTH"].max()

    for h in [1, 3, 6, 12]:
        forecast_date = last_date + pd.DateOffset(months=h)

        results.append({
            "CFGCENTROID": center_id,
            "MONTH": forecast_date,
            "horizon": h,
            "forecast": forecast[h-1]
        })

forecast_df = pd.DataFrame(results)

forecast_df.to_parquet(
    "data/processed/monthly_forecasts.parquet",
    index=False
)

print("Forecast mensal exportado com sucesso.")
