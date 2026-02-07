import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ler base mensal de todos os centros
df = pd.read_parquet("data/processed/monthly_all_centers.parquet")

centers = df["CFGCENTROID"].unique()

results = []

for center in centers:
    series = (
        df[df["CFGCENTROID"] == center]
        .set_index("MONTH")["y"]
        .asfreq("MS")
    )

    if len(series) < 24:
        continue  # séries curtas demais

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    ).fit()

    forecast = model.forecast(12)

    tmp = (
        forecast
        .to_frame("y_hat")
        .assign(CFGCENTROID=center)
        .reset_index()
        .rename(columns={"index": "MONTH"})
    )

    results.append(tmp)

# juntar tudo
forecast_all = pd.concat(results, ignore_index=True)

forecast_all.to_parquet(
    "data/processed/forecast_monthly_all_centers.parquet",
    index=False
)

print("Forecast criado para", forecast_all["CFGCENTROID"].nunique(), "centros")
