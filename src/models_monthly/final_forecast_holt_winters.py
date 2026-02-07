import pandas as pd
import matplotlib.pyplot as plt
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

SEASONAL_PERIODS = 12
FORECAST_HORIZON = 12  # máximo (vamos extrair 1, 3, 6 e 12)

# -------------------------------------------------------------------
# 3. Carregar série
# -------------------------------------------------------------------

series = (
    pd.read_parquet(DATA_FILE)["y"]
    .asfreq("MS")
)

# -------------------------------------------------------------------
# 4. Ajustar modelo final (com todos os dados)
# -------------------------------------------------------------------

model = ExponentialSmoothing(
    series,
    trend="add",
    seasonal="add",
    seasonal_periods=SEASONAL_PERIODS,
    initialization_method="estimated"
)

fitted_model = model.fit(optimized=True)

forecast = fitted_model.forecast(FORECAST_HORIZON)

# -------------------------------------------------------------------
# 5. Extrair horizontes específicos
# -------------------------------------------------------------------

forecast_points = {
    "1 mês": forecast.iloc[0],
    "3 meses": forecast.iloc[2],
    "6 meses": forecast.iloc[5],
    "12 meses": forecast.iloc[11],
}

print("\nForecast pontual (Holt-Winters):")
for k, v in forecast_points.items():
    print(f"{k}: {v:.0f} inspeções")

# -------------------------------------------------------------------
# 6. Gráfico
# -------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(series, label="Histórico", marker="o")
plt.plot(forecast, label="Previsão (até 12 meses)", marker="o")

# Marcar horizontes específicos
for label, value in forecast_points.items():
    date = forecast[forecast == value].index[0]
    plt.scatter(date, value, s=80)
    plt.text(date, value, f" {label}", fontsize=9, verticalalignment="bottom")

plt.title("Previsão mensal de inspeções – Holt-Winters aditivo (Centro 31)")
plt.xlabel("Data")
plt.ylabel("Número de inspeções")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.savefig("outputs/forecast_holt_winters_center_31.png", dpi=300)
