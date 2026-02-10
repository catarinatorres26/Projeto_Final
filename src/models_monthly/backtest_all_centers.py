import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --------------------------------------------------
# 1. Carregar dados
# --------------------------------------------------
df = pd.read_parquet("data/processed/monthly_all_centers.parquet")

results = []
plots = []

# --------------------------------------------------
# 2. Backtest one-step-ahead por centro
# --------------------------------------------------
for center_id, g in df.groupby("CFGCENTROID"):
    g = g.sort_values("MONTH")

    y = g["y"].values
    dates = g["MONTH"].values

    # histórico mínimo (2 anos)
    if len(y) < 24:
        continue

    y_train = y[:-1]
    y_true = y[-1]
    test_date = dates[-1]

    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        ).fit()

        y_pred = model.forecast(1)[0]

        # métricas
        mae = abs(y_true - y_pred)
        rmse = np.sqrt((y_true - y_pred) ** 2)
        wape = abs(y_true - y_pred) / abs(y_true)

        results.append({
            "CFGCENTROID": center_id,
            "real": y_true,
            "previsto": round(y_pred),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "wape": round(wape, 4)
        })

        # dados para gráfico real vs previsto
        plots.append({
            "CFGCENTROID": center_id,
            "date": test_date,
            "real": y_true,
            "previsto": round(y_pred)
        })

    except Exception:
        # se algum centro falhar, não rebenta o loop
        results.append({
            "CFGCENTROID": center_id,
            "real": y_true,
            "previsto": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "wape": np.nan
        })

# --------------------------------------------------
# 3. Resultados finais
# --------------------------------------------------
results_df = pd.DataFrame(results)
plots_df = pd.DataFrame(plots)

# guardar para relatório / gráficos
results_df.to_parquet(
    "data/processed/backtest_metrics_all_centers.parquet",
    index=False
)

plots_df.to_parquet(
    "data/processed/backtest_plot_all_centers.parquet",
    index=False
)

# --------------------------------------------------
# 4. Resumo
# --------------------------------------------------
print("Backtest mensal (1 passo) – resumo")
print(results_df.describe())
