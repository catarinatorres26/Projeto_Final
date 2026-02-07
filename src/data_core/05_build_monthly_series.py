import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# 1. Caminhos
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "intermediate" / "data_clean.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 2. Parâmetros metodológicos
# -------------------------------------------------------------------

MIN_MONTHS_REQUIRED = 24        # mínimo para sazonalidade anual
REMOVE_INCOMPLETE_MONTHS = True

CENTER_COL = "CENTRO_ID"
DATE_COL = "DATA_RECEPCAO"

# -------------------------------------------------------------------
# 3. Leitura dos dados limpos
# -------------------------------------------------------------------

df = pd.read_parquet(INPUT_FILE)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# -------------------------------------------------------------------
# 4. Criação da variável mensal (início do mês)
# -------------------------------------------------------------------

df["MONTH"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()

# -------------------------------------------------------------------
# 5. Agregação: contagem de inspeções por centro × mês
# -------------------------------------------------------------------

monthly_counts = (
    df.groupby([CENTER_COL, "MONTH"])
      .size()
      .reset_index(name="y")
)

# -------------------------------------------------------------------
# 6. Identificação de centros com histórico suficiente
# -------------------------------------------------------------------

months_per_center = (
    monthly_counts
    .groupby(CENTER_COL)["MONTH"]
    .nunique()
    .sort_values(ascending=False)
)

eligible_centers = months_per_center[
    months_per_center >= MIN_MONTHS_REQUIRED
].index

if len(eligible_centers) == 0:
    raise ValueError("Nenhum centro tem histórico mensal suficiente.")

# -------------------------------------------------------------------
# 7. Seleção do centro representativo
#    Critério:
#    1) maior nº de meses
#    2) maior média mensal de inspeções
# -------------------------------------------------------------------

candidate_df = monthly_counts[
    monthly_counts[CENTER_COL].isin(eligible_centers)
]

center_stats = (
    candidate_df
    .groupby(CENTER_COL)
    .agg(
        n_months=("MONTH", "nunique"),
        mean_monthly_inspections=("y", "mean")
    )
    .sort_values(
        by=["n_months", "mean_monthly_inspections"],
        ascending=False
    )
)

selected_center = center_stats.index[0]
print(f"Centro selecionado: {selected_center}")

# -------------------------------------------------------------------
# 8. Série mensal do centro selecionado
# -------------------------------------------------------------------

series_df = (
    candidate_df[candidate_df[CENTER_COL] == selected_center]
    .set_index("MONTH")
    .sort_index()
)

# -------------------------------------------------------------------
# 9. Garantir regularidade temporal
# -------------------------------------------------------------------

full_index = pd.date_range(
    start=series_df.index.min(),
    end=series_df.index.max(),
    freq="MS"
)

monthly_series = (
    series_df["y"]
    .reindex(full_index, fill_value=0)
)

# -------------------------------------------------------------------
# 10. Remoção de meses incompletos
# -------------------------------------------------------------------

if REMOVE_INCOMPLETE_MONTHS and len(monthly_series) > 2:
    monthly_series = monthly_series.iloc[1:-1]

monthly_series = monthly_series.asfreq("MS")

# -------------------------------------------------------------------
# 11. Validação final
# -------------------------------------------------------------------

if len(monthly_series) < MIN_MONTHS_REQUIRED:
    raise ValueError("Série mensal final ficou curta demais.")

# -------------------------------------------------------------------
# 12. Guardar output
# -------------------------------------------------------------------

output_path = OUTPUT_DIR / f"monthly_series_{selected_center}.parquet"
monthly_series.to_frame(name="y").to_parquet(output_path)

print(f"Série mensal guardada em: {output_path}")
print(f"Número de observações: {len(monthly_series)}")
