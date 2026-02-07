import pandas as pd

# ler produção
df = pd.read_csv(
    "data/raw/Producao.csv",
    usecols=["CFGCENTROID", "DATA_RECEPCAO"],
    parse_dates=["DATA_RECEPCAO"]
)

# criar coluna mês
df["MONTH"] = df["DATA_RECEPCAO"].dt.to_period("M").dt.to_timestamp()

# agregação mensal por centro
monthly = (
    df.groupby(["CFGCENTROID", "MONTH"])
      .size()
      .reset_index(name="y")
      .sort_values(["CFGCENTROID", "MONTH"])
)

# guardar em parquet
monthly.to_parquet("data/processed/monthly_all_centers.parquet", index=False)

print("Base mensal criada:", monthly.shape)
