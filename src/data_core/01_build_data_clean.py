import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# 1. Caminhos
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "intermediate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCAO_FILE = RAW_DIR / "Producao.csv"

# -------------------------------------------------------------------
# 2. Leitura do ficheiro de produção
# -------------------------------------------------------------------

df = pd.read_csv(
    PRODUCAO_FILE,
    sep=",",
    encoding="utf-8",
    low_memory=False
)

# -------------------------------------------------------------------
# 3. Normalização de nomes de colunas
# -------------------------------------------------------------------

df = df.rename(columns={
    "CFGCENTROID": "CENTRO_ID",
    "DATA_RECEPCAO": "DATA_RECEPCAO"
})

# -------------------------------------------------------------------
# 4. Seleção das colunas relevantes
# -------------------------------------------------------------------

required_cols = ["CENTRO_ID", "DATA_RECEPCAO"]

missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Colunas em falta: {missing}")

df = df[required_cols]

# -------------------------------------------------------------------
# 5. Conversões de tipo
# -------------------------------------------------------------------

df["DATA_RECEPCAO"] = pd.to_datetime(
    df["DATA_RECEPCAO"],
    dayfirst=True,
    errors="coerce"
)

# -------------------------------------------------------------------
# 6. Limpeza básica
# -------------------------------------------------------------------

df = df.dropna(subset=["CENTRO_ID", "DATA_RECEPCAO"])

# -------------------------------------------------------------------
# 7. Ordenação temporal
# -------------------------------------------------------------------

df = df.sort_values(["CENTRO_ID", "DATA_RECEPCAO"])

# -------------------------------------------------------------------
# 8. Guardar data_clean
# -------------------------------------------------------------------

output_path = OUTPUT_DIR / "data_clean.parquet"
df.to_parquet(output_path, index=False)

print(f"data_clean criado com {len(df)} inspeções (linhas)")
print(f"Ficheiro guardado em: {output_path}")
