# src/app.py
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = BASE_PATH / "data" / "raw"
DATA_PROCESSED_PATH = BASE_PATH / "data" / "processed"

CENTERS_FILE = DATA_RAW_PATH / "Coordenadas centros.xlsx"

METRICS_PATH = DATA_PROCESSED_PATH / "backtest_metrics_all_centers.parquet"
PLOTS_PATH   = DATA_PROCESSED_PATH / "backtest_plot_all_centers.parquet"

# -----------------------------
# Parquet loader (robusto)
# -----------------------------
@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)          # evita o dataset scanner
    table = pf.read()
    return table.to_pandas()

@st.cache_data
def load_centers(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={
        "# Centro": "center_id",
        "Designação Centro (Base IMT, 2013)": "center_name",
    })
    df["center_id"] = df["center_id"].astype(str)
    df["center_name"] = df["center_name"].astype(str)
    df["label"] = df["center_name"] + " (" + df["center_id"] + ")"
    return df[["center_id", "center_name", "label"]].drop_duplicates()

def pick_col(cols, options):
    cols_set = set(cols)
    for c in options:
        if c in cols_set:
            return c
    return None

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Controlauto - Backtesting", layout="wide")
st.title("Controlauto — Backtesting (Métricas + Gráficos)")

# validações de ficheiros
if not CENTERS_FILE.exists():
    st.error(f"Não encontrei o ficheiro de centros: {CENTERS_FILE}")
    st.stop()
if not METRICS_PATH.exists():
    st.error(f"Não encontrei: {METRICS_PATH}")
    st.stop()
if not PLOTS_PATH.exists():
    st.error(f"Não encontrei: {PLOTS_PATH}")
    st.stop()

centros_df = load_centers(str(CENTERS_FILE))
metrics = load_parquet(str(METRICS_PATH))
plots   = load_parquet(str(PLOTS_PATH))

# normalizar nomes de colunas
metrics.columns = metrics.columns.str.strip().str.lower()
plots.columns   = plots.columns.str.strip().str.lower()

# -----------------------------
# Inferir colunas (robusto)
# -----------------------------
# centro
c_center_m = pick_col(metrics.columns, ["cfgcentroid", "center_id", "id_centro", "centroid", "centro_id"])
c_center_p = pick_col(plots.columns,   ["cfgcentroid", "center_id", "id_centro", "centroid", "centro_id"])

# periodicidade / granularidade (se existir)
c_freq_m = pick_col(metrics.columns, ["freq", "frequency", "granularity", "periodicidade", "horizon_granularity"])
c_freq_p = pick_col(plots.columns,   ["freq", "frequency", "granularity", "periodicidade", "horizon_granularity"])

# modelo (se existir)
c_model_m = pick_col(metrics.columns, ["model", "modelo", "model_name"])
c_model_p = pick_col(plots.columns,   ["model", "modelo", "model_name"])

# timestamp para plot
c_ts_p = pick_col(plots.columns, ["timestamp", "ds", "date", "datetime", "month", "day", "time"])

# valores para plot
c_pred_p = pick_col(plots.columns, ["y_hat", "y_pred", "yhat", "prediction", "forecast", "previsto"])
c_true_p = pick_col(plots.columns, ["y", "y_true", "actual", "real", "observed"])

if c_center_m is None or c_center_p is None:
    st.error(
        "Não consegui identificar a coluna de centro nos parquets.\n\n"
        f"metrics colunas: {list(metrics.columns)}\n"
        f"plots colunas: {list(plots.columns)}"
    )
    st.stop()

if c_ts_p is None or c_pred_p is None:
    st.error(
        "Não consegui identificar colunas essenciais para o gráfico (timestamp + previsão).\n\n"
        f"plots colunas: {list(plots.columns)}"
    )
    st.stop()

# normalizar tipo do centro
metrics[c_center_m] = metrics[c_center_m].astype(str)
plots[c_center_p]   = plots[c_center_p].astype(str)

# converter timestamp
plots[c_ts_p] = pd.to_datetime(plots[c_ts_p], errors="coerce")

# -----------------------------
# Dropdowns
# -----------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    centro_label = st.selectbox("Centro", sorted(centros_df["label"].unique()))
center_id = centro_label.split("(")[-1].replace(")", "").strip()

# opções de periodicidade (se existir)
freq_options = ["Horária", "Diária", "Mensal"]
if c_freq_m and c_freq_p:
    # tentar mapear valores reais existentes no dataset
    # mostramos valores tal como existem no parquet (mais seguro)
    f1 = sorted(set(metrics[c_freq_m].dropna().astype(str).unique()))
    f2 = sorted(set(plots[c_freq_p].dropna().astype(str).unique()))
    freq_options = sorted(set(f1).intersection(set(f2))) or sorted(set(f1 + f2))

with col2:
    freq_choice = st.selectbox("Periodicidade", freq_options)

# opções de modelo (se existir)
model_options = ["(todos)"]
if c_model_m and c_model_p:
    m1 = sorted(set(metrics[c_model_m].dropna().astype(str).unique()))
    m2 = sorted(set(plots[c_model_p].dropna().astype(str).unique()))
    model_options = ["(todos)"] + (sorted(set(m1).intersection(set(m2))) or sorted(set(m1 + m2)))

with col3:
    model_choice = st.selectbox("Modelo", model_options)

# -----------------------------
# Filtrar dados
# -----------------------------
metrics_view = metrics[metrics[c_center_m] == center_id].copy()
plots_view   = plots[plots[c_center_p] == center_id].copy()

if c_freq_m and freq_choice is not None:
    metrics_view = metrics_view[metrics_view[c_freq_m].astype(str) == str(freq_choice)]
if c_freq_p and freq_choice is not None:
    plots_view = plots_view[plots_view[c_freq_p].astype(str) == str(freq_choice)]

if c_model_m and model_choice != "(todos)":
    metrics_view = metrics_view[metrics_view[c_model_m].astype(str) == str(model_choice)]
if c_model_p and model_choice != "(todos)":
    plots_view = plots_view[plots_view[c_model_p].astype(str) == str(model_choice)]

plots_view = plots_view.dropna(subset=[c_ts_p]).sort_values(c_ts_p)

# -----------------------------
# Mostrar resultados
# -----------------------------
st.subheader(f"{centro_label} — {freq_choice}" + (f" — {model_choice}" if model_choice != "(todos)" else ""))

# Métricas
st.markdown("### Métricas de backtesting")
if metrics_view.empty:
    st.warning("Sem métricas para esta seleção.")
else:
    # tenta mostrar só colunas “relevantes” primeiro
    preferred_metrics = [c for c in ["mae", "rmse", "mape", "smape", "r2"] if c in metrics_view.columns]
    show_cols = []
    for c in [c_freq_m, c_model_m, *preferred_metrics]:
        if c and c in metrics_view.columns and c not in show_cols:
            show_cols.append(c)
    # fallback: mostra tudo
    st.dataframe(metrics_view[show_cols] if show_cols else metrics_view, use_container_width=True)

# Gráfico
st.markdown("### Série real vs previsão (backtest)")
if plots_view.empty:
    st.warning("Sem dados de plot para esta seleção.")
else:
    # preparar df para plot
    plot_df = pd.DataFrame({"timestamp": plots_view[c_ts_p]})
    plot_df["previsao"] = pd.to_numeric(plots_view[c_pred_p], errors="coerce")
    if c_true_p and c_true_p in plots_view.columns:
        plot_df["real"] = pd.to_numeric(plots_view[c_true_p], errors="coerce")

    plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # linha com streamlit (rápido)
    st.line_chart(plot_df.set_index("timestamp"))

    # tabela final
    st.dataframe(plot_df.tail(200), use_container_width=True)

# Debug opcional
with st.expander("Debug: colunas detetadas"):
    st.write("metrics.columns:", list(metrics.columns))
    st.write("plots.columns:", list(plots.columns))
    st.write("Coluna centro metrics:", c_center_m)
    st.write("Coluna centro plots:", c_center_p)
    st.write("Coluna freq metrics:", c_freq_m)
    st.write("Coluna freq plots:", c_freq_p)
    st.write("Coluna modelo metrics:", c_model_m)
    st.write("Coluna modelo plots:", c_model_p)
    st.write("Coluna timestamp plots:", c_ts_p)
    st.write("Coluna previsão plots:", c_pred_p)
    st.write("Coluna real plots:", c_true_p)
