# src/app.py
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PROCESSED_PATH = BASE_PATH / "data" / "processed"

CENTERS_PARQUET = DATA_PROCESSED_PATH / "centers.parquet"
METRICS_PATH = DATA_PROCESSED_PATH / "backtest_metrics_all_centers.parquet"
PLOTS_PATH = DATA_PROCESSED_PATH / "backtest_plot_all_centers.parquet"
FORECAST_PATH = DATA_PROCESSED_PATH / "forecast_monthly_all_centers.parquet"


LOGO_PATH = BASE_PATH / "data" / "features" / "controlauto_logo.png"

# -----------------------------
# Loaders (robustos + cache)
# -----------------------------
@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)     # evita dataset scanner
    return pf.read().to_pandas()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Forecast de Inspeções", layout="wide")

# Header com logo
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=260)
    else:
        st.warning(f"Logo não encontrado em: {LOGO_PATH}")
with col_title:
    st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-transform: uppercase;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        margin-bottom: 5px;
    }
    .custom-divider {
        height: 4px;
        background-color: #F37021;
        border-radius: 1px;
        margin-top: 1px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 28px;
        font-weight: 600;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        margin-bottom: 4px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Forecast de Inspeções</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

st.divider()

# Validar ficheiros necessários
missing_files = [p for p in [CENTERS_PARQUET, METRICS_PATH, PLOTS_PATH] if not p.exists()]
if missing_files:
    st.error("Ficheiros em falta:\n\n" + "\n".join([str(p) for p in missing_files]))
    st.stop()

# Load data
centros_df = normalize_cols(load_parquet(str(CENTERS_PARQUET)))
metrics_df = load_parquet(str(METRICS_PATH))
plots_df = load_parquet(str(PLOTS_PATH))
forecast_df = load_parquet(str(FORECAST_PATH))


# Normalizar colunas e nomes para consistência (CFGCENTROID -> cfgcentroid)
metrics_df = normalize_cols(metrics_df).rename(columns={"cfgcentroid": "cfgcentroid"})
plots_df = normalize_cols(plots_df).rename(columns={"cfgcentroid": "cfgcentroid"})
forecast_df = normalize_cols(forecast_df)


# Como os teus parquets vêm com CFGCENTROID, após lower vira "cfgcentroid" automaticamente.
# Garantir:
if "cfgcentroid" not in metrics_df.columns:
    # caso raro: se veio com outro nome, tenta mapear
    if "center_id" in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={"center_id": "cfgcentroid"})
    else:
        st.error(f"Não encontrei coluna de centro em metrics: {list(metrics_df.columns)}")
        st.stop()

if "cfgcentroid" not in plots_df.columns:
    if "center_id" in plots_df.columns:
        plots_df = plots_df.rename(columns={"center_id": "cfgcentroid"})
    else:
        st.error(f"Não encontrei coluna de centro em plots: {list(plots_df.columns)}")
        st.stop()

# centros parquet tem: center_id, center_name, label
if "center_id" not in centros_df.columns or "label" not in centros_df.columns:
    st.error(f"centers.parquet precisa de colunas ['center_id','label']. Tenho: {list(centros_df.columns)}")
    st.stop()


# ----------------------------------
# Filtrar apenas centros com dados (mensal)
# ----------------------------------
centros_com_dados = set(metrics_df["cfgcentroid"].dropna().astype(int).unique())

# (opcional mas recomendado) garantir que também existem no plots
centros_com_dados &= set(plots_df["cfgcentroid"].dropna().astype(int).unique())

centros_df["center_id"] = centros_df["center_id"].astype(str)
centros_df["center_id_num"] = pd.to_numeric(centros_df["center_id"], errors="coerce")

centros_df = centros_df[
    centros_df["center_id_num"].isin(centros_com_dados)
].copy()

# ordenar por id crescente
centros_df = centros_df.sort_values(["center_id_num", "center_id"], na_position="last")


# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    #centro_label = st.selectbox("Seleciona um Centro", centros_df["label"].tolist(), index=0)
    st.markdown('<div class="section-title">Seleciona um Centro</div>', unsafe_allow_html=True)
    centro_label = st.selectbox("", centros_df["label"].tolist(), index=0)
    center_id = centro_label.split(" - ")[0].strip()

with col2:
    # default = Mensal
    #periodicidade = st.selectbox("Escolhe a Periodicidade", ["Mensal", "Semanal", "Diário"], index=0)
    st.markdown('<div class="section-title">Escolhe a Periodicidade</div>', unsafe_allow_html=True)
    periodicidade = st.selectbox("", ["Mensal", "Semanal", "Diário"], index=0)


# Nota: por agora só tens mensal. Mantemos as opções para a demo e mostramos aviso se não houver dados.
# Quando tiveres os datasets semanal/diário, podes criar ficheiros separados por periodicidade,
# ou adicionar uma coluna "periodicidade" e filtrar aqui.

# -----------------------------
# Filter
# -----------------------------
# Neste momento, assume-se que os ficheiros atuais são Mensal.
if periodicidade != "Mensal":
    st.info("⚠️ Por agora só existem resultados **Mensais**. Quando os ficheiros Semanal/Diário estiverem prontos, este dropdown vai filtrar automaticamente.")
    # Continuamos a mostrar mensal para não ficar vazio
    periodicidade_effective = "Mensal"
else:
    periodicidade_effective = "Mensal"

metrics_view = metrics_df[metrics_df["cfgcentroid"].astype(str) == str(center_id)].copy()
plots_view = plots_df[plots_df["cfgcentroid"].astype(str) == str(center_id)].copy()

st.divider()

# -----------------------------
# Metrics cards (MAE, RMSE, WAPE)
# -----------------------------
st.markdown(f"### Métricas — {centro_label} ({periodicidade_effective})")

if metrics_view.empty:
    st.warning("Sem métricas para este centro.")
else:
    # Como os teus ficheiros parecem ter 1 linha por centro, pegamos a primeira.
    row = metrics_view.iloc[0]

    mae = float(row["mae"]) if "mae" in metrics_view.columns else None
    rmse = float(row["rmse"]) if "rmse" in metrics_view.columns else None
    wape = float(row["wape"]) if "wape" in metrics_view.columns else None

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE - Erro médio absoluto", f"{mae:.2f}" if mae is not None else "—")
    c2.metric("RMSE - Raiz erro quadratico médio", f"{rmse:.2f}" if rmse is not None else "—")
    c3.metric("WAPE - Percentagem erro absoluto", f"{wape:.2%}" if wape is not None and wape <= 1 else (f"{wape:.4f}" if wape is not None else "—"))

    # tabela completa (opcional)
    with st.expander("Ver linha completa de métricas"):
        st.dataframe(metrics_view, use_container_width=True)


if False:
    # -----------------------------
    # Plot: Real vs Previsto
    # -----------------------------

    st.markdown("### Real vs Previsto (Backtesting)")

    if plots_view.empty:
        st.warning("Sem dados de plot para este centro.")
    else:
        plots_view["date"] = pd.to_datetime(plots_view["date"], errors="coerce")
        plots_view = plots_view.dropna(subset=["date"]).sort_values("date")

        plot_df = plots_view[["date", "real", "previsto"]].copy()
        plot_df = plot_df.rename(columns={"date": "data"})

        plot_df["data"] = plot_df["data"].dt.date
        plot_df["real"] = pd.to_numeric(plot_df["real"], errors="coerce")
        plot_df["previsto"] = pd.to_numeric(plot_df["previsto"], errors="coerce")

        # gráfico
        chart_df = plot_df.set_index("data")[["real", "previsto"]]

        if len(chart_df) == 1:
            st.bar_chart(chart_df)
        else:
            st.line_chart(chart_df)

        # tabela sem índice estranho
        st.dataframe(
            plot_df.reset_index(drop=True),
            use_container_width=True
        )


# -----------------------------
# Forecast (Mensal)
# -----------------------------
st.markdown("### Forecast (Próximos Meses)")

# filtrar forecast para o centro selecionado
if "cfgcentroid" not in forecast_df.columns:
    st.error(f"forecast precisa da coluna 'CFGCENTROID' (cfgcentroid). Tenho: {list(forecast_df.columns)}")
    st.stop()

fc_view = forecast_df[forecast_df["cfgcentroid"].astype(str) == str(center_id)].copy()

if fc_view.empty:
    st.warning("Sem dados de forecast para este centro.")
else:
    # validar colunas
    if "month" not in fc_view.columns or "y_hat" not in fc_view.columns:
        st.error(f"forecast precisa de colunas ['MONTH','y_hat'] (month, y_hat). Tenho: {list(fc_view.columns)}")
        st.stop()

    fc_view["month"] = pd.to_datetime(fc_view["month"], errors="coerce")
    fc_view["y_hat"] = pd.to_numeric(fc_view["y_hat"], errors="coerce").round(0).astype("Int64")
    fc_view = fc_view.dropna(subset=["month"]).sort_values("month")

    fc_plot = fc_view[["month", "y_hat"]].rename(columns={"month": "data", "y_hat": "forecast"}).copy()
    fc_plot["data"] = fc_plot["data"].dt.date

    st.line_chart(fc_plot.set_index("data")[["forecast"]])

    st.dataframe(
        fc_plot.reset_index(drop=True),
        use_container_width=True
    )


if False:
    # -----------------------------
    # Debug (opcional)
    # -----------------------------
    with st.expander("Debug: colunas lidas"):
        st.write("centers columns:", list(centros_df.columns))
        st.write("metrics columns:", list(metrics_df.columns))
        st.write("plots columns:", list(plots_df.columns))
