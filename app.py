import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------
st.set_page_config(
    page_title="Dashboard Pro Streamlit",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# TÍTULO
# ---------------------------
st.title("📊 Dashboard Inteligente em Streamlit")
st.write("App completo com análise automática de dados.")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("⚙️ Configurações")

file = st.sidebar.file_uploader("Carregar ficheiro CSV", type=["csv"])

use_demo = st.sidebar.checkbox("Usar dados de exemplo", value=True)

# ---------------------------
# DADOS
# ---------------------------
if file:
    df = pd.read_csv(file)

elif use_demo:
    np.random.seed(42)
    df = pd.DataFrame({
        "Vendas": np.random.randint(100, 1000, 100),
        "Lucro": np.random.randint(50, 500, 100),
        "Clientes": np.random.randint(10, 200, 100),
        "Região": np.random.choice(["Norte", "Centro", "Sul"], 100)
    })

else:
    st.warning("Carrega um ficheiro ou ativa dados demo.")
    st.stop()

# ---------------------------
# MOSTRAR DADOS
# ---------------------------
st.subheader("📋 Dados")

st.dataframe(df, use_container_width=True)

# ---------------------------
# ESTATÍSTICAS
# ---------------------------
st.subheader("📈 Estatísticas")

col1, col2, col3 = st.columns(3)

col1.metric("Linhas", df.shape[0])
col2.metric("Colunas", df.shape[1])
col3.metric("Valores Nulos", df.isna().sum().sum())

st.write(df.describe())

# ---------------------------
# FILTROS
# ---------------------------
st.subheader("🎛️ Filtros")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

filtered_df = df.copy()

# Filtro categórico
for col in cat_cols:
    values = st.multiselect(f"Filtrar {col}", df[col].unique(), default=df[col].unique())
    filtered_df = filtered_df[filtered_df[col].isin(values)]

# ---------------------------
# GRÁFICOS
# ---------------------------
st.subheader("📊 Visualização")

if len(numeric_cols) >= 2:

    x = st.selectbox("Eixo X", numeric_cols)
    y = st.selectbox("Eixo Y", numeric_cols, index=1)

    chart_type = st.radio(
        "Tipo de gráfico",
        ["Linha", "Dispersão", "Histograma"]
    )

    fig, ax = plt.subplots()

    if chart_type == "Linha":
        ax.plot(filtered_df[x], filtered_df[y])

    elif chart_type == "Dispersão":
        sns.scatterplot(data=filtered_df, x=x, y=y, ax=ax)

    else:
        ax.hist(filtered_df[x], bins=20)

    st.pyplot(fig)

else:
    st.info("Precisas de pelo menos 2 colunas numéricas.")

# ---------------------------
# DOWNLOAD
# ---------------------------
st.subheader("⬇️ Exportar Dados")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Descarregar CSV",
    csv,
    "dados_filtrados.csv",
    "text/csv"
)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Dashboard criado com Streamlit 🚀")