import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Trade SCANNER 2026", layout="wide")
st.title("🚀 Swing Trade SCANNER - Liquidez + 12 Indicadores")

# ====================== CACHE ======================
@st.cache_data(ttl=86400)
def get_sp500():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)[['Symbol', 'Security']]
    return df

@st.cache_data(ttl=86400)
def get_nasdaq100():
    url = "https://raw.githubusercontent.com/Gary-Strauss/nasdaq100-scraper/main/data/nasdaq100_constituents.csv"
    df = pd.read_csv(url)
    df = df[['Symbol', 'Company']].rename(columns={'Company': 'Security'})
    return df

# ====================== FILTROS NO SIDEBAR ======================
st.sidebar.header("🔍 Filtros de Liquidez (Swing Trade)")
universe = st.sidebar.selectbox(
    "Universo de ações",
    ["S&P 500", "NASDAQ 100", "S&P 500 + NASDAQ 100 (Combinado)"]
)

min_vol = st.sidebar.slider("Volume Médio Diário Mínimo (milhões de ações)", 1, 50, 5) * 1_000_000
min_price = st.sidebar.slider("Preço Mínimo ($)", 5, 100, 10)
only_buy = st.sidebar.checkbox("Mostrar apenas Sinais de Compra / Compra Forte", value=True)
max_show = st.sidebar.slider("Número máximo de ações a mostrar", 50, 500, 200)

# ====================== INDICADORES (mantido igual ao anterior) ======================
def calculate_indicators(ticker):
    yf_ticker = ticker.replace('.', '-')
    df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # ... (todo o código de indicadores anterior - SMA, RSI, MACD, BB, Stoch, CCI, ADX, OBV, Ichimoku, SuperTrend, Williams %R, MFI) ...
    # (para não repetir 100 linhas aqui, copia da versão anterior – está tudo igual)

    # (insere aqui o bloco completo de calculate_indicators da resposta anterior)

    return df

def generate_signal(df):
    # (mesmo código da versão anterior – mantido)
    # ... 
    return signal_text, score   # igual ao anterior

# ====================== CARREGAR UNIVERSO ======================
if universe == "S&P 500":
    pool = get_sp500()
elif universe == "NASDAQ 100":
    pool = get_nasdaq100()
else:
    sp = get_sp500()
    nas = get_nasdaq100()
    pool = pd.concat([sp, nas]).drop_duplicates(subset='Symbol')

# ====================== CALCULAR VOLUMES E FILTROS ======================
@st.cache_data(ttl=3600)
def get_liquid_stocks(pool, min_vol, min_price):
    tickers = pool['Symbol'].tolist()
    with st.spinner(f"A obter dados de {len(tickers)} ações..."):
        vol_data = yf.download(tickers, period="30d", progress=False, threads=True)['Volume']
        avg_vol = vol_data.mean()

        df_vol = pd.DataFrame({
            'Symbol': avg_vol.index,
            'Avg_Daily_Volume': avg_vol.values,
            'Security': pool.set_index('Symbol').loc[avg_vol.index, 'Security'].values
        })

        # Filtro inicial de liquidez
        df_vol = df_vol[(df_vol['Avg_Daily_Volume'] >= min_vol)]

    return df_vol

top_df = get_liquid_stocks(pool, min_vol, min_price)

# ====================== CALCULAR SINAIS ======================
if 'signals_df' not in st.session_state or st.sidebar.button("🔄 Recalcular Sinais"):
    with st.spinner("Calculando 12 indicadores em todas as ações..."):
        signals = []
        data_cache = {}
        for ticker in top_df['Symbol'][:500]:  # limite de segurança
            try:
                df = calculate_indicators(ticker)
                if df is not None:
                    latest = df.iloc[-1]
                    signal_text, score = generate_signal(df)
                    if only_buy and "Compra" not in signal_text:
                        continue
                    if latest['Close'] < min_price:
                        continue

                    signals.append({
                        'Símbolo': ticker,
                        'Empresa': top_df[top_df['Symbol'] == ticker]['Security'].iloc[0],
                        'Preço': round(latest['Close'], 2),
                        'Variação %': round((latest['Close'] / df.iloc[-2]['Close'] - 1) * 100, 2),
                        'Vol. Médio': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                        'Sinal': signal_text,
                        'Score': score
                    })
                    data_cache[ticker] = df
            except:
                continue

        st.session_state.signals_df = pd.DataFrame(signals)
        st.session_state.data_cache = data_cache

signals_df = st.session_state.signals_df
signals_df = signals_df.head(max_show)  # aplica limite

st.subheader(f"📊 Scanner de Liquidez - {len(signals_df)} ações encontradas")
st.dataframe(
    signals_df.sort_values('Score', ascending=False),
    use_container_width=True,
    height=700
)

# Download CSV
if not signals_df.empty:
    csv = signals_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV completo",
        data=csv,
        file_name=f"swing_scanner_{universe.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# O resto da app (detalhe + tabs + explicações + backtest) continua igual à versão anterior
# (copia as secções de "Detalhe da Ação" e os tabs com os expanders da resposta anterior)

st.caption("🚀 SCANNER de Liquidez por Grok • Filtra só ações com volume real para swing • Apenas educativo")