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
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        df = pd.read_html(url)[0]  # tabela dos constituintes
        df = df[['Ticker', 'Company']].rename(columns={'Ticker': 'Symbol', 'Company': 'Security'})
        return df
    except:
        st.warning("⚠️ Não consegui carregar NASDAQ-100 (Wikipedia). Usando só S&P 500.")
        return pd.DataFrame(columns=['Symbol', 'Security'])

# ====================== FILTROS NO SIDEBAR ======================
st.sidebar.header("🔍 Filtros de Liquidez (Swing Trade)")
universe = st.sidebar.selectbox(
    "Universo de ações",
    ["S&P 500", "NASDAQ 100", "S&P 500 + NASDAQ 100 (Combinado)"]
)

min_vol = st.sidebar.slider("Volume Médio Diário Mínimo (milhões)", 1, 50, 5) * 1_000_000
min_price = st.sidebar.slider("Preço Mínimo ($)", 5, 100, 10)
only_buy = st.sidebar.checkbox("Mostrar apenas Sinais de Compra / Compra Forte", value=True)
max_show = st.sidebar.slider("Número máximo de ações a mostrar", 50, 500, 200)

# ====================== INDICADORES (12 + SuperTrend etc.) ======================
def calculate_indicators(ticker):
    yf_ticker = ticker.replace('.', '-')
    df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # (todo o código de indicadores anterior - SMA, RSI, MACD, BB, Stoch, CCI, ADX, OBV, Ichimoku, SuperTrend, Williams %R, MFI)
    # ... (mantido exatamente igual à versão anterior que funcionava)

    return df

def generate_signal(df):
    # (mantido exatamente igual à versão anterior)
    if len(df) < 200: return "Dados insuficientes", 0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    # ... (todo o cálculo de score com os 12 indicadores)
    if score >= 14: return "🟢 Compra Forte", score
    if score >= 9:  return "🟢 Compra", score
    if score <= -14: return "🔴 Venda Forte", score
    if score <= -9:  return "🔴 Venda", score
    return "⚪ Neutro", score

# ====================== CARREGAR UNIVERSO ======================
if universe == "S&P 500":
    pool = get_sp500()
elif universe == "NASDAQ 100":
    pool = get_nasdaq100()
else:
    sp = get_sp500()
    nas = get_nasdaq100()
    pool = pd.concat([sp, nas]).drop_duplicates(subset='Symbol').reset_index(drop=True)

# ====================== CALCULAR VOLUMES ======================
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
        df_vol = df_vol[df_vol['Avg_Daily_Volume'] >= min_vol]
    return df_vol

top_df = get_liquid_stocks(pool, min_vol, min_price)

# ====================== CALCULAR SINAIS ======================
if 'signals_df' not in st.session_state or st.sidebar.button("🔄 Recalcular Sinais"):
    with st.spinner("Calculando 12 indicadores..."):
        signals = []
        data_cache = {}
        for ticker in top_df['Symbol'][:600]:  # limite segurança
            try:
                df = calculate_indicators(ticker)
                if df is not None:
                    latest = df.iloc[-1]
                    signal_text, score = generate_signal(df)
                    if only_buy and "Compra" not in signal_text: continue
                    if latest['Close'] < min_price: continue

                    signals.append({
                        'Símbolo': ticker,
                        'Empresa': top_df[top_df['Symbol']==ticker]['Security'].iloc[0],
                        'Preço': round(latest['Close'],2),
                        'Variação %': round((latest['Close']/df.iloc[-2]['Close']-1)*100,2),
                        'Vol. Médio': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                        'Sinal': signal_text,
                        'Score': score
                    })
                    data_cache[ticker] = df
            except:
                continue
        st.session_state.signals_df = pd.DataFrame(signals)
        st.session_state.data_cache = data_cache

signals_df = st.session_state.signals_df.head(max_show)

st.subheader(f"📊 Scanner de Liquidez - {len(signals_df)} ações encontradas")
st.dataframe(signals_df.sort_values('Score', ascending=False), use_container_width=True, height=700)

if not signals_df.empty:
    csv = signals_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"swing_scanner_{universe}.csv", "text/csv")

# ====================== DETALHE + TABS COM EXPLICAÇÕES (igual ao anterior) ======================
# (copia aqui a parte de "Detalhe da Ação" e todos os tabs com os expanders "Como analisar" da versão anterior - está tudo igual)

st.caption("🚀 SCANNER de Liquidez por Grok • Filtra só ações com volume real • Apenas educativo")