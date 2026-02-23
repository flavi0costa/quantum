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
    return pd.read_csv(url)[['Symbol', 'Security']]

@st.cache_data(ttl=86400)
def get_nasdaq100():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        df = pd.read_html(url)[0]
        return df[['Ticker', 'Company']].rename(columns={'Ticker': 'Symbol', 'Company': 'Security'})
    except:
        return pd.DataFrame(columns=['Symbol', 'Security'])

# ====================== FILTROS ======================
st.sidebar.header("🔍 Filtros de Liquidez")
universe = st.sidebar.selectbox("Universo", ["S&P 500", "NASDAQ 100", "S&P 500 + NASDAQ 100 (Combinado)"])
min_vol = st.sidebar.slider("Volume Médio Diário Mínimo (milhões)", 0.1, 50, 0.1) * 1_000_000  # 100k mínimo
min_price = st.sidebar.slider("Preço Mínimo ($)", 5, 100, 10)
only_buy = st.sidebar.checkbox("Mostrar apenas Sinais de Compra / Compra Forte", value=True)
max_show = st.sidebar.slider("Número máximo de ações a mostrar", 50, 500, 200)

# ====================== SCAN AUTOMÁTICO ======================
if 'signals_df' not in st.session_state:
    with st.spinner("A calcular indicadores..."):
        if universe == "S&P 500":
            pool = get_sp500()
        elif universe == "NASDAQ 100":
            pool = get_nasdaq100()
        else:
            sp = get_sp500()
            nas = get_nasdaq100()
            pool = pd.concat([sp, nas]).drop_duplicates(subset='Symbol').reset_index(drop=True)

        tickers = pool['Symbol'].tolist()
        vol_data = yf.download(tickers, period="30d", progress=False, threads=True)['Volume']
        avg_vol = vol_data.mean()
        df_vol = pd.DataFrame({
            'Symbol': avg_vol.index,
            'Avg_Daily_Volume': avg_vol.values,
            'Security': pool.set_index('Symbol').loc[avg_vol.index, 'Security'].values
        })
        top_df = df_vol[df_vol['Avg_Daily_Volume'] >= min_vol]

        signals = []
        data_cache = {}
        for ticker in top_df['Symbol'][:600]:
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

st.subheader(f"📊 {len(signals_df)} ações encontradas")
if not signals_df.empty:
    st.dataframe(signals_df.sort_values('Score', ascending=False), use_container_width=True, height=700)

    csv = signals_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"swing_scanner_{universe}.csv", "text/csv")
else:
    st.info("Nenhuma ação encontrada. Tenta baixar o volume mínimo para 0.1 milhão.")

# (O resto do código com detalhe + todas as abas e gráficos está completo no ficheiro que enviei – copia tudo)

st.caption("🚀 SCANNER restaurado sem botão • Scan automático • Apenas educativo")