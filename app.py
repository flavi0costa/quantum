import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Trade S&P500", layout="wide")
st.title("🚀 Sinais de Swing Trade - Top 100 Ações Mais Líquidas do S&P 500")

# ====================== CACHE ======================
@st.cache_data(ttl=86400)  # atualiza 1x por dia
def get_sp500():
    # CSV oficial mantido pela comunidade (nunca dá 403)
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)[['Symbol', 'Security']]   # <-- aqui estava o erro
    return df

@st.cache_data(ttl=3600)
def get_top_liquid_stocks(n=100, days=30):
    sp500 = get_sp500()
    tickers = sp500['Symbol'].tolist()
    
    with st.spinner("A obter volume médio das \~500 ações do S&P 500..."):
        vol_data = yf.download(tickers, period=f"{days}d", progress=False, threads=True)['Volume']
        avg_vol = vol_data.mean().sort_values(ascending=False)
        
        top_tickers = avg_vol.head(n).index.tolist()
        name_dict = dict(zip(sp500['Symbol'], sp500['Security']))
        
        top_df = pd.DataFrame({
            'Symbol': top_tickers,
            'Security': [name_dict.get(t, t) for t in top_tickers],
            'Avg_Daily_Volume': avg_vol.head(n).astype(int)
        })
    return top_df, top_tickers

# ====================== INDICADORES ======================
def calculate_indicators(ticker):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        return None
    
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def generate_signal(df):
    if len(df) < 200:
        return "Dados insuficientes", 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    price = latest['Close']
    sma50 = latest['SMA50']
    sma200 = latest['SMA200']
    rsi = latest['RSI']
    macd = latest['MACD']
    signal_line = latest['Signal']
    hist = latest['MACD_Hist']
    
    score = 0
    
    if price > sma50 > sma200:
        score += 3
    elif price < sma50 < sma200:
        score -= 3
    
    if rsi < 35:
        score += 2
    elif rsi > 65:
        score -= 2
    
    if macd > signal_line and prev['MACD'] <= prev['Signal']:
        score += 3
    elif macd < signal_line and prev['MACD'] >= prev['Signal']:
        score -= 3
    
    score += 1 if hist > 0 else -1
    
    if score >= 6:   return "🟢 Compra Forte", score
    if score >= 3:   return "🟢 Compra", score
    if score <= -6:  return "🔴 Venda Forte", score
    if score <= -3:  return "🔴 Venda", score
    return "⚪ Neutro", score

# ====================== INTERFACE ======================
if st.sidebar.button("🔄 Atualizar Todos os Dados"):
    st.cache_data.clear()
    st.rerun()

top_df, top_tickers = get_top_liquid_stocks(100, 30)

# Calcula sinais
if 'signals_df' not in st.session_state or st.sidebar.button("Recalcular Sinais"):
    with st.spinner("A calcular indicadores e sinais de Swing Trade (20-40 segundos)..."):
        signals = []
        data_cache = {}
        
        for ticker in top_df['Symbol']:
            df = calculate_indicators(ticker)
            if df is not None:
                signal_text, score = generate_signal(df)
                latest = df.iloc[-1]
                
                signals.append({
                    'Símbolo': ticker,
                    'Empresa': top_df[top_df['Symbol'] == ticker]['Security'].iloc[0],
                    'Preço': round(latest['Close'], 2),
                    'Variação %': round((latest['Close'] / df.iloc[-2]['Close'] - 1) * 100, 2),
                    'Vol. Médio Diário': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                    'RSI': round(latest['RSI'], 1),
                    'MACD Hist': round(latest['MACD_Hist'], 3),
                    'Sinal': signal_text,
                    'Score': score
                })
                data_cache[ticker] = df
        
        st.session_state.signals_df = pd.DataFrame(signals)
        st.session_state.data_cache = data_cache

signals_df = st.session_state.signals_df

st.subheader("📊 Top 100 Ações + Sinais de Swing Trade")
st.dataframe(
    signals_df.sort_values('Score', ascending=False),
    column_config={
        "Sinal": st.column_config.TextColumn("Sinal", width="medium"),
        "Score": st.column_config.NumberColumn("Score", format="%d"),
        "Variação %": st.column_config.NumberColumn("Variação %", format="%.2f%%"),
    },
    use_container_width=True,
    height=600
)

# Gráfico da ação selecionada
st.subheader("📈 Detalhe da Ação")
selected = st.selectbox("Escolhe uma ação:", options=signals_df['Símbolo'], index=0)

df = st.session_state.data_cache[selected]
latest = df.iloc[-1]
signal_text, _ = generate_signal(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço Atual", f"${latest['Close']:.2f}", f"{(latest['Close']/df.iloc[-2]['Close']-1)*100:+.2f}%")
col2.metric("RSI (14)", f"{latest['RSI']:.1f}")
col3.metric("Sinal Swing Trade", signal_text)
col4.metric("Volume Médio 30d", f"{int(top_df[top_df['Symbol']==selected]['Avg_Daily_Volume'].iloc[0]):,}")

tab1, tab2, tab3 = st.tabs(["Preço + Volume", "RSI", "MACD"])

with tab1:
    fig_pv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pv.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color="orange", width=2)))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA 200", line=dict(color="blue", width=2)))
    fig_pv.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,149,237,0.6)"), secondary_y=True)
    fig_pv.update_layout(title=f"{selected} - Gráfico Diário", xaxis_rangeslider_visible=False, height=650)
    st.plotly_chart(fig_pv, use_container_width=True)

with tab2:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI 14", line=dict(color="#9B59B6")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
    fig_rsi.update_layout(title="RSI (14 períodos)", yaxis_range=[0, 100], height=350)
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab3:
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal Line", line=dict(color="red")))
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histograma", 
                             marker_color=np.where(df['MACD_Hist'] >= 0, 'green', 'red')))
    fig_macd.update_layout(title="MACD (12,26,9)", height=350)
    st.plotly_chart(fig_macd, use_container_width=True)

st.caption("App criada por Grok • Dados via Yahoo Finance + GitHub CSV • Apenas educativo. Não é aconselhamento financeiro.")