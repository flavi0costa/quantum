import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Trade S&P500", layout="wide")
st.title("🚀 Sinais de Swing Trade - Top 100 Ações Mais Líquidas do S&P 500")

# ====================== CACHE ======================
@st.cache_data(ttl=86400)
def get_sp500():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)[['Symbol', 'Security']]
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
    yf_ticker = ticker.replace('.', '-')
    df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
    
    if df.empty or len(df) < 200:
        return None
    
    # Corrige MultiIndex (caso raro do yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
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
    
    # === NOVOS INDICADORES ===
    # Bollinger Bands (20,2)
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # Stochastic Oscillator (14,3,3)
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
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
    prev_macd = prev['MACD']
    prev_signal = prev['Signal']
    
    # Novos
    stoch_k = latest.get('Stoch_K', np.nan)
    stoch_d = latest.get('Stoch_D', np.nan)
    prev_stoch_k = prev.get('Stoch_K', np.nan)
    prev_stoch_d = prev.get('Stoch_D', np.nan)
    
    score = 0
    
    # Tendência
    if pd.notna(price) and pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200:
            score += 3
        elif price < sma50 < sma200:
            score -= 3
    
    # RSI
    if pd.notna(rsi):
        if rsi < 35: score += 2
        elif rsi > 65: score -= 2
    
    # MACD
    if pd.notna(macd) and pd.notna(signal_line) and pd.notna(prev_macd) and pd.notna(prev_signal):
        if macd > signal_line and prev_macd <= prev_signal:
            score += 3
        elif macd < signal_line and prev_macd >= prev_signal:
            score -= 3
    
    # Histograma MACD
    if pd.notna(hist):
        score += 1 if hist > 0 else -1
    
    # === NOVO: Stochastic ===
    if pd.notna(stoch_k) and pd.notna(stoch_d) and pd.notna(prev_stoch_k) and pd.notna(prev_stoch_d):
        if stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d:
            score += 3 if stoch_k < 40 else 1
        elif stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d:
            score -= 3 if stoch_k > 60 else -1
    
    # === NOVO: Bollinger Bands ===
    if 'BB_Lower' in latest and 'BB_Upper' in latest:
        bb_lower = latest['BB_Lower']
        bb_upper = latest['BB_Upper']
        if pd.notna(bb_lower) and pd.notna(bb_upper) and (bb_upper - bb_lower) > 0:
            bb_position = (price - bb_lower) / (bb_upper - bb_lower)
            if bb_position < 0.20 and pd.notna(sma50) and price > sma50:
                score += 2
            elif bb_position > 0.80 and pd.notna(sma50) and price < sma50:
                score -= 2
    
    if score >= 7:   return "🟢 Compra Forte", score
    if score >= 4:   return "🟢 Compra", score
    if score <= -7:  return "🔴 Venda Forte", score
    if score <= -4:  return "🔴 Venda", score
    return "⚪ Neutro", score

# ====================== INTERFACE ======================
if st.sidebar.button("🔄 Atualizar Todos os Dados"):
    st.cache_data.clear()
    st.rerun()

top_df, top_tickers = get_top_liquid_stocks(100, 30)

if 'signals_df' not in st.session_state or st.sidebar.button("Recalcular Sinais"):
    with st.spinner("A calcular indicadores e sinais de Swing Trade (25-55 segundos)..."):
        signals = []
        data_cache = {}
        skipped = 0
        
        for ticker in top_df['Symbol']:
            try:
                df = calculate_indicators(ticker)
                if df is not None:
                    signal_text, score = generate_signal(df)
                    latest = df.iloc[-1]
                    
                    # Cálculo BB %B para exibição
                    bb_position = np.nan
                    if 'BB_Lower' in latest and 'BB_Upper' in latest:
                        bl = latest['BB_Lower']
                        bu = latest['BB_Upper']
                        if pd.notna(bl) and pd.notna(bu) and (bu - bl) != 0:
                            bb_position = (latest['Close'] - bl) / (bu - bl)
                    
                    signals.append({
                        'Símbolo': ticker,
                        'Empresa': top_df[top_df['Symbol'] == ticker]['Security'].iloc[0],
                        'Preço': round(latest['Close'], 2),
                        'Variação %': round((latest['Close'] / df.iloc[-2]['Close'] - 1) * 100, 2),
                        'Vol. Médio Diário': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                        'RSI': round(latest['RSI'], 1) if pd.notna(latest['RSI']) else "N/A",
                        'MACD Hist': round(latest['MACD_Hist'], 3) if pd.notna(latest['MACD_Hist']) else "N/A",
                        'Stoch_K': round(latest['Stoch_K'], 1) if pd.notna(latest.get('Stoch_K')) else "N/A",
                        'BB_%B': round(bb_position * 100, 1) if pd.notna(bb_position) else "N/A",
                        'Sinal': signal_text,
                        'Score': score
                    })
                    data_cache[ticker] = df
                else:
                    skipped += 1
            except:
                skipped += 1
                continue
        
        st.session_state.signals_df = pd.DataFrame(signals)
        st.session_state.data_cache = data_cache
        if skipped > 0:
            st.success(f"✅ {len(signals)} ações processadas | {skipped} ignoradas")

signals_df = st.session_state.signals_df

st.subheader("📊 Top 100 Ações + Sinais de Swing Trade")
st.dataframe(
    signals_df.sort_values('Score', ascending=False),
    column_config={
        "Sinal": st.column_config.TextColumn("Sinal", width="medium"),
        "Score": st.column_config.NumberColumn("Score", format="%d"),
        "Variação %": st.column_config.NumberColumn("Variação %", format="%.2f%%"),
        "Stoch_K": st.column_config.NumberColumn("Stoch %K", format="%.1f"),
        "BB_%B": st.column_config.NumberColumn("BB %B", format="%.1f"),
    },
    use_container_width=True,
    height=650
)

# ====================== DETALHE ======================
st.subheader("📈 Detalhe da Ação")
selected = st.selectbox("Escolhe uma ação:", options=signals_df['Símbolo'], index=0)

df = st.session_state.data_cache[selected]
latest = df.iloc[-1]
signal_text, _ = generate_signal(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço Atual", f"${latest['Close']:.2f}", f"{(latest['Close']/df.iloc[-2]['Close']-1)*100:+.2f}%")
col2.metric("RSI (14)", f"{latest['RSI']:.1f}" if pd.notna(latest.get('RSI')) else "N/A")
col3.metric("Sinal Swing Trade", signal_text)
col4.metric("Volume Médio 30d", f"{int(top_df[top_df['Symbol']==selected]['Avg_Daily_Volume'].iloc[0]):,}")

# Métricas extras dos novos indicadores
c1, c2 = st.columns(2)
stoch_k = latest.get('Stoch_K', np.nan)
stoch_d = latest.get('Stoch_D', np.nan)
bb_position = np.nan
if 'BB_Lower' in latest and 'BB_Upper' in latest:
    bl = latest['BB_Lower']
    bu = latest['BB_Upper']
    if pd.notna(bl) and pd.notna(bu) and (bu - bl) != 0:
        bb_position = (latest['Close'] - bl) / (bu - bl)

c1.metric("Stochastic %K / %D", f"{stoch_k:.1f} / {stoch_d:.1f}" if pd.notna(stoch_k) else "N/A")
c2.metric("Bollinger %B", f"{bb_position*100:.1f}%" if pd.notna(bb_position) else "N/A")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preço + Volume", "RSI", "MACD", "Bollinger Bands", "Stochastic"])

with tab1:
    fig_pv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pv.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color="orange")))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA 200", line=dict(color="blue")))
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
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histograma", marker_color=np.where(df['MACD_Hist'] >= 0, 'green', 'red')))
    fig_macd.update_layout(title="MACD (12,26,9)", height=350)
    st.plotly_chart(fig_macd, use_container_width=True)

with tab4:
    fig_bb = make_subplots()
    fig_bb.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color="red", dash="dash")))
    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="BB Mid", line=dict(color="gray")))
    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color="green", dash="dash")))
    fig_bb.update_layout(title=f"{selected} - Bollinger Bands (20,2)", xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig_bb, use_container_width=True)

with tab5:
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name="%K", line=dict(color="#1f77b4")))
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name="%D", line=dict(color="#ff7f0e")))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
    fig_stoch.update_layout(title="Stochastic Oscillator (14,3,3)", yaxis_range=[0, 100], height=350)
    st.plotly_chart(fig_stoch, use_container_width=True)

st.caption("App criada por COSTA • Indicadores: SMA + RSI + MACD + Bollinger + Stochastic • Apenas educativo")