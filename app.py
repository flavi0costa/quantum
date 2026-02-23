import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Trade ULTIMATE 2026", layout="wide")
st.title("🚀 Swing Trade ULTIMATE - Top 100 S&P 500 (12 Indicadores + Backtest)")

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
    with st.spinner("Obtendo volumes..."):
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

# ====================== INDICADORES (igual ao anterior) ======================
def calculate_indicators(ticker):
    yf_ticker = ticker.replace('.', '-')
    df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad)

    df['TR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    alpha = 1/14
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']), df['High'] - df['High'].shift(), 0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()), df['Low'].shift() - df['Low'], 0)
    df['+DM'] = df['+DM'].clip(lower=0)
    df['-DM'] = df['-DM'].clip(lower=0)
    plus_dm_smooth = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    tr_smooth = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(52)
    df['Chikou'] = df['Close'].shift(-26)

    # SuperTrend
    multiplier = 3.0
    df['HL2'] = (df['High'] + df['Low']) / 2
    df['SuperUpper'] = df['HL2'] + multiplier * df['ATR']
    df['SuperLower'] = df['HL2'] - multiplier * df['ATR']
    df['SuperTrend'] = np.nan
    for i in range(1, len(df)):
        if pd.isna(df['SuperTrend'].iloc[i-1]):
            df.loc[df.index[i], 'SuperTrend'] = df['SuperLower'].iloc[i]
        else:
            if df['Close'].iloc[i-1] > df['SuperTrend'].iloc[i-1]:
                df.loc[df.index[i], 'SuperTrend'] = max(df['SuperLower'].iloc[i], df['SuperTrend'].iloc[i-1])
            else:
                df.loc[df.index[i], 'SuperTrend'] = min(df['SuperUpper'].iloc[i], df['SuperTrend'].iloc[i-1])

    # Williams %R
    high14 = df['High'].rolling(14).max()
    low14 = df['Low'].rolling(14).min()
    df['Williams_%R'] = -100 * (high14 - df['Close']) / (high14 - low14)

    # MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    mfr = pos_mf / neg_mf
    df['MFI'] = 100 - (100 / (1 + mfr))

    return df

def generate_signal(df):
    if len(df) < 200: return "Dados insuficientes", 0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    if pd.notna(latest['Close']) and pd.notna(latest.get('SMA50')) and pd.notna(latest.get('SMA200')):
        if latest['Close'] > latest['SMA50'] > latest['SMA200']: score += 3
        elif latest['Close'] < latest['SMA50'] < latest['SMA200']: score -= 3

    if pd.notna(latest.get('RSI')):
        if latest['RSI'] < 35: score += 2
        elif latest['RSI'] > 65: score -= 2

    if pd.notna(latest.get('MACD')) and pd.notna(latest.get('Signal')):
        if latest['MACD'] > latest['Signal'] and prev['MACD'] <= prev['Signal']: score += 3
        elif latest['MACD'] < latest['Signal'] and prev['MACD'] >= prev['Signal']: score -= 3

    if pd.notna(latest.get('MACD_Hist')):
        score += 1 if latest['MACD_Hist'] > 0 else -1

    if pd.notna(latest.get('Stoch_K')) and pd.notna(latest.get('Stoch_D')):
        if latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']:
            score += 3 if latest['Stoch_K'] < 40 else 1
        elif latest['Stoch_K'] < latest['Stoch_D'] and prev['Stoch_K'] >= prev['Stoch_D']:
            score -= 3 if latest['Stoch_K'] > 60 else -1

    if 'BB_Lower' in latest and 'BB_Upper' in latest:
        if pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']):
            bb_pos = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            if bb_pos < 0.20 and latest['Close'] > latest.get('SMA50', np.nan): score += 2
            elif bb_pos > 0.80 and latest['Close'] < latest.get('SMA50', np.nan): score -= 2

    if pd.notna(latest.get('CCI')):
        if latest['CCI'] < -100: score += 2
        elif latest['CCI'] > 100: score -= 2

    if pd.notna(latest.get('ADX')) and pd.notna(latest.get('+DI')) and pd.notna(latest.get('-DI')):
        if latest['ADX'] > 25:
            score += 2 if latest['+DI'] > latest['-DI'] else -2

    if pd.notna(latest.get('OBV')) and pd.notna(prev.get('OBV')):
        score += 1 if latest['OBV'] > prev['OBV'] else -1

    if pd.notna(latest.get('SuperTrend')):
        if latest['Close'] > latest['SuperTrend']: score += 4
        else: score -= 4

    if pd.notna(latest.get('Williams_%R')):
        if latest['Williams_%R'] < -80: score += 3
        elif latest['Williams_%R'] > -20: score -= 3

    if pd.notna(latest.get('MFI')):
        if latest['MFI'] < 20: score += 3
        elif latest['MFI'] > 80: score -= 3

    if score >= 14: return "🟢 Compra Forte", score
    if score >= 9:  return "🟢 Compra", score
    if score <= -14: return "🔴 Venda Forte", score
    if score <= -9:  return "🔴 Venda", score
    return "⚪ Neutro", score

# ====================== INTERFACE ======================
if st.sidebar.button("🔄 Atualizar Tudo"):
    st.cache_data.clear()
    st.rerun()

top_df, _ = get_top_liquid_stocks(100, 30)

if 'signals_df' not in st.session_state or st.sidebar.button("Recalcular Sinais"):
    with st.spinner("Calculando 12 indicadores..."):
        signals = []
        data_cache = {}
        for ticker in top_df['Symbol']:
            try:
                df = calculate_indicators(ticker)
                if df is not None:
                    signal_text, score = generate_signal(df)
                    latest = df.iloc[-1]
                    bb_pos = np.nan
                    if 'BB_Lower' in latest and 'BB_Upper' in latest and pd.notna(latest['BB_Lower']):
                        bb_pos = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    signals.append({
                        'Símbolo': ticker,
                        'Empresa': top_df[top_df['Symbol']==ticker]['Security'].iloc[0],
                        'Preço': round(latest['Close'],2),
                        'Variação %': round((latest['Close']/df.iloc[-2]['Close']-1)*100,2),
                        'Vol. Médio': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                        'RSI': round(latest.get('RSI'),1) if pd.notna(latest.get('RSI')) else "N/A",
                        'SuperTrend': "Acima" if pd.notna(latest.get('SuperTrend')) and latest['Close'] > latest['SuperTrend'] else "Abaixo",
                        'Williams_%R': round(latest.get('Williams_%R'),1) if pd.notna(latest.get('Williams_%R')) else "N/A",
                        'MFI': round(latest.get('MFI'),1) if pd.notna(latest.get('MFI')) else "N/A",
                        'ATR': round(latest.get('ATR'),2) if pd.notna(latest.get('ATR')) else "N/A",
                        'Sinal': signal_text,
                        'Score': score
                    })
                    data_cache[ticker] = df
            except:
                continue
        st.session_state.signals_df = pd.DataFrame(signals)
        st.session_state.data_cache = data_cache

signals_df = st.session_state.signals_df

st.subheader("📊 Top 100 + 12 Indicadores")
st.dataframe(signals_df.sort_values('Score', ascending=False), use_container_width=True, height=700)

st.subheader("📈 Detalhe da Ação")
selected = st.selectbox("Escolhe uma ação:", options=signals_df['Símbolo'], index=0)

df = st.session_state.data_cache[selected]
latest = df.iloc[-1]
signal_text, _ = generate_signal(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Preço", f"${latest['Close']:.2f}", f"{(latest['Close']/df.iloc[-2]['Close']-1)*100:+.2f}%")
col2.metric("Sinal", signal_text)
col3.metric("ATR", f"{latest.get('ATR',0):.2f}")
col4.metric("Stop 2×ATR", f"${latest['Close'] - 2*latest.get('ATR',0):.2f}")

tabs = st.tabs(["Preço + Vol", "RSI", "MACD", "Bollinger", "Stochastic", "CCI", "ADX", "Ichimoku", "Volume Profile", "SuperTrend", "Williams %R", "MFI", "🔙 Backtesting"])

# ==================== ABAS COM EXPLICAÇÕES ====================
with tabs[0]:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA200", line=dict(color="blue")))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,149,237,0.6)"), secondary_y=True)
    fig.update_layout(title=f"{selected} - Gráfico Diário", height=650)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar Preço + SMA para Swing Trade"):
        st.markdown("""
        **Compra forte**: Preço > SMA50 > SMA200 (tendência de alta clara)  
        **Compra**: Preço cruza para cima da SMA50  
        **Venda**: Preço < SMA50 < SMA200  
        **Dica**: Usa sempre com SuperTrend e ADX para confirmar força da tendência
        """)

with tabs[1]:
    fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
    fig.add_hline(70, line_dash="dash", line_color="red")
    fig.add_hline(30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (14)", yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar RSI para Swing Trade"):
        st.markdown("""
        **Compra**: RSI < 35 (sobrevendido)  
        **Compra extra forte**: RSI < 30 + cruzamento para cima  
        **Venda**: RSI > 65 (sobrecomprado)  
        **Dica**: Espera RSI voltar acima de 50 após divergência para confirmar entrada
        """)

with tabs[2]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histograma", marker_color=np.where(df['MACD_Hist']>=0,'green','red')))
    fig.update_layout(title="MACD", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar MACD para Swing Trade"):
        st.markdown("""
        **Compra**: Linha MACD cruza para cima da linha Signal + histograma positivo  
        **Venda**: Linha MACD cruza para baixo da linha Signal  
        **Dica**: Quanto maior o histograma, mais forte o movimento
        """)

with tabs[3]:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="Upper", line=dict(color="red",dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="Mid", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="Lower", line=dict(color="green",dash="dash")))
    fig.update_layout(title="Bollinger Bands (20,2)", height=450)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar Bollinger Bands para Swing Trade"):
        st.markdown("""
        **Compra**: Preço toca banda inferior + tendência de alta (acima SMA50)  
        **Venda**: Preço toca banda superior + tendência de baixa  
        **Squeeze**: Bandas muito estreitas → espera explosão de volatilidade
        """)

with tabs[4]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name="%K"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name="%D"))
    fig.add_hline(80, line_dash="dash", line_color="red")
    fig.add_hline(20, line_dash="dash", line_color="green")
    fig.update_layout(title="Stochastic (14,3,3)", yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar Stochastic para Swing Trade"):
        st.markdown("""
        **Compra**: %K cruza para cima de %D abaixo de 40  
        **Venda**: %K cruza para baixo de %D acima de 60  
        **Dica**: Muito bom em mercados laterais
        """)

with tabs[5]:
    fig = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name="CCI"))
    fig.add_hline(100, line_dash="dash", line_color="red")
    fig.add_hline(-100, line_dash="dash", line_color="green")
    fig.update_layout(title="CCI (20)", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar CCI para Swing Trade"):
        st.markdown("""
        **Compra**: CCI < -100 (sobrevendido)  
        **Venda**: CCI > 100 (sobrecomprado)  
        **Dica**: Divergências com preço são sinais muito fortes
        """)

with tabs[6]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX", line=dict(color="purple",width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], name="+DI", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], name="-DI", line=dict(color="red")))
    fig.add_hline(25, line_dash="dash", line_color="black")
    fig.update_layout(title="ADX +DI/-DI", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar ADX para Swing Trade"):
        st.markdown("""
        **Tendência forte**: ADX > 25  
        **Compra**: ADX > 25 e +DI > -DI  
        **Venda**: ADX > 25 e -DI > +DI  
        **Dica**: Se ADX < 20 evita operar (mercado sem tendência)
        """)

with tabs[7]:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], name="Tenkan", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], name="Kijun", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df['SenkouA'], name="Senkou A", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df['SenkouB'], name="Senkou B", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Chikou'], name="Chikou", line=dict(color="gray", dash="dot")))
    fig.update_layout(title="Ichimoku Cloud", height=550)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar Ichimoku para Swing Trade"):
        st.markdown("""
        **Compra**: Preço acima da nuvem + Tenkan > Kijun  
        **Venda**: Preço abaixo da nuvem  
        **Dica**: Nuvem grossa = suporte/resistência forte
        """)

with tabs[8]:
    st.subheader("Volume Profile (últimos 252 dias)")
    df_vp = df.tail(252).copy()
    if len(df_vp) > 10:
        p_min, p_max = df_vp['Low'].min(), df_vp['High'].max()
        bins = np.linspace(p_min, p_max, 31)
        bin_mids = (bins[:-1] + bins[1:]) / 2
        vols = []
        for i in range(len(bins)-1):
            mask = (df_vp['Close'] >= bins[i]) & (df_vp['Close'] < bins[i+1])
            vols.append(df_vp['Volume'][mask].sum())
        fig_vp = go.Figure(go.Bar(x=vols, y=bin_mids, orientation='h', marker_color='rgba(55,83,109,0.85)'))
        fig_vp.update_layout(title="Volume Profile", xaxis_title="Volume", yaxis_title="Preço", height=600)
        st.plotly_chart(fig_vp, use_container_width=True)
    with st.expander("📋 Como analisar Volume Profile para Swing Trade"):
        st.markdown("""
        **Compra**: Preço perto de zona de alto volume (suporte forte)  
        **Venda**: Preço perto de zona de alto volume (resistência)  
        **Dica**: O POC (ponto de maior volume) é excelente nível de stop
        """)

with tabs[9]:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name="SuperTrend", line=dict(color="purple", width=3)))
    fig.update_layout(title="SuperTrend (10,3)", height=500)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar SuperTrend para Swing Trade"):
        st.markdown("""
        **Compra**: Preço acima da linha SuperTrend (trailing stop)  
        **Venda**: Preço cruza para baixo da linha SuperTrend  
        **Dica**: Melhor indicador de tendência atual (usa como stop dinâmico!)
        """)

with tabs[10]:
    fig = go.Figure(go.Scatter(x=df.index, y=df['Williams_%R'], name="Williams %R"))
    fig.add_hline(-20, line_dash="dash", line_color="red")
    fig.add_hline(-80, line_dash="dash", line_color="green")
    fig.update_layout(title="Williams %R (14)", yaxis_range=[-100,0], height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar Williams %R para Swing Trade"):
        st.markdown("""
        **Compra**: Williams %R < -80 (extremamente sobrevendido)  
        **Venda**: Williams %R > -20 (extremamente sobrecomprado)  
        **Dica**: Reage mais rápido que o RSI em reversões
        """)

with tabs[11]:
    fig = go.Figure(go.Scatter(x=df.index, y=df['MFI'], name="MFI"))
    fig.add_hline(80, line_dash="dash", line_color="red")
    fig.add_hline(20, line_dash="dash", line_color="green")
    fig.update_layout(title="MFI (14)", yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Como analisar MFI para Swing Trade"):
        st.markdown("""
        **Compra**: MFI < 20 (dinheiro a sair + preço a cair = divergência)  
        **Venda**: MFI > 80  
        **Dica**: Detecta divergências que o RSI não vê (fluxo de dinheiro)
        """)

with tabs[12]:
    st.subheader("🔙 Backtesting Histórico")
    if st.button("▶️ Executar Backtest Completo", type="primary"):
        with st.spinner("A correr backtest..."):
            hist_signals = []
            for i in range(200, len(df)):
                sub = df.iloc[:i+1]
                sig, _ = generate_signal(sub)
                hist_signals.append(sig)
            bt_df = df.iloc[200:].copy()
            bt_df['Signal'] = hist_signals
            capital = 10000.0
            position = 0
            entry_price = 0.0
            atr_entry = 0.0
            equity = [capital]
            trades_pnl = []
            for i in range(len(bt_df)):
                row = bt_df.iloc[i]
                price = row['Close']
                sig = row['Signal']
                atr = row.get('ATR', 0)
                if position == 0 and "Compra" in sig:
                    position = 1
                    entry_price = price
                    atr_entry = atr
                elif position == 1:
                    stop = entry_price - 2 * atr_entry if atr_entry > 0 else entry_price * 0.95
                    if price <= stop or "Venda" in sig:
                        pnl = (price - entry_price) / entry_price
                        trades_pnl.append(pnl)
                        capital *= (1 + pnl)
                        position = 0
                equity.append(capital)
            num_trades = len(trades_pnl)
            winrate = len([p for p in trades_pnl if p > 0]) / num_trades * 100 if num_trades > 0 else 0
            total_ret = (capital / 10000 - 1) * 100
            st.success(f"Capital Final: **${capital:,.2f}** ({total_ret:+.1f}%) | Trades: **{num_trades}** | Win Rate: **{winrate:.1f}%**")
            fig_eq = go.Figure(go.Scatter(x=bt_df.index, y=equity[1:], name="Equity"))
            fig_eq.update_layout(title="Curva de Equity", height=400)
            st.plotly_chart(fig_eq, use_container_width=True)

st.caption("🚀 App ULTIMATE por Grok • 12 Indicadores com guias práticos • Apenas educativo • Não é aconselhamento financeiro")