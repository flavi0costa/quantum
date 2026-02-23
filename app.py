import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Trade S&P500 PRO", layout="wide")
st.title("🚀 Swing Trade PRO - Top 100 S&P 500 (9 Indicadores + Backtest)")

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

# ====================== INDICADORES ======================
def calculate_indicators(ticker):
    yf_ticker = ticker.replace('.', '-')
    df = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        return None

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

    # Bollinger
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # Stochastic
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad)

    # ADX + TR (usado também para ATR)
    df['TR'] = pd.concat([df['High']-df['Low'], 
                          abs(df['High']-df['Close'].shift()), 
                          abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']), df['High'] - df['High'].shift(), 0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()), df['Low'].shift() - df['Low'], 0)
    df['+DM'] = df['+DM'].clip(lower=0)
    df['-DM'] = df['-DM'].clip(lower=0)
    alpha = 1/14
    tr_smooth = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-DI'] = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()

    # === NOVOS ===
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    # ATR
    df['ATR'] = df['TR'].rolling(14).mean()
    # Ichimoku
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SenkouB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(52)
    df['Chikou'] = df['Close'].shift(-26)

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

    stoch_k = latest.get('Stoch_K', np.nan)
    stoch_d = latest.get('Stoch_D', np.nan)
    prev_stoch_k = prev.get('Stoch_K', np.nan)
    prev_stoch_d = prev.get('Stoch_D', np.nan)
    cci = latest.get('CCI', np.nan)
    adx = latest.get('ADX', np.nan)
    plus_di = latest.get('+DI', np.nan)
    minus_di = latest.get('-DI', np.nan)
    obv = latest.get('OBV', np.nan)
    prev_obv = prev.get('OBV', np.nan)

    score = 0

    if pd.notna(price) and pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200: score += 3
        elif price < sma50 < sma200: score -= 3

    if pd.notna(rsi):
        if rsi < 35: score += 2
        elif rsi > 65: score -= 2

    if pd.notna(macd) and pd.notna(signal_line) and pd.notna(prev_macd) and pd.notna(prev_signal):
        if macd > signal_line and prev_macd <= prev_signal: score += 3
        elif macd < signal_line and prev_macd >= prev_signal: score -= 3

    if pd.notna(hist):
        score += 1 if hist > 0 else -1

    if pd.notna(stoch_k) and pd.notna(stoch_d) and pd.notna(prev_stoch_k) and pd.notna(prev_stoch_d):
        if stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d:
            score += 3 if stoch_k < 40 else 1
        elif stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d:
            score -= 3 if stoch_k > 60 else -1

    if 'BB_Lower' in latest and 'BB_Upper' in latest:
        bb_lower = latest['BB_Lower']
        bb_upper = latest['BB_Upper']
        if pd.notna(bb_lower) and pd.notna(bb_upper) and (bb_upper - bb_lower) > 0:
            bb_position = (price - bb_lower) / (bb_upper - bb_lower)
            if bb_position < 0.20 and pd.notna(sma50) and price > sma50: score += 2
            elif bb_position > 0.80 and pd.notna(sma50) and price < sma50: score -= 2

    if pd.notna(cci):
        if cci < -100: score += 2
        elif cci > 100: score -= 2

    if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
        if adx > 25:
            if plus_di > minus_di: score += 2
            else: score -= 2

    # OBV
    if pd.notna(obv) and pd.notna(prev_obv):
        if obv > prev_obv: score += 1
        elif obv < prev_obv: score -= 1

    if score >= 10: return "🟢 Compra Forte", score
    if score >= 6:  return "🟢 Compra", score
    if score <= -10: return "🔴 Venda Forte", score
    if score <= -6:  return "🔴 Venda", score
    return "⚪ Neutro", score

# ====================== INTERFACE ======================
if st.sidebar.button("🔄 Atualizar Tudo"):
    st.cache_data.clear()
    st.rerun()

top_df, _ = get_top_liquid_stocks(100, 30)

if 'signals_df' not in st.session_state or st.sidebar.button("Recalcular Sinais"):
    with st.spinner("Calculando 9 indicadores + sinais (30-70 seg)..."):
        signals = []
        data_cache = {}
        skipped = 0
        for ticker in top_df['Symbol']:
            try:
                df = calculate_indicators(ticker)
                if df is not None:
                    signal_text, score = generate_signal(df)
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]

                    bb_pos = np.nan
                    if 'BB_Lower' in latest and 'BB_Upper' in latest:
                        bl, bu = latest['BB_Lower'], latest['BB_Upper']
                        if pd.notna(bl) and pd.notna(bu) and (bu - bl) != 0:
                            bb_pos = (latest['Close'] - bl) / (bu - bl)

                    obv_trend = "↑ Rising" if pd.notna(latest.get('OBV')) and pd.notna(prev.get('OBV')) and latest['OBV'] > prev['OBV'] else "↓ Falling"

                    signals.append({
                        'Símbolo': ticker,
                        'Empresa': top_df[top_df['Symbol']==ticker]['Security'].iloc[0],
                        'Preço': round(latest['Close'],2),
                        'Variação %': round((latest['Close']/df.iloc[-2]['Close']-1)*100,2),
                        'Vol. Médio': f"{int(top_df[top_df['Symbol']==ticker]['Avg_Daily_Volume'].iloc[0]):,}",
                        'RSI': round(latest['RSI'],1) if pd.notna(latest.get('RSI')) else "N/A",
                        'Stoch_K': round(latest.get('Stoch_K'),1) if pd.notna(latest.get('Stoch_K')) else "N/A",
                        'CCI': round(latest.get('CCI'),1) if pd.notna(latest.get('CCI')) else "N/A",
                        'ADX': round(latest.get('ADX'),1) if pd.notna(latest.get('ADX')) else "N/A",
                        'OBV': obv_trend,
                        'ATR': round(latest.get('ATR'),2) if pd.notna(latest.get('ATR')) else "N/A",
                        'BB_%B': round(bb_pos*100,1) if pd.notna(bb_pos) else "N/A",
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

signals_df = st.session_state.signals_df

st.subheader("📊 Top 100 + 9 Indicadores + Sinais")
st.dataframe(
    signals_df.sort_values('Score', ascending=False),
    column_config={
        "Sinal": st.column_config.TextColumn("Sinal", width="medium"),
        "Score": st.column_config.NumberColumn("Score", format="%d"),
        "Variação %": st.column_config.NumberColumn(format="%.2f%%"),
        "OBV": st.column_config.TextColumn("OBV"),
        "ATR": st.column_config.NumberColumn("ATR", format="%.2f"),
    },
    use_container_width=True,
    height=700
)

# ====================== DETALHE ======================
st.subheader("📈 Detalhe + Stops + Backtest")
selected = st.selectbox("Escolhe ação:", options=signals_df['Símbolo'], index=0)

df = st.session_state.data_cache[selected]
latest = df.iloc[-1]
signal_text, _ = generate_signal(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Preço", f"${latest['Close']:.2f}", f"{(latest['Close']/df.iloc[-2]['Close']-1)*100:+.2f}%")
col2.metric("RSI", f"{latest['RSI']:.1f}")
col3.metric("Sinal", signal_text)
col4.metric("ATR (14)", f"{latest['ATR']:.2f}" if pd.notna(latest.get('ATR')) else "N/A")
stop_price = latest['Close'] - 2 * latest['ATR'] if pd.notna(latest.get('ATR')) else np.nan
col5.metric("Stop Loss Sugerido (2×ATR)", f"${stop_price:.2f}" if pd.notna(stop_price) else "N/A")

tabs = st.tabs(["Preço + Vol", "RSI", "MACD", "Bollinger", "Stochastic", "CCI", "ADX", "Ichimoku", "Volume Profile", "🔙 Backtesting"])

with tabs[0]:
    fig_pv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pv.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(color="orange")))
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA200", line=dict(color="blue")))
    fig_pv.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,149,237,0.6)"), secondary_y=True)
    fig_pv.update_layout(title=f"{selected} - Diário", height=650)
    st.plotly_chart(fig_pv, use_container_width=True)

with tabs[1]:  # RSI (igual anterior)
    fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
    fig.add_hline(70, line_dash="dash", line_color="red")
    fig.add_hline(30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (14)", yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:  # MACD (igual)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Hist", marker_color=np.where(df['MACD_Hist']>=0,'green','red')))
    fig.update_layout(title="MACD", height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # Bollinger (igual)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="Upper", line=dict(color="red",dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="Mid", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="Lower", line=dict(color="green",dash="dash")))
    fig.update_layout(title="Bollinger Bands (20,2)", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:  # Stochastic (igual)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name="%K"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name="%D"))
    fig.add_hline(80, line_dash="dash", line_color="red")
    fig.add_hline(20, line_dash="dash", line_color="green")
    fig.update_layout(title="Stochastic (14,3,3)", yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:  # CCI (igual)
    fig = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name="CCI"))
    fig.add_hline(100, line_dash="dash", line_color="red")
    fig.add_hline(-100, line_dash="dash", line_color="green")
    fig.update_layout(title="CCI (20)", height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[6]:  # ADX (igual)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX", line=dict(color="purple",width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], name="+DI", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], name="-DI", line=dict(color="red")))
    fig.add_hline(25, line_dash="dash", line_color="black")
    fig.update_layout(title="ADX +DI/-DI", height=350)
    st.plotly_chart(fig, use_container_width=True)

with tabs[7]:  # Ichimoku
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], name="Tenkan", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], name="Kijun", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df['SenkouA'], name="Senkou A", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=df['SenkouB'], name="Senkou B", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df.index, y=df['Chikou'], name="Chikou", line=dict(color="gray", dash="dot")))
    fig.update_layout(title="Ichimoku Cloud Completo", height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("☁️ Nuvem: preço acima = bullish | abaixo = bearish")

with tabs[8]:  # Volume Profile
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

with tabs[9]:  # Backtesting
    st.subheader("🔙 Backtesting Histórico (Long-only + Stop 2×ATR)")
    if st.button("▶️ Executar Backtest Completo", type="primary"):
        with st.spinner("Simulando 252 dias de trades..."):
            # Calcula sinais históricos
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
                    atr_entry = atr if pd.notna(atr) else 0
                
                elif position == 1:
                    stop = entry_price - 2 * atr_entry if atr_entry > 0 else entry_price * 0.95
                    if price <= stop or "Venda" in sig:
                        pnl = (price - entry_price) / entry_price
                        trades_pnl.append(pnl)
                        capital *= (1 + pnl)
                        position = 0
                equity.append(capital)
            
            # Métricas
            num_trades = len(trades_pnl)
            winrate = len([p for p in trades_pnl if p > 0]) / num_trades * 100 if num_trades > 0 else 0
            total_ret = (capital / 10000 - 1) * 100
            max_dd = min(np.minimum.accumulate(equity) / np.maximum.accumulate(equity) - 1) * 100 if len(equity) > 1 else 0
            
            st.success(f"Capital Final: **${capital:,.2f}** ({total_ret:+.1f}%) | Trades: **{num_trades}** | Win Rate: **{winrate:.1f}%** | Max DD: **{max_dd:.1f}%**")
            
            fig_eq = go.Figure(go.Scatter(x=bt_df.index, y=equity[1:], name="Equity", line=dict(color="green", width=3)))
            fig_eq.update_layout(title="Curva de Equity", height=400)
            st.plotly_chart(fig_eq, use_container_width=True)

st.caption("App PRO por COSTA • 9 Indicadores + Backtest real + Stops ATR • Apenas educativo • Não é conselho financeiro")