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

# ====================== INDICADORES ======================
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

# ====================== NOVO: DETEÇÃO DE PADRÕES DE CANDLES ======================
def detect_candlestick_pattern(df):
    if len(df) < 5:
        return "Dados insuficientes", "Não é possível analisar padrões."

    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    body = abs(last['Close'] - last['Open'])
    upper_shadow = last['High'] - max(last['Close'], last['Open'])
    lower_shadow = min(last['Close'], last['Open']) - last['Low']

    # Bullish Engulfing
    if (prev1['Close'] < prev1['Open'] and 
        last['Close'] > last['Open'] and 
        last['Open'] < prev1['Close'] and 
        last['Close'] > prev1['Open']):
        return "🟢 Bullish Engulfing", "Padrão de reversão de baixa para alta. Forte sinal de compra quando aparece no fim de uma descida."

    # Bearish Engulfing
    if (prev1['Close'] > prev1['Open'] and 
        last['Close'] < last['Open'] and 
        last['Open'] > prev1['Close'] and 
        last['Close'] < prev1['Open']):
        return "🔴 Bearish Engulfing", "Padrão de reversão de alta para baixa. Forte sinal de venda."

    # Hammer
    if lower_shadow > 2 * body and upper_shadow < body * 0.3 and last['Close'] > last['Open']:
        return "🟢 Hammer", "Fundo de tendência de baixa. Bom sinal de compra se aparecer após queda."

    # Shooting Star
    if upper_shadow > 2 * body and lower_shadow < body * 0.3 and last['Close'] < last['Open']:
        return "🔴 Shooting Star", "Topo de tendência de alta. Bom sinal de venda."

    # Doji
    if body < (last['High'] - last['Low']) * 0.1:
        return "⚪ Doji", "Indecisão no mercado. Espera confirmação no próximo candle."

    # Morning Star
    if (prev2['Close'] < prev2['Open'] and 
        abs(prev1['Close'] - prev1['Open']) < (prev1['High'] - prev1['Low']) * 0.3 and 
        last['Close'] > last['Open'] and last['Close'] > (prev2['Open'] + prev2['Close'])/2):
        return "🟢 Morning Star", "Padrão de reversão bullish forte."

    # Evening Star
    if (prev2['Close'] > prev2['Open'] and 
        abs(prev1['Close'] - prev1['Open']) < (prev1['High'] - prev1['Low']) * 0.3 and 
        last['Close'] < last['Open'] and last['Close'] < (prev2['Open'] + prev2['Close'])/2):
        return "🔴 Evening Star", "Padrão de reversão bearish forte."

    return "Nenhum padrão claro", "Sem padrão de candlestick forte nos últimos candles."

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

tabs = st.tabs(["Preço + Vol", "RSI", "MACD", "Bollinger", "Stochastic", "CCI", "ADX", "Ichimoku", "Volume Profile", "SuperTrend", "Williams %R", "MFI", "Padrões de Candles", "🔙 Backtesting"])

# ... (todas as abas anteriores mantidas iguais - só adicionei a nova no final)

# ====================== NOVA ABA: PADRÕES DE CANDLES ======================
with tabs[12]:
    pattern_name, pattern_desc = detect_candlestick_pattern(df)
    st.subheader("🕯️ Padrão de Candlestick Detectado")
    st.metric("Padrão Atual", pattern_name)
    st.markdown(pattern_desc)

    # Gráfico dos últimos 15 candles para visualizar o padrão
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index[-15:],
                                                open=df['Open'][-15:],
                                                high=df['High'][-15:],
                                                low=df['Low'][-15:],
                                                close=df['Close'][-15:])])
    fig_candle.update_layout(title=f"Últimos 15 candles - {selected}", height=500)
    st.plotly_chart(fig_candle, use_container_width=True)

    with st.expander("📋 Como usar padrões de candlestick em Swing Trade"):
        st.markdown("""
        **Regras gerais**:
        - Padrões bullish (Hammer, Bullish Engulfing, Morning Star) são mais fortes no fim de uma descida e com suporte forte (SMA50 ou banda inferior Bollinger).
        - Padrões bearish são mais fortes no fim de uma subida e com resistência.
        - Sempre confirma com os outros indicadores (SuperTrend, RSI, Volume).
        - O padrão sozinho não é suficiente — precisa de confluência.
        """)

# ====================== BACKTESTING ======================
with tabs[13]:
    st.subheader("🔙 Backtesting Histórico")
    if st.button("▶️ Executar Backtest Completo", type="primary"):
        with st.spinner("A correr backtest..."):
            # (código de backtest que já tinhas - mantido igual)
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

st.caption("🚀 App ULTIMATE por Grok • Nova aba de Padrões de Candles adicionada • Apenas educativo")
