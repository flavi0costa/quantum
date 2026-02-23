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
        st.warning("⚠️ NASDAQ-100 não carregou. Usando só S&P 500.")
        return pd.DataFrame(columns=['Symbol', 'Security'])

# ====================== FILTROS ======================
st.sidebar.header("🔍 Filtros de Liquidez")
universe = st.sidebar.selectbox("Universo", ["S&P 500", "NASDAQ 100", "S&P 500 + NASDAQ 100 (Combinado)"])
min_vol = st.sidebar.slider("Volume Médio Diário Mínimo (milhões)", 1, 50, 5) * 1_000_000
min_price = st.sidebar.slider("Preço Mínimo ($)", 5, 100, 10)
only_buy = st.sidebar.checkbox("Mostrar apenas Sinais de Compra / Compra Forte", value=True)
max_show = st.sidebar.slider("Número máximo de ações a mostrar", 50, 500, 200)

if st.sidebar.button("🚀 Iniciar Scan / Recalcular", type="primary", use_container_width=True):
    with st.spinner("A escanear ações com liquidez... (30–90 segundos)"):
        if universe == "S&P 500":
            pool = get_sp500()
        elif universe == "NASDAQ 100":
            pool = get_nasdaq100()
        else:
            sp = get_sp500()
            nas = get_nasdaq100()
            pool = pd.concat([sp, nas]).drop_duplicates(subset='Symbol').reset_index(drop=True)

        @st.cache_data(ttl=3600)
        def get_liquid_stocks(pool, min_vol, min_price):
            tickers = pool['Symbol'].tolist()
            vol_data = yf.download(tickers, period="30d", progress=False, threads=True)['Volume']
            avg_vol = vol_data.mean()
            df_vol = pd.DataFrame({
                'Symbol': avg_vol.index,
                'Avg_Daily_Volume': avg_vol.values,
                'Security': pool.set_index('Symbol').loc[avg_vol.index, 'Security'].values
            })
            return df_vol[df_vol['Avg_Daily_Volume'] >= min_vol]

        top_df = get_liquid_stocks(pool, min_vol, min_price)

        signals = []
        data_cache = {}
        for ticker in top_df['Symbol'][:600]:
            try:
                df = calculate_indicators(ticker)  # função definida abaixo
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

# ====================== MOSTRAR RESULTADOS ======================
if 'signals_df' in st.session_state and not st.session_state.signals_df.empty:
    signals_df = st.session_state.signals_df.head(max_show)
    st.subheader(f"📊 {len(signals_df)} ações encontradas")
    st.dataframe(signals_df.sort_values('Score', ascending=False), use_container_width=True, height=700)

    csv = signals_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, f"swing_scanner_{universe}.csv", "text/csv")

# ====================== DETALHE + ABAS (todas restauradas) ======================
st.subheader("📈 Detalhe da Ação")
if 'signals_df' in st.session_state and not st.session_state.signals_df.empty:
    selected = st.selectbox("Escolhe uma ação:", options=st.session_state.signals_df['Símbolo'], index=0)
    if selected in st.session_state.data_cache:
        df = st.session_state.data_cache[selected]
        latest = df.iloc[-1]
        signal_text, _ = generate_signal(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Preço", f"${latest['Close']:.2f}", f"{(latest['Close']/df.iloc[-2]['Close']-1)*100:+.2f}%")
        col2.metric("Sinal", signal_text)
        col3.metric("ATR", f"{latest.get('ATR',0):.2f}")
        col4.metric("Stop 2×ATR", f"${latest['Close'] - 2*latest.get('ATR',0):.2f}")

        tabs = st.tabs(["Preço + Vol", "RSI", "MACD", "Bollinger", "Stochastic", "CCI", "ADX", "Ichimoku", "Volume Profile", "SuperTrend", "Williams %R", "MFI", "🔙 Backtesting"])

        with tabs[0]:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="SMA200", line=dict(color="blue")))
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,149,237,0.6)"), secondary_y=True)
            fig.update_layout(title=f"{selected} - Gráfico Diário", height=650)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar Preço + SMA"):
                st.markdown("**Compra**: Preço > SMA50 > SMA200  \n**Venda**: Preço < SMA50 < SMA200")

        with tabs[1]:
            fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
            fig.add_hline(70, line_dash="dash", line_color="red")
            fig.add_hline(30, line_dash="dash", line_color="green")
            fig.update_layout(title="RSI (14)", yaxis_range=[0,100], height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar RSI"):
                st.markdown("**Compra**: RSI < 35  \n**Venda**: RSI > 65")

        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal"))
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histograma", marker_color=np.where(df['MACD_Hist']>=0,'green','red')))
            fig.update_layout(title="MACD", height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar MACD"):
                st.markdown("**Compra**: MACD cruza acima da Signal  \n**Venda**: MACD cruza abaixo da Signal")

        with tabs[3]:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="Upper", line=dict(color="red",dash="dash")))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="Mid", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="Lower", line=dict(color="green",dash="dash")))
            fig.update_layout(title="Bollinger Bands (20,2)", height=450)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar Bollinger"):
                st.markdown("**Compra**: Preço toca banda inferior + tendência de alta")

        with tabs[4]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name="%K"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name="%D"))
            fig.add_hline(80, line_dash="dash", line_color="red")
            fig.add_hline(20, line_dash="dash", line_color="green")
            fig.update_layout(title="Stochastic (14,3,3)", yaxis_range=[0,100], height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar Stochastic"):
                st.markdown("**Compra**: %K cruza acima de %D abaixo de 40")

        with tabs[5]:
            fig = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name="CCI"))
            fig.add_hline(100, line_dash="dash", line_color="red")
            fig.add_hline(-100, line_dash="dash", line_color="green")
            fig.update_layout(title="CCI (20)", height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar CCI"):
                st.markdown("**Compra**: CCI < -100")

        with tabs[6]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX", line=dict(color="purple",width=3)))
            fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], name="+DI", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], name="-DI", line=dict(color="red")))
            fig.add_hline(25, line_dash="dash", line_color="black")
            fig.update_layout(title="ADX +DI/-DI", height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar ADX"):
                st.markdown("**Tendência forte**: ADX > 25  \n**Compra**: +DI > -DI")

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
            with st.expander("📋 Como analisar Ichimoku"):
                st.markdown("**Compra**: Preço acima da nuvem")

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
                fig_vp.update_layout(title="Volume Profile", height=600)
                st.plotly_chart(fig_vp, use_container_width=True)
            with st.expander("📋 Como analisar Volume Profile"):
                st.markdown("**Compra**: Preço perto de zona de alto volume (suporte)")

        with tabs[9]:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"))
            fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name="SuperTrend", line=dict(color="purple", width=3)))
            fig.update_layout(title="SuperTrend (10,3)", height=500)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar SuperTrend"):
                st.markdown("**Compra**: Preço acima da linha SuperTrend")

        with tabs[10]:
            fig = go.Figure(go.Scatter(x=df.index, y=df['Williams_%R'], name="Williams %R"))
            fig.add_hline(-20, line_dash="dash", line_color="red")
            fig.add_hline(-80, line_dash="dash", line_color="green")
            fig.update_layout(title="Williams %R (14)", yaxis_range=[-100,0], height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar Williams %R"):
                st.markdown("**Compra**: Williams %R < -80")

        with tabs[11]:
            fig = go.Figure(go.Scatter(x=df.index, y=df['MFI'], name="MFI"))
            fig.add_hline(80, line_dash="dash", line_color="red")
            fig.add_hline(20, line_dash="dash", line_color="green")
            fig.update_layout(title="MFI (14)", yaxis_range=[0,100], height=350)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📋 Como analisar MFI"):
                st.markdown("**Compra**: MFI < 20")

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

st.caption("🚀 SCANNER por Grok • Botão de scan + todas as abas restauradas • Apenas educativo")