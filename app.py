import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚀 SUPER QUANT BOT — S&P500 AI SCANNER (Robust Final Version)")

# ==============================
# FIXED S&P500 LIST (corrigida)
# ==============================
symbols = [
    "AAPL","MSFT","AMZN","TSLA","GOOGL","NVDA","META",
    "BRK-B","JPM","V","UNH","HD","PG","MA","DIS","BAC",
    "VZ","ADBE","NFLX","PYPL","KO","PEP","INTC","CSCO"
    # Acrescenta mais conforme necessário
]

@st.cache_data
def load_sp500():
    return symbols

symbols = load_sp500()

# ==============================
# DOWNLOAD DATA
# ==============================
@st.cache_data
def load_data(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d")
        if df.empty:
            return None
        df = df.dropna()
        return df
    except:
        return None

# ==============================
# FIX MULTIINDEX / CLEAN DATA
# ==============================
def fix_yfinance(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=["Close"])
    return df

# ==============================
# FEATURE ENGINEERING
# ==============================
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["EMA"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df = df.dropna()
    return df

# ==============================
# MACHINE LEARNING MODEL
# ==============================
def train_model(df):
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df = df.dropna(subset=["RSI","MACD","EMA","Target"])
    if len(df) < 20:
        return None
    X = df[["RSI","MACD","EMA"]]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X,y)
    return model

# ==============================
# SIGNAL GENERATION
# ==============================
def get_signal(model, df):
    latest = df[["RSI","MACD","EMA"]].iloc[-1:]
    pred = model.predict(latest)[0]
    return "BUY" if pred == 1 else "SELL"

# ==============================
# USER INTERFACE
# ==============================
selected = st.selectbox("Select Stock", symbols)

df = load_data(selected)
if df is None:
    st.warning(f"⚠️ Nenhum dado disponível para {selected}")
else:
    df = fix_yfinance(df)
    df = add_indicators(df)

    model = train_model(df)
    if model is not None:
        signal = get_signal(model, df)
    else:
        signal = "INSUFFICIENT DATA"

    st.subheader(f"📊 AI Signal for {selected}")
    if signal == "BUY":
        st.success("🟢 BUY SIGNAL")
    elif signal == "SELL":
        st.error("🔴 SELL SIGNAL")
    else:
        st.warning("⚠️ Not enough data to generate signal")

    # ==============================
    # PRICE CHART
    # ==============================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], name="EMA"))
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# FULL MARKET SCANNER
# ==============================
st.subheader("🔥 FULL S&P500 AI SCANNER")

if st.button("Run Market Scan"):
    results = []
    progress = st.progress(0)
    valid_symbols = 0

    for i, sym in enumerate(symbols):
        data = load_data(sym)
        if data is None or data.empty:
            progress.progress((i+1)/len(symbols))
            continue  # pula se sem dados
        try:
            data = fix_yfinance(data)
            data = add_indicators(data)
            model = train_model(data)
            if model is None:
                progress.progress((i+1)/len(symbols))
                continue
            sig = get_signal(model, data)
            results.append({
                "Symbol": sym,
                "Signal": sig,
                "Price": round(data["Close"].iloc[-1],2),
                "RSI": round(data["RSI"].iloc[-1],2)
            })
            valid_symbols += 1
        except:
            pass
        progress.progress((i+1)/len(symbols))

    if valid_symbols == 0:
        st.warning("⚠️ Nenhum ticker válido encontrado. Verifique os símbolos ou o período de dados.")
    else:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.sort_values("RSI", ascending=False))

        # ==============================
        # BACKTESTING VISUAL (SIMPLE)
        # ==============================
        st.subheader("📈 Backtesting Visual")
        top_buy = df_results[df_results["Signal"]=="BUY"].sort_values("RSI", ascending=False).head(5)
        if len(top_buy) > 0:
            fig2, ax2 = plt.subplots(figsize=(10,5))
            for sym in top_buy["Symbol"]:
                data = load_data(sym)
                data = fix_yfinance(data)
                data = add_indicators(data)
                ax2.plot(data.index, data["Close"], label=sym)
            ax2.set_title("Top 5 BUY Signals - Price History")
            ax2.set_ylabel("Price ($)")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("Nenhuma ação com sinal BUY disponível para backtesting")