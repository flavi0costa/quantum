import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🚀 SUPER QUANT BOT — S&P500 AI SCANNER")

# ==============================
# LOAD S&P500 LIST
# ==============================
@st.cache_data
def load_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist()

symbols = load_sp500()

# ==============================
# DOWNLOAD DATA
# ==============================
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    df = df.dropna()
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

    X = df[["RSI", "MACD", "EMA"]]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

# ==============================
# SIGNAL GENERATION
# ==============================
def get_signal(model, df):
    latest = df[["RSI", "MACD", "EMA"]].iloc[-1:]
    pred = model.predict(latest)[0]
    return "BUY" if pred == 1 else "SELL"

# ==============================
# UI CONTROLS
# ==============================
selected = st.selectbox("Select Stock", symbols)

df = load_data(selected)
df = add_indicators(df)

model = train_model(df)
signal = get_signal(model, df)

# ==============================
# SHOW SIGNAL
# ==============================
st.subheader(f"📊 AI Signal for {selected}")

if signal == "BUY":
    st.success("🟢 BUY SIGNAL")
else:
    st.error("🔴 SELL SIGNAL")

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

    for i, sym in enumerate(symbols[:100]):  # limit for speed
        try:
            data = load_data(sym)
            data = add_indicators(data)

            if len(data) < 50:
                continue

            m = train_model(data)
            sig = get_signal(m, data)

            results.append({
                "Symbol": sym,
                "Signal": sig,
                "Price": data["Close"].iloc[-1],
                "RSI": round(data["RSI"].iloc[-1], 2)
            })
        except:
            pass

        progress.progress((i + 1) / 100)

    results_df = pd.DataFrame(results)

    st.dataframe(results_df.sort_values("RSI", ascending=False))