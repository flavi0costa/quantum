import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

st.title("🧠 SUPER QUANT TERMINAL PRO")

SYMBOLS = ["AAPL","TSLA","NVDA","MSFT","AMD","META","AMZN"]

# ========================
# AI MODEL
# ========================
@st.cache_resource
def train_ai():
    df = yf.download("AAPL", period="5y")
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["EMA"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()

    df["Future"] = df["Close"].shift(-5)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)

    df.dropna(inplace=True)

    X = df[["RSI","MACD","EMA","ATR"]]
    y = df["Target"]

    model = RandomForestClassifier()
    model.fit(X,y)

    return model

model = train_ai()

# ========================
# SIDEBAR CONTROLS
# ========================
st.sidebar.title("⚙️ Controls")

capital = st.sidebar.number_input("Capital", value=500)
risk = st.sidebar.slider("Risk %",1,10,2)/100

scan_button = st.sidebar.button("🚀 Scan Market")

# ========================
# FUNCTIONS
# ========================

def smart_money(df):
    last = df.iloc[-1]
    prev_low = df["Low"].rolling(20).min().iloc[-2]
    prev_high = df["High"].rolling(20).max().iloc[-2]

    stop_hunt = last["Low"] < prev_low and last["Close"] > prev_low
    fake_breakout = last["High"] > prev_high and last["Close"] < prev_high

    return stop_hunt or fake_breakout

def explosion_signal(df):
    atr = df["High"] - df["Low"]
    compression = atr.iloc[-1] < atr.rolling(20).mean().iloc[-1]*0.6
    volume = df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]*1.5
    return compression and volume

def ai_probability(df):
    last = df.iloc[-1]
    features = [[
        last["RSI"],
        last["MACD"],
        last["EMA"],
        last["ATR"]
    ]]
    return model.predict_proba(features)[0][1]

def position_size(capital, risk, stop):
    return (capital*risk)/stop

# ========================
# MAIN SCANNER
# ========================
if scan_button:

    results = []

    with st.spinner("Scanning market..."):

        for sym in SYMBOLS:

            df = yf.download(sym, period="6mo")

            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["Close"]).macd()
            df["EMA"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
            df["ATR"] = ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range()

            df.dropna(inplace=True)

            prob = ai_probability(df)
            smart = smart_money(df)
            explode = explosion_signal(df)

            score = prob*60
            if smart: score+=20
            if explode: score+=20

            last_price = df["Close"].iloc[-1]
            stop = df["ATR"].iloc[-1]*2

            size = position_size(capital, risk, stop)

            results.append({
                "Symbol": sym,
                "AI Prob": round(prob*100,1),
                "Smart Money": smart,
                "Pre-Explosion": explode,
                "Score": round(score),
                "Price": round(last_price,2),
                "Position Size": int(size)
            })

    df_results = pd.DataFrame(results).sort_values("Score", ascending=False)

    st.subheader("🏆 Opportunity Ranking")
    st.dataframe(df_results, use_container_width=True)

    # ========================
    # BEST TRADE VIEW
    # ========================
    best = df_results.iloc[0]["Symbol"]
    st.subheader(f"📈 Best Opportunity: {best}")

    df = yf.download(best, period="6mo")

    fig = px.line(df, y="Close", title=best)
    st.plotly_chart(fig, use_container_width=True)

    st.success("Scan completed!")

else:
    st.info("Press 'Scan Market' to start analysis.")