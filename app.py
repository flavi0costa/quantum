import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🧠 SUPER QUANT TERMINAL")

SYMBOLS = ["AAPL","TSLA","NVDA","MSFT","AMD","META","AMZN"]

# =========================
# FIX UNIVERSAL YFINANCE
# =========================
def fix_yfinance(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    for col in ["Close","High","Low","Volume"]:
        df[col] = pd.Series(df[col].values, index=df.index)

    return df

# =========================
# AI TRAINING
# =========================
@st.cache_resource
def train_ai():

    df = yf.download("AAPL", period="5y", progress=False)
    df = fix_yfinance(df)

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["EMA"] = ta.trend.EMAIndicator(df["Close"],20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"]
    ).average_true_range()

    df["Future"] = df["Close"].shift(-5)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)

    df.dropna(inplace=True)

    X = df[["RSI","MACD","EMA","ATR"]]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X,y)

    return model

model = train_ai()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

capital = st.sidebar.number_input("Capital ($)", value=500)
risk = st.sidebar.slider("Risk %",1,10,2)/100

if st.sidebar.button("🚀 Scan Market"):

    results = []

    with st.spinner("Scanning market..."):

        for sym in SYMBOLS:

            df = yf.download(sym, period="6mo", progress=False)
            df = fix_yfinance(df)

            # Indicators
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["Close"]).macd()
            df["EMA"] = ta.trend.EMAIndicator(df["Close"],20).ema_indicator()
            df["ATR"] = ta.volatility.AverageTrueRange(
                df["High"], df["Low"], df["Close"]
            ).average_true_range()

            df.dropna(inplace=True)

            # AI probability
            last = df.iloc[-1]
            features = [[last["RSI"],last["MACD"],last["EMA"],last["ATR"]]]
            prob = model.predict_proba(features)[0][1]

            # Smart money
            prev_low = df["Low"].rolling(20).min().iloc[-2]
            prev_high = df["High"].rolling(20).max().iloc[-2]

            stop_hunt = last["Low"] < prev_low and last["Close"] > prev_low
            fake_break = last["High"] > prev_high and last["Close"] < prev_high
            smart = stop_hunt or fake_break

            # Explosion setup
            atr_range = df["High"] - df["Low"]
            compression = atr_range.iloc[-1] < atr_range.rolling(20).mean().iloc[-1]*0.6
            volume_spike = last["Volume"] > df["Volume"].rolling(20).mean().iloc[-1]*1.5
            explosion = compression and volume_spike

            # Score
            score = prob*60
            if smart: score+=20
            if explosion: score+=20

            # Position size
            stop = last["ATR"]*2
            size = (capital*risk)/stop

            results.append({
                "Symbol": sym,
                "AI %": round(prob*100,1),
                "Smart Money": smart,
                "Explosion Setup": explosion,
                "Score": round(score),
                "Price": round(last["Close"],2),
                "Position Size": int(size)
            })

    df_results = pd.DataFrame(results).sort_values("Score", ascending=False)

    st.subheader("🏆 Market Opportunities")
    st.dataframe(df_results, use_container_width=True)

    # Best trade chart
    best = df_results.iloc[0]["Symbol"]
    st.subheader(f"📈 Best Opportunity: {best}")

    df = yf.download(best, period="6mo", progress=False)
    df = fix_yfinance(df)

    fig = px.line(df, y="Close", title=best)
    st.plotly_chart(fig, use_container_width=True)

    st.success("Scan Complete!")

else:
    st.info("Click 'Scan Market' to start analysis.")