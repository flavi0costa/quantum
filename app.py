import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🚀 SUPER QUANT BOT — S&P500 AI SCANNER (Final Robust Version)")

# ==============================
# FIXED S&P500 LIST
# ==============================
symbols = [
    "AAPL","MSFT","AMZN","TSLA","GOOGL","NVDA","META",
    "BRK-B","JPM","V","UNH","HD","PG","MA","DIS","BAC",
    "VZ","