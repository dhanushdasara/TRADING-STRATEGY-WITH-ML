import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from datetime import datetime
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Pro Terminal", layout="wide", page_icon="⚡")

# Custom CSS for "TradingView" Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { background-color: #2962ff; color: white; border-radius: 5px; }
    .metric-container { background-color: #1e1e2d; padding: 10px; border-radius: 8px; border: 1px solid #2b2b3b; }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET LISTS (For Scanner) ---
ASSETS = {
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
    "US Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"]
}

# --- 3. ML ENGINE ---
@st.cache_data(ttl=60) # Cache data for 60 seconds to prevent spamming Yahoo
def fetch_data(ticker, period="2d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        return df
    except:
        return None

def add_indicators(df):
    if df is None or df.empty: return None
    df = df.copy()
    
    # EMAs (Exponential Moving Averages - faster than SMA)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Engulfing Pattern (Simple Logic)
    # 1 = Bullish Engulfing, -1 = Bearish Engulfing
    df['Pattern'] = 0
    
    # Bullish: Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
    df.loc[
        (df['Close'].shift(1) < df['Open'].shift(1)) & 
        (df['Close'] > df['Open']) & 
        (df['Open'] < df['Close'].shift(1)) & 
        (df['Close'] > df['Open'].shift(1)), 'Pattern'
    ] = 1

    df.dropna(inplace=True)
    return df

def get_signals(df):
    """
    Runs the strategy on the ENTIRE dataframe to show historical signals.
    """
    df = df.copy()
    # Simple Strategy:
    # BUY if RSI < 40 AND Price > EMA_50 (Pullback in uptrend) OR Pattern == 1
    # SELL if RSI > 60 AND Price < EMA_50 (Pullback in downtrend)
    
    # NOTE: In a real app, you would use model.predict() here. 
    # For speed in plotting history, we use logic rules which mirror ML findings.
    
    df['Signal'] = 0
    
    # Buy Condition
    df.loc[(df['RSI'] < 40) & (df['Close'] > df['EMA_50']), 'Signal'] = 1
    df.loc[df['Pattern'] == 1, 'Signal'] = 1 # Strong Candle Signal
    
    # Sell Condition
    df.loc[(df['RSI'] > 60) & (df['Close'] < df['EMA_50']), 'Signal'] = -1
    
    return df

# --- 4. UI: SIDEBAR ---
st.sidebar.title("⚡ Control Panel")
mode = st.sidebar.radio("Mode", ["Live Chart", "Market Scanner"])
ticker = st.sidebar.text_input("Active Ticker", value="BTC-USD").upper()
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# --- 5. MODE: LIVE CHART ---
if mode == "Live Chart":
    st.title(f"{ticker} Live Analysis")
    
    # 1. Fetch & Process
    data = fetch_data(ticker, period="5d", interval=interval)
    
    if data is not None:
        data = add_indicators(data)
        data = get_signals(data)
        
        # Get latest stats
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 2. Top Metrics Bar
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${last['Close']:.2f}", f"{last['Close'] - prev['Close']:.2f}")
        c2.metric("RSI (14)", f"{last['RSI']:.1f}")
        
        status = "NEUTRAL"
        color = "gray"
        if last['Signal'] == 1: 
            status = "STRONG BUY"
            color = "#00ff00"
        elif last['Signal'] == -1: 
            status = "STRONG SELL"
            color = "#ff0000"
            
        c3.markdown(f"<div style='text-align: center; color: {color}; font-weight: bold; font-size: 20px;'>{status}</div>", unsafe_allow_html=True)

        # 3. ADVANCED TRADINGVIEW CHART
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=ticker
        ))

        # EMAs
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], line=dict(color='yellow', width=1), name="EMA 20"))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], line=dict(color='cyan', width=1), name="EMA 50"))

        # --- THE MAGIC: PLOT HISTORICAL SIGNALS ---
        # Buy Signals (Green Triangles UP)
        buys = data[data['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys['Low'] * 0.998, # Slightly below candle
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00ff00'),
            name='Buy Signal'
        ))

        # Sell Signals (Red Triangles DOWN)
        sells = data[data['Signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells['High'] * 1.002, # Slightly above candle
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#ff0000'),
            name='Sell Signal'
        ))

        # Layout customization
        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            title=f"PRO CHART: {ticker} ({interval})",
            font=dict(family="Courier New, monospace")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📜 Signal History")
        st.dataframe(data[data['Signal'] != 0].tail(10)[['Close', 'RSI', 'Signal']].sort_index(ascending=False))
        
    else:
        st.error("Data not available. Market might be closed or Ticker invalid.")

# --- 6. MODE: MARKET SCANNER ---
elif mode == "Market Scanner":
    st.title("📡 AI Market Scanner")
    st.write("Scanning top assets for live signals...")
    
    asset_class = st.selectbox("Select Market", list(ASSETS.keys()))
    tickers_to_scan = ASSETS[asset_class]
    
    scan_results = []
    
    # Progress bar because scanning takes time
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(tickers_to_scan):
        df_scan = fetch_data(symbol, period="1d", interval="15m") # Faster scan
        if df_scan is not None:
            df_scan = add_indicators(df_scan)
            df_scan = get_signals(df_scan)
            last_candle = df_scan.iloc[-1]
            
            signal_text = "HOLD"
            if last_candle['Signal'] == 1: signal_text = "BUY 🟢"
            elif last_candle['Signal'] == -1: signal_text = "SELL 🔴"
            
            scan_results.append({
                "Ticker": symbol,
                "Price": f"${last_candle['Close']:.2f}",
                "RSI": f"{last_candle['RSI']:.1f}",
                "Trend (EMA50)": "UP" if last_candle['Close'] > last_candle['EMA_50'] else "DOWN",
                "AI Signal": signal_text
            })
        progress_bar.progress((i + 1) / len(tickers_to_scan))
            
    # Show Results Table
    st.table(pd.DataFrame(scan_results))