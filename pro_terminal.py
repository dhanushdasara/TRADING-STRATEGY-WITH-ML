import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from lightweight_charts.widgets import StreamlitChart

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Master Terminal", page_icon="💎")

# --- 1. ROBUST DATA ENGINE ---
@st.cache_data(ttl=1)
def get_data(ticker, interval="5m", period="5d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        if 'date' in df.columns: df = df.rename(columns={'date': 'time'})
        elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'time'})
        
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        # IST FIX (+5h 30m)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['time'] = df['time'] + pd.Timedelta(hours=5, minutes=30)
        df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600) 
def get_usd_inr_rate():
    try:
        rate_df = yf.download("USDINR=X", period="1d", progress=False)
        if not rate_df.empty:
            if isinstance(rate_df.columns, pd.MultiIndex):
                return rate_df['Close'].iloc[-1].iloc[0]
            return rate_df['Close'].iloc[-1]
        return 86.0 
    except:
        return 86.0

# --- 2. SIGNALS & INDICATORS ---
def calculate_chart_signals(df, ema_len):
    if df.empty: return df
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA'] = df['close'].ewm(span=ema_len, adjust=False).mean()

    df['Signal'] = 'HOLD'
    buy_condition = (df['RSI'] < 35) & (df['close'] > df['open'])
    sell_condition = (df['RSI'] > 75) & (df['close'] < df['open'])
    df.loc[buy_condition, 'Signal'] = 'BUY'
    df.loc[sell_condition, 'Signal'] = 'SELL'
    
    return df

# --- 3. ML FORECASTING (FIXED INFINITY ERROR) ---
def calculate_ridge_features(df):
    if df.empty: return df, None
    df = df.copy()
    
    # 1. Calculate VROC (The source of the error)
    # pct_change() can create 'inf' if volume is 0
    df['VROC'] = df['volume'].pct_change(periods=14) * 100
    
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    
    # 2. CRITICAL FIX: Replace Infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. Drop NaNs (Cleans both missing values AND the infinite ones we just swapped)
    df.dropna(inplace=True)
    
    feature_cols = ['close', 'volume', 'VROC', 'SMA_20', 'STD_20']
    return df, feature_cols

def run_ridge_forecast(df, feature_cols):
    X = df[feature_cols].values
    y = df['close'].shift(-1).dropna().values
    X_train = X[:-1]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y)
    
    last_row = df.iloc[-1][feature_cols].values.reshape(1, -1)
    predictions = []
    curr_feats = last_row.copy()
    
    for i in range(3):
        pred = model.predict(curr_feats)[0]
        predictions.append(pred)
        curr_feats[0, 0] = pred 
        
    return predictions

# --- 4. SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.radio("Select Mode", ["📉 Live Technical Chart", "🧠 AI Ridge Forecast", "📡 Market Scanner"])
st.sidebar.markdown("---")
ticker = st.sidebar.text_input("Ticker", value="BTC-USD").upper()

# INR CONVERSION
show_inr = st.sidebar.checkbox("💱 Convert to INR (₹)", value=False)
currency_symbol = "$"
conversion_rate = 1.0

if show_inr and not ticker.endswith(".NS") and not ticker.endswith(".BO"):
    conversion_rate = get_usd_inr_rate()
    currency_symbol = "₹"
elif ticker.endswith(".NS") or ticker.endswith(".BO"):
    currency_symbol = "₹" 

# --- 5. MODE 1: LIVE CHART ---
if mode == "📉 Live Technical Chart":
    st.sidebar.subheader("⚙️ Chart Settings")
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=0)
    ema_len = st.sidebar.slider("EMA Length", 10, 200, 50)
    
    st.subheader(f"📉 {ticker} Professional Live Chart")
    
    df = get_data(ticker, interval=interval, period="60d")
    
    if df.empty:
        st.error("❌ Data Download Failed.")
    else:
        if conversion_rate != 1.0:
            cols_to_convert = ['open', 'high', 'low', 'close']
            df[cols_to_convert] = df[cols_to_convert] * conversion_rate
            
        st.success(f"✅ Data Loaded. Currency: {currency_symbol}")
        
        df = calculate_chart_signals(df, ema_len)
        last = df.iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Price", f"{currency_symbol}{last['close']:,.2f}")
        c2.metric("RSI", f"{last['RSI']:.1f}")
        color = "gray"
        if last['Signal'] == 'BUY': color = "green"
        elif last['Signal'] == 'SELL': color = "red"
        c3.markdown(f"**Signal:** :{color}[{last['Signal']}]")
        
        try:
            chart = StreamlitChart(width=1100, height=600)
            chart.set(df[['time', 'open', 'high', 'low', 'close']])
            
            line = chart.create_line(name='EMA', color='orange', width=2)
            line.set(df[['time', 'EMA']].dropna())
            
            signals = df[df['Signal'].isin(['BUY', 'SELL'])].dropna()
            for i, row in signals.iterrows():
                if row['Signal'] == 'BUY':
                    chart.marker(time=row['time'], position='below', shape='arrow_up', color='green', text='BUY')
                elif row['Signal'] == 'SELL':
                    chart.marker(time=row['time'], position='above', shape='arrow_down', color='red', text='SELL')
            
            chart.load()
            
            st.write("🟢 **Live Loop Active**")
            
            while True:
                time.sleep(3)
                tick_df = get_data(ticker, interval=interval, period="1d")
                if not tick_df.empty:
                    latest = tick_df.iloc[-1]
                    if conversion_rate != 1.0:
                        latest['open'] *= conversion_rate
                        latest['high'] *= conversion_rate
                        latest['low'] *= conversion_rate
                        latest['close'] *= conversion_rate
                    chart.update(latest)
                    
        except Exception as e:
            st.error(f"Chart Error: {e}")

# --- 6. MODE 2: RIDGE FORECAST ---
elif mode == "🧠 AI Ridge Forecast":
    st.subheader(f"🧠 {ticker} Recursive Ridge Regression")
    
    df = get_data(ticker, interval="1h", period="1y")
    
    if not df.empty and len(df) > 50:
        if conversion_rate != 1.0:
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] * conversion_rate
            
        df_ml, feats = calculate_ridge_features(df)
        
        # FINAL SAFETY CHECK
        if df_ml is not None and not df_ml.empty:
            forecasts = run_ridge_forecast(df_ml, feats)
            curr_price = df_ml.iloc[-1]['close']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{currency_symbol}{curr_price:,.2f}")
            c2.metric("T+1 Pred", f"{currency_symbol}{forecasts[0]:,.2f}")
            c3.metric("T+2 Pred", f"{currency_symbol}{forecasts[1]:,.2f}")
            c4.metric("T+3 Pred", f"{currency_symbol}{forecasts[2]:,.2f}")
            
            final_pred = forecasts[2]
            if final_pred > curr_price:
                st.success(f"🚀 **RIDGE: BUY SIGNAL**")
            else:
                st.error(f"🔻 **RIDGE: SELL SIGNAL**")
                
            st.dataframe(df_ml.tail(5))
        else:
            st.error("Error: Data contained too many Infinity values (e.g. 0 volume). Try a different ticker.")
    else:
        st.warning("Not enough data.")

# --- 7. MODE 3: SCANNER ---
elif mode == "📡 Market Scanner":
    st.subheader("📡 Live Market Scanner (USD Only)")
    watchlist = ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA", "NVDA"]
    
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(watchlist):
            d = get_data(t, interval="15m", period="5d")
            if not d.empty:
                d = calculate_chart_signals(d, 50)
                last = d.iloc[-1]
                act = "WAIT"
                if last['Signal'] == 'BUY': act = "🟢 BUY"
                elif last['Signal'] == 'SELL': act = "🔴 SELL"
                results.append({"Ticker": t, "Price": f"${last['close']:.2f}", "Action": act})
            bar.progress((i+1)/len(watchlist))
        st.dataframe(pd.DataFrame(results))