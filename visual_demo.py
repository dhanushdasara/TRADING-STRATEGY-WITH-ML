import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mplfinance as mpf
import matplotlib.animation as animation
import random

# --- CONFIGURATION ---
TICKER = 'BTC-USD'
INTERVAL = '1m'       # 1-minute candles
REFRESH_MS = 2000     # Update every 2 seconds (Fast for Demo)
USE_MOCK_ON_FAIL = True # If Real data fails, use Fake data (Crucial for Demos)

# --- GLOBAL STATE ---
class BotState:
    def __init__(self):
        self.model = None
        self.df = pd.DataFrame()
        self.last_signal = "HOLD"
        self.buy_signals = []
        self.sell_signals = []

bot = BotState()

# --- 1. DATA ENGINE (With Fail-Safe) ---
def get_data():
    """Attempts to fetch real data; generates fake data if that fails."""
    try:
        # Try Real Data
        df = yf.download(TICKER, period='1d', interval=INTERVAL, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty and USE_MOCK_ON_FAIL:
            raise ValueError("Empty Data")
        return df
    
    except Exception as e:
        # Fallback: Generate Mock Data for the Demo
        print(f"⚠️ Market Data Warning: {e}. Switching to SIMULATION MODE.")
        return generate_mock_data()

def generate_mock_data():
    """Generates realistic-looking random Bitcoin candles."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='1min')
    base_price = 65000
    close = base_price + np.cumsum(np.random.randn(60) * 50)
    high = close + np.random.rand(60) * 30
    low = close - np.random.rand(60) * 30
    open_p = close - np.random.randn(60) * 20
    
    df = pd.DataFrame({
        'Open': open_p, 'High': high, 'Low': low, 'Close': close, 'Volume': np.random.randint(100, 1000, 60)
    }, index=dates)
    df.index.name = 'Date'
    return df

# --- 2. ML LOGIC ---
def process_data(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(5).mean() # Short window for demo speed
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(5).mean()))
    df.fillna(0, inplace=True)
    return df

def train_model():
    print("Training Model...", end="")
    df = get_data()
    df = process_data(df)
    
    # Fake Target for demo training
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(df[['RSI', 'SMA_50']], df['Target'])
    bot.model = model
    print(" Done.")

# --- 3. ANIMATION LOOP ---
def animate(ival):
    # 1. Fetch
    df = get_data()
    df = process_data(df)
    
    # 2. Predict
    latest = df.iloc[-1:][['RSI', 'SMA_50']]
    prediction = bot.model.predict(latest.values)[0]
    
    # 3. Generate Signals for Plot
    buys = np.full(len(df), np.nan)
    sells = np.full(len(df), np.nan)
    
    # Simple logic: If AI predicts UP, show Green Arrow
    last_idx = df.index[-1]
    loc_idx = df.index.get_loc(last_idx)
    
    if prediction == 1:
        buys[loc_idx] = df['Low'].iloc[-1] * 0.999
        print(f"[{pd.Timestamp.now().time()}] AI Signal: BUY 🟢")
    else:
        sells[loc_idx] = df['High'].iloc[-1] * 1.001
        print(f"[{pd.Timestamp.now().time()}] AI Signal: SELL 🔴")

    # 4. Update Chart
    ax1.clear()
    ax2.clear()
    
    apds = [
        mpf.make_addplot(buys, type='scatter', markersize=200, marker='^', color='g', ax=ax1),
        mpf.make_addplot(sells, type='scatter', markersize=200, marker='v', color='r', ax=ax1)
    ]
    
    mpf.plot(df, type='candle', ax=ax1, volume=ax2, addplot=apds, style='yahoo', 
             axtitle=f'LIVE AI TRADING BOT: {TICKER}')

# --- MAIN ---
if __name__ == "__main__":
    train_model()
    
    # Setup Figure
    fig = mpf.figure(style='yahoo', figsize=(12, 8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    
    ani = animation.FuncAnimation(fig, animate, interval=REFRESH_MS)
    mpf.show()