import time
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import sys

# --- CONFIGURATION ---
TICKER = 'BTC-USD'
# SET THIS TO 10 SECONDS FOR YOUR DEMO SO THE COORDINATOR SEES ACTION FAST!
REFRESH_RATE = 10  

class PaperTradingBot:
    def __init__(self, ticker, initial_balance=10000):
        self.ticker = ticker
        self.balance = initial_balance  # Fake Money
        self.shares = 0
        self.portfolio_value = initial_balance
        self.model = None
        self.trades_log = [] # Keep track of buy/sells to show history
        
    def fetch_data(self):
        """Get live market data"""
        # Download enough data to calculate indicators
        try:
            df = yf.download(self.ticker, period='1mo', interval='5m', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            return None

    def calculate_features(self, df):
        """Feature Engineering (The 'Brain' of the bot)"""
        df = df.copy()
        
        # 1. Technical Indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. Candlestick Pattern: Engulfing
        op, hi, lo, cl = df['Open'], df['High'], df['Low'], df['Close']
        prev_op, prev_cl = op.shift(1), cl.shift(1)
        
        df['CDL_ENGULFING'] = np.where(
            (prev_cl < prev_op) & (cl > op) & (op < prev_cl) & (cl > prev_op), 
            1, 0
        )
        
        df.dropna(inplace=True)
        return df

    def train(self):
        """Train the model on the spot"""
        print("Training model on latest market data...", end="")
        df = self.fetch_data()
        df = self.calculate_features(df)
        
        # Create Target: 1 if price went UP in the next period
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        features = ['RSI', 'SMA_50', 'SMA_200', 'CDL_ENGULFING']
        X = df[features]
        y = df['Target']
        
        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        self.model.fit(X, y)
        print(" Done!")

    def run_live(self):
        """The Main Loop"""
        print(f"\n--- 🔴 LIVE TRADING BOT STARTED: {self.ticker} ---")
        print(f"Initial Paper Balance: ${self.balance:,.2f}")
        print("-" * 50)
        
        while True:
            try:
                # 1. Get Real-Time Data
                df = self.fetch_data()
                if df is None:
                    continue
                    
                df = self.calculate_features(df)
                current_data = df.iloc[-1]
                current_price = current_data['Close']
                timestamp = df.index[-1]
                
                # 2. Ask AI for Prediction
                features = ['RSI', 'SMA_50', 'SMA_200', 'CDL_ENGULFING']
                prediction = self.model.predict(df[features].iloc[-1:].values)[0]
                
                # 3. Simulate Trading Logic
                signal = "HOLD"
                
                # BUY LOGIC: AI says UP and we have money
                if prediction == 1 and self.balance > current_price:
                    self.shares = self.balance / current_price # Buy Max
                    self.balance = 0
                    signal = "BUY  🟢"
                    self.trades_log.append(f"Bought at ${current_price:.2f}")

                # SELL LOGIC: AI says DOWN and we have shares
                elif prediction == 0 and self.shares > 0:
                    self.balance = self.shares * current_price # Sell All
                    self.shares = 0
                    signal = "SELL 🔴"
                    self.trades_log.append(f"Sold at ${current_price:.2f}")
                
                # Update Portfolio Value
                if self.shares > 0:
                    self.portfolio_value = self.shares * current_price
                else:
                    self.portfolio_value = self.balance

                # 4. Display Dashboard
                sys.stdout.write(f"\r[{datetime.now().strftime('%H:%M:%S')}] Price: ${current_price:,.2f} | AI Signal: {signal} | Portfolio: ${self.portfolio_value:,.2f} | RSI: {current_data['RSI']:.1f}   ")
                sys.stdout.flush()
                
                time.sleep(REFRESH_RATE)
                
            except KeyboardInterrupt:
                print("\n\n--- 🏁 TRADING SESSION ENDED ---")
                print(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
                print("Trade History:", self.trades_log)
                break

if __name__ == "__main__":
    bot = PaperTradingBot(TICKER)
    bot.train()
    bot.run_live()