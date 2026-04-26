import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) manually.
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_candlestick_patterns(df):
    """
    Adds Candlestick patterns and Technical Indicators manually
    without using external libraries like pandas-ta.
    """
    df = df.copy()
    
    # 1. Add Target Variable: 1 if tomorrow's Close > today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # --- Manual Indicator Calculations ---
    
    # Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # --- Manual Candlestick Pattern Detection ---
    
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    
    # Pattern 1: Doji
    # Definition: Open and Close are virtually equal (body is < 10% of total range)
    body = (cl - op).abs()
    rng = hi - lo
    df['CDL_DOJI'] = np.where(body <= (rng * 0.1), 100, 0)
    
    # Pattern 2: Bullish Engulfing
    # Definition: Yesterday Red, Today Green, Today engulfs yesterday
    prev_op = op.shift(1)
    prev_cl = cl.shift(1)
    
    # Logic: 
    # 1. Yesterday was Red (prev_cl < prev_op)
    # 2. Today is Green (cl > op)
    # 3. Today Opens lower than Yesterday closed (op < prev_cl)
    # 4. Today Closes higher than Yesterday opened (cl > prev_op)
    df['CDL_ENGULFING'] = np.where(
        (prev_cl < prev_op) & 
        (cl > op) & 
        (op < prev_cl) & 
        (cl > prev_op), 
        100, 0
    )
    
    # Pattern 3: Hammer
    # Definition: Small body near top, long lower shadow
    # Lower shadow is at least 2x the body
    lower_shadow = np.minimum(op, cl) - lo
    df['CDL_HAMMER'] = np.where(
        (lower_shadow >= (body * 2)) & (rng > 0),
        100, 0
    )

    # Clean Data (remove rows with NaN from rolling averages)
    df.dropna(inplace=True)
    
    return df