import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical data for a given ticker.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError("No data found. Check your internet or ticker symbol.")
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv(f'data/{ticker}.csv')
    return df

def load_data(ticker):
    """
    Loads data from local CSV if available.
    """
    path = f'data/{ticker}.csv'
    if os.path.exists(path):
        # Read and handle multi-level columns if necessary
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Fix yfinance multi-index column issue if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    else:
        return None