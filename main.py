from src.data_loader import fetch_data
from src.features import add_candlestick_patterns
from src.model import train_model
from src.backtest import run_backtest

# --- Configuration ---
TICKER = 'BTC-USD' # You can change this to 'AAPL', 'EURUSD=X', etc.
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

def main():
    # 1. Get Data
    df = fetch_data(TICKER, START_DATE, END_DATE)
    
    # 2. Feature Engineering (Detect Patterns)
    print("Detecting Candlestick Patterns...")
    df_processed = add_candlestick_patterns(df)
    print(f"Data processed. Shape: {df_processed.shape}")
    
    # 3. Train Model
    print("Training Random Forest Model...")
    model, X_test, y_test, predictions = train_model(df_processed)
    
    # 4. Backtest
    print("Running Backtest...")
    run_backtest(df, predictions, X_test.index)

if __name__ == "__main__":
    main()