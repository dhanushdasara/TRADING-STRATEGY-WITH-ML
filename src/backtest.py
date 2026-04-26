import matplotlib.pyplot as plt
import pandas as pd

def run_backtest(df, predictions, test_index):
    """
    Simulates the strategy returns vs. Buy & Hold returns.
    """
    # Slice the original dataframe to match the test set
    test_df = df.loc[test_index].copy()
    test_df['Predicted_Signal'] = predictions
    
    # Calculate Returns
    # Market Return: Daily % change of Close price
    test_df['Market_Return'] = test_df['Close'].pct_change()
    
    # Strategy Return: Market Return * Signal (1 or 0)
    # Shift signal by 1 because we trade 'tomorrow' based on 'today's' prediction
    test_df['Strategy_Return'] = test_df['Market_Return'] * test_df['Predicted_Signal'].shift(1)
    
    # Cumulative Returns
    test_df['Cumulative_Market'] = (1 + test_df['Market_Return']).cumprod()
    test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Cumulative_Market'], label='Buy & Hold (Market)', color='gray')
    plt.plot(test_df['Cumulative_Strategy'], label='ML Strategy (Candlesticks)', color='green')
    plt.title('Backtest: ML Candlestick Strategy vs Market')
    plt.legend()
    plt.savefig('my_result.png')
    print("Graph saved as 'my_result.png' in your folder.")
    
    return test_df