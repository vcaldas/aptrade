import os
import pandas as pd

def load_data(filename):
    """    Load data from a CSV file and prepare it for backtesting.
    Args:
        filename (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Prepared DataFrame with the necessary columns and datetime index.
    """
    df = pd.read_csv(filename)        
    # Ensure 't' is present and convert to datetime
    if 'window_start' in df.columns:
        df['Date'] = pd.to_datetime(df['window_start'], unit='ns')
        df.set_index('Date', inplace=True)
    
    # print(df.index.tz)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')

    # # Select and rename columns for Backtesting library
    bt_cols = ["ticker", "open", "high", "low", "close", "volume"]
    df = df[bt_cols]
    df.columns = ["ticker", "Open", "High", "Low", "Close", "Volume"]
    return df