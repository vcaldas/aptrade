
import pandas as pd
from backtesting import Backtest, Strategy
from polygon import RESTClient

from aptrade.strategy import BreakoutStrategy

client = RESTClient("POLYGON_API_KEY") # POLYGON_API_KEY is used

def get_symbol(symbol, start = "2025-06-30", end = "2025-07-02"):
    
    aggs = client.get_aggs(
        symbol,
        1,
        "minute",

        start,
        end,
    )
    df = pd.DataFrame(aggs)
    df.head()

    # Ensure 't' is present and convert to datetime
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')

    # Select and rename columns for Backtesting library
    bt_cols = ["open", "high", "low", "close", "volume"]
    df = df[bt_cols]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


df = get_symbol("TRNR", start="2025-06-30", end="2025-07-01")
def per_share_commission(size, price):
    return abs(size) * 0.0005

myargs = {
    'symbol': 'TRNR',
   
}
# # 3. create a backtest, pass in data and the strategy you want to run on the data
bt = Backtest(df, BreakoutStrategy, cash=25_000, commission=per_share_commission, margin=0.25)
# bt._strategy.symbol = "TRNR" 
stats = bt.run(symbol="TRNR")
# bt.plot()
print(bt._strategy.info)
# print(stats['_trades'])