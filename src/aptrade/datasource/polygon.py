import os
from polygon import RESTClient
from dotenv import load_dotenv
import pandas as pd

# Load API key from environment variable or .env file
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

client = RESTClient(API_KEY)

exchanges = ["AMEX", "NYSE", "NASDAQ"]
all_tickers = []

for exch in exchanges:
    print(f"Fetching tickers for {exch}...")
    for t in client.list_tickers(market="stocks", exchange=exch, active=True):
        all_tickers.append({
            "ticker": t.ticker,
            "name": t.name,
            "exchange": exch
        })

# Convert to DataFrame for convenience
assets_df = pd.DataFrame(all_tickers)
print(assets_df.head())
print(f"Total assets found: {len(assets_df)}")