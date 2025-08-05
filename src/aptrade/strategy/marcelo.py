import pandas as pd
from backtesting import Backtest, Strategy
import datetime as dt

class MarceloStrategy(Strategy):
    open_range_minutes = 10
    last_minute_bar_in_opening_range = dt.time(9, 20 + open_range_minutes)
    exit_minute_bar = dt.time(15, 59)
    risk_percent = 0.01
    take_profit_multiple = 10
    pull_back_threshold = 20  # pullback threshold in percentage
    daily_high_threshold = 100  # daily high threshold in percentage

    def init(self):
        self.current_day        = None   # tracks the current day (YYYY-MM-DD)
        self.current_day_open = None
        self.opening_range_high = None
        self.opening_range_low = None
        self.pre_market_close = None
        self.pull_back = None  # tracks the pullback percentage
        self.hit_percentage = False
        self.hit_pull_back = False
        self.traded_today = False  # tracks if we have traded today

    def _reset_range(self, day, open, pre_market_close=None, pull_back=None):
        """Reset the opening range at the start of a new day."""
        self.opening_range_high = None
        self.opening_range_low = None
        self.current_day = day
        self.current_day_open = open
        self.pre_market_close = None
        self.pull_back = None  # tracks the pullback percentage
        self.hit_percentage = False
        self.hit_pull_back = False
        self.traded_today = False  # reset traded status for the new day

    def _get_position_size(self, entry_price: float, stop_price: float) -> int:
        per_share_risk = abs(entry_price - stop_price)  #  |P − S|  =  R

        if per_share_risk == 0:
            return 0

        # Risk-based cap: position that loses 1 % of equity at the stop
        shares_by_risk = (self.risk_percent * self.equity) / per_share_risk

        # Leverage-based cap: shares affordable with 4× buying power
        shares_by_leverage = (self.max_leverage * self.equity) / entry_price

        # Final size: smaller of the two, floored to an int
        return int(min(shares_by_risk, shares_by_leverage))

    def next(self):
        # Implement the strategy logic here
        t = self.data.index[-1]  # assuming data is indexed by datetime
        current_bar_date = t.date()

        # Detect a new day
        if current_bar_date != self.current_day:
            # Find the last bar before 9:30 (pre-market close)
            pre_market_mask = (self.data.index.date == current_bar_date) & (self.data.index.time < dt.time(9, 30))
            if pre_market_mask.any():
                self.pre_market_close = self.data.Close[pre_market_mask][-1]
            else:
                self.pre_market_close = None
            self._reset_range(current_bar_date, self.data.Open[-1], pre_market_close=self.pre_market_close)
            print(f"New day detected: {current_bar_date}")
            if self.pre_market_close is not None:
                open_price = self.data.Open[-1]
                pct_change = 100 * (open_price - self.pre_market_close) / self.pre_market_close
                print(f"% Change from pre-market close to open: {pct_change:.2f}%")

        # calculate the opening range
        if t.time() < self.last_minute_bar_in_opening_range:
            if self.opening_range_high is not None:
                self.opening_range_high = max(self.opening_range_high, self.data.High[-1])
            else:
                self.opening_range_high = self.data.High[-1]

            if self.opening_range_low is not None:
                self.opening_range_low = min(self.opening_range_low, self.data.Low[-1])
            else:
                self.opening_range_low = self.data.Low[-1]

            if t.time() < self.last_minute_bar_in_opening_range:
                return  # don’t trade yet_in_opening_range:

    # Right when the opening range closes, log the opening range high and low for debugging
        if t.time() == self.last_minute_bar_in_opening_range:
          print(f"opening range high is {self.opening_range_high}")
          print(f"opening range low is {self.opening_range_low}")
          pct = 100 * (self.opening_range_high - self.opening_range_low) / self.opening_range_low
          print(f"opening range percentage is {pct:.2f}%")
        # Check if the opening range is set
          if pct > self.daily_high_threshold:
            self.hit_percentage = True
            print(f"Opening range percentage hit threshold: {pct:.2f}%")
        
        # Market is open, check if we are in the opening range
        if t.time() >= self.last_minute_bar_in_opening_range:
            draw = 100 * (self.opening_range_high - self.data.Close[-1]) / self.opening_range_low
            self.pull_back = draw
            if draw > self.pull_back_threshold:
                # print(f"Pullback detected: {draw:.2f}%")
                self.hit_pull_back = True
                
        
        
        if not self.position and not self.traded_today:
            range_size = self.opening_range_high - self.opening_range_low
            planned_entry_price = self.data.Close[-1]
            
            if self.hit_percentage and self.hit_pull_back:
                print("Both opening range percentage and pullback thresholds hit, proceeding with trade logic.")
            
                stop_loss_price = self.opening_range_high
                per_share_risk = abs(planned_entry_price - stop_loss_price)
                position_size = int((self.risk_percent * self.equity) / per_share_risk)
                take_profit_price = planned_entry_price*0.70
                print("take_profit_price", take_profit_price)
                print(f"going short, position size {position_size} shares at planned price {planned_entry_price}, stop loss {stop_loss_price}")
                order = self.sell(size=position_size, tp=take_profit_price, sl=stop_loss_price)
                self.traded_today = True  # mark as traded today
            else:
                pass
             
        current_bar_date_time = f"{current_bar_date} {t.time()}"

        if self.position and t.time() == self.exit_minute_bar:
          print("closing out position")
          self.position.close()
