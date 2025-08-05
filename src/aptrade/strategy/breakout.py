import pandas as pd
from backtesting import Backtest, Strategy
import datetime as dt

class BreakoutStrategy(Strategy):
    open_range_minutes = 10
    premarket_end = dt.time(9, 30)
    exit_minute_bar = dt.time(15, 59)
    risk_percent = 0.01
    stop_loss_pct = 0.30
    profit_target_pct = 0.30
    pull_back_threshold = 0.20  # 20% pullback
    high_move_threshold = 1.00  # 100% move
    reclaim_pct = 0.90  # 90% reclaim

    def init(self):
        self.current_day = None
        self.prev_close = None
        self.premarket_high = None
        self.premarket_high_time = None
        self.premarket_qualified = False
        self.pullback_detected = False
        self.pullback_low = None
        self.reclaim_detected = False
        self.entry_price = None
        self.traded_today = False

    def _reset_day(self, day, prev_close):
        self.current_day = day
        self.prev_close = prev_close
        self.premarket_high = None
        self.premarket_high_time = None
        self.premarket_qualified = False
        self.pullback_detected = False
        self.pullback_low = None
        self.reclaim_detected = False
        self.entry_price = None
        self.traded_today = False

    def next(self):
        t = self.data.index[-1]
        current_bar_date = t.date()
        current_time = t.time()

        # Detect new day and get previous close
        if self.current_day != current_bar_date:
            # Find previous day's last close
            prev_day_mask = self.data.index.date < current_bar_date
            if prev_day_mask.any():
                self.prev_close = self.data.Close[prev_day_mask][-1]
            else:
                self.prev_close = None
            self._reset_day(current_bar_date, self.prev_close)
            print(f"New day: {current_bar_date}, prev close: {self.prev_close}")

        # 1. Track pre-market high and check for >100% move
        if current_time < self.premarket_end and self.prev_close:
            if self.premarket_high is None or self.data.High[-1] > self.premarket_high:
                self.premarket_high = self.data.High[-1]
                self.premarket_high_time = t
            move = (self.premarket_high - self.prev_close) / self.prev_close
            if move >= self.high_move_threshold:
                self.premarket_qualified = True
                print(f"{current_time} | {self.data.High[-1]} : Premarkety high {self.premarket_high:.2f} is {move*100:.2f}% above prev close")

        # 2. After 100% move, look for 20% pullback from high
        if self.premarket_qualified and not self.pullback_detected:
            # Only consider prices after premarket high
            if self.premarket_high_time and t > self.premarket_high_time:
                pullback = (self.premarket_high - self.data.Low[-1]) / self.premarket_high
                if pullback >= self.pull_back_threshold:
                    self.pullback_detected = True
                    self.pullback_low = self.data.Low[-1]
                    print(f"Pullback detected: {pullback*100:.2f}% from high {self.premarket_high:.2f} to low {self.pullback_low:.2f}")

        # 3. Wait for reclaim to 90% of high (entry trigger)
        if self.pullback_detected and not self.reclaim_detected:
            reclaim_price = self.premarket_high * self.reclaim_pct
            if self.data.Close[-1] >= reclaim_price:
                self.reclaim_detected = True
                self.entry_price = reclaim_price
                print(f"Reclaim detected: price {self.data.Close[-1]:.2f} >= {reclaim_price:.2f}")

        # 4. Enter short if not already in position and not traded today
        if self.reclaim_detected and not self.position and not self.traded_today:
            stop_loss = self.entry_price * (1 + self.stop_loss_pct)
            take_profit = self.entry_price * (1 - self.profit_target_pct)
            per_share_risk = abs(self.entry_price - stop_loss)
            position_size = int((self.risk_percent * self.equity) / per_share_risk)
            print(f"Short entry at {self.entry_price:.2f}, stop {stop_loss:.2f}, tp {take_profit:.2f}, size {position_size}")
            self.sell(size=position_size, tp=take_profit, sl=stop_loss)
            self.traded_today = True

        # 5. Close at end of day
        if self.position and current_time == self.exit_minute_bar:
            print("Closing position at end of day")
            self.position.close()