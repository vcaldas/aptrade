#!/usr/bin/env python
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import os
import os.path
from pathlib import Path

import aptrade as bt
import aptrade.indicators as btind


class MyStrategy(bt.Strategy):
    params = (("smaperiod", 15),)

    def log(self, txt, dt=None):
        """Logging function fot this strategy"""
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        # SimpleMovingAverage on main data
        # Equivalent to -> sma = btind.SMA(self.data, period=self.p.smaperiod)
        sma = btind.SMA(period=self.p.smaperiod)

        # CrossOver (1: up, -1: down) close / sma
        self.buysell = btind.CrossOver(self.data.close, sma, plot=True)

        # Sentinel to None: new ordersa allowed
        self.order = None

    def next(self):
        # Access -1, because drawdown[0] will be calculated after "next"
        self.log(f"DrawDown: {self.stats.drawdown.drawdown[-1]:.2f}")
        self.log(f"MaxDrawDown: {self.stats.drawdown.maxdrawdown[-1]:.2f}")

        # Check if we are in the market
        if self.position:
            if self.buysell < 0:
                self.log(f"SELL CREATE, {self.data.close[0]:.2f}")
                self.sell()

        elif self.buysell > 0:
            self.log(f"BUY CREATE, {self.data.close[0]:.2f}")
            self.buy()


def runstrat():
    cerebro = bt.Cerebro()

    datapath = Path(__file__).resolve().parents[2] / "datas" / "2006-day-001.txt"
    data = bt.feeds.BacktraderCSVData(dataname=str(datapath))
    cerebro.adddata(data)

    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.DrawDown_Old)

    cerebro.addstrategy(MyStrategy)
    cerebro.run()

    if os.environ.get("APTRADE_SAMPLE_SKIP_PLOT") != "1":
        cerebro.plot()


if __name__ == "__main__":
    runstrat()
