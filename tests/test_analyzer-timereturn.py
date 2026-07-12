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

import time

try:
    time_clock = time.process_time
except:
    time_clock = time.clock

import aptrade as bt
import aptrade.indicators as btind
import testcommon


class CurrentTestStrategy(bt.Strategy):
    params = (
        ("period", 15),
        ("printdata", True),
        ("printops", True),
        ("stocklike", True),
    )

    def log(self, txt, dt=None, nodate=False):
        if not nodate:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print(f"{dt.isoformat()}, {txt}")
        else:
            print(f"---------- {txt}")

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if isinstance(order, bt.BuyOrder):
                if self.p.printops:
                    txt = f"BUY, {order.executed.price:.2f}"
                    self.log(txt, order.executed.dt)
                chkprice = f"{order.executed.price:.2f}"
                self.buyexec.append(chkprice)
            else:  # elif isinstance(order, SellOrder):
                if self.p.printops:
                    txt = f"SELL, {order.executed.price:.2f}"
                    self.log(txt, order.executed.dt)

                chkprice = f"{order.executed.price:.2f}"
                self.sellexec.append(chkprice)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            if self.p.printops:
                self.log(f"{order.Status[order.status]} ,")

        # Allow new orders
        self.orderid = None

    def __init__(self):
        # Flag to allow new orders in the system or not
        self.orderid = None

        self.sma = btind.SMA(self.data, period=self.p.period)
        self.cross = btind.CrossOver(self.data.close, self.sma, plot=True)

    def start(self):
        if not self.p.stocklike:
            self.broker.setcommission(commission=2.0, mult=10.0, margin=1000.0)

        if self.p.printdata:
            self.log("-------------------------", nodate=True)
            self.log(
                f"Starting portfolio value: {self.broker.getvalue():.2f}", nodate=True
            )

        self.tstart = time_clock()

        self.buycreate = []
        self.sellcreate = []
        self.buyexec = []
        self.sellexec = []

    def stop(self):
        tused = time_clock() - self.tstart
        if self.p.printdata:
            self.log(f"Time used: {str(tused)}")
            self.log(f"Final portfolio value: {self.broker.getvalue():.2f}")
            self.log(f"Final cash value: {self.broker.getcash():.2f}")
            self.log("-------------------------")
        else:
            pass

    def next(self):
        if self.p.printdata:
            self.log(
                f"Open, High, Low, Close, {self.data.open[0]:.2f}, {self.data.high[0]:.2f}, {self.data.low[0]:.2f}, {self.data.close[0]:.2f}, Sma, {self.sma[0]:f}"
            )
            self.log(f"Close {self.data.close[0]:.2f} - Sma {self.sma[0]:.2f}")

        if self.orderid:
            # if an order is active, no new orders are allowed
            return

        if not self.position.size:
            if self.cross > 0.0:
                if self.p.printops:
                    self.log(f"BUY CREATE , {self.data.close[0]:.2f}")

                self.orderid = self.buy()
                chkprice = f"{self.data.close[0]:.2f}"
                self.buycreate.append(chkprice)

        elif self.cross < 0.0:
            if self.p.printops:
                self.log(f"SELL CREATE , {self.data.close[0]:.2f}")

            self.orderid = self.close()
            chkprice = f"{self.data.close[0]:.2f}"
            self.sellcreate.append(chkprice)


chkdatas = 1


def test_run(main=False):
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    cerebros = testcommon.runtest(
        datas,
        CurrentTestStrategy,
        printdata=main,
        stocklike=False,
        printops=main,
        plot=main,
        analyzer=(bt.analyzers.TimeReturn, {"timeframe": bt.TimeFrame.Years}),
    )

    for cerebro in cerebros:
        strat = cerebro.runstrats[0][0]  # no optimization, only 1
        analyzer = strat.analyzers[0]  # only 1
        analysis = analyzer.get_analysis()
        if main:
            print(analysis)
            print(str(analysis[next(iter(analysis.keys()))]))
        else:
            # Handle different precision
            sval = "0.2794999999999983"

            assert str(analysis[next(iter(analysis.keys()))]) == sval


if __name__ == "__main__":
    test_run(main=True)
