from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import backtrader as bt
import datetime
from strategy import TestStrategy

if __name__ == "__main__":
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    # strats = cerebro.optstrategy(
    #     TestStrategy,
    #     maperiod=range(10, 31), printlog=True)
    
    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, "./orcl-1995-2014.txt")

    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2001, 1, 2),
        reverse=False,
    )
    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)