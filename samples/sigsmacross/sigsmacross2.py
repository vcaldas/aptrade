#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
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
import argparse
from datetime import datetime

import aptrade as bt


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)


def runstrat(pargs=None):
    args = parse_args(pargs)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    data0 = bt.feeds.YahooFinanceData(
        dataname=args.data,
        fromdate=datetime.strptime(args.fromdate, "%Y-%m-%d"),
        todate=datetime.strptime(args.todate, "%Y-%m-%d"),
    )

    cerebro.adddata(data0)

    cerebro.run()
    if not args.noplot:
        cerebro.plot()


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="sigsmacross2",
    )

    parser.add_argument(
        "--data",
        required=False,
        default="../../datas/yhoo-1996-2015.txt",
        help="Yahoo CSV data path",
    )

    parser.add_argument(
        "--fromdate",
        required=False,
        default="2011-01-01",
        help="Starting date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--todate",
        required=False,
        default="2012-12-31",
        help="Ending date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--noplot",
        required=False,
        action="store_true",
        help="Skip plotting so the sample can run without optional chart dependencies",
    )

    return parser.parse_args(pargs)


if __name__ == "__main__":
    runstrat()
