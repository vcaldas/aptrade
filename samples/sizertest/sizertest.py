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

import argparse
import datetime
from dataclasses import dataclass

import aptrade as bt
from aptrade.sizers import AbstractSizer


class CloseSMA(bt.Strategy):
    params = (("period", 15),)

    def __init__(self):
        sma = bt.indicators.SMA(self.data, period=self.p.period)
        self.crossover = bt.indicators.CrossOver(self.data, sma)

    def next(self):
        if self.crossover > 0:
            self.buy()

        elif self.crossover < 0:
            self.sell()


class LongOnly(bt.sizers.AbstractSizer):
    params = (("stake", 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.p.stake

        # Sell situation
        position = self.broker.getposition(data)
        if not position.size:
            return 0  # do not sell if nothing is open
        print(f"LongOnly: position.size={position.size}, size={self.p.stake}")
        return self.p.stake


@dataclass(slots=True, frozen=True)
class FixedReverserParameters:
    stake: int = 1


class FixedReverser(AbstractSizer):
    Parameters = FixedReverserParameters

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.strategy.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        print(f"FixedReverser: position.size={position.size}, size={size}")
        return size


# class FixedReverser(bt.sizers.AbstractSizer):
#     params = (("stake", 1),)

#     def _getsizing(self, comminfo, cash, data, isbuy):
#         position = self.strategy.getposition(data)
#         size = self.p.stake * (1 + (position.size != 0))
#         print(f"FixedReverser: position.size={position.size}, size={size}")
#         return size


def runstrat(args=None):
    args = parse_args(args)
    print(args)
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(args.cash)

    dkwargs = {}
    if args.fromdate:
        fromdate = datetime.datetime.strptime(args.fromdate, "%Y-%m-%d")
        dkwargs["fromdate"] = fromdate

    if args.todate:
        todate = datetime.datetime.strptime(args.todate, "%Y-%m-%d")
        dkwargs["todate"] = todate

    data0 = bt.feeds.YahooFinanceCSVData(dataname=args.data0, **dkwargs)
    cerebro.adddata(data0, name="Data0")

    cerebro.addstrategy(CloseSMA, period=args.period)

    if args.longonly:
        cerebro.addsizer(LongOnly, stake=args.stake)
    else:
        cerebro.addsizer(FixedReverser, stake=args.stake)

    cerebro.run()
    if args.plot:
        pkwargs = {}
        if args.plot is not True:  # evals to True but is not True
            pkwargs = eval("dict(" + args.plot + ")")  # args were passed

        cerebro.plot(**pkwargs)


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sample for sizer",
    )

    parser.add_argument(
        "--data0",
        required=False,
        default="../../datas/yhoo-1996-2015.txt",
        help="Data to be read in",
    )

    parser.add_argument(
        "--fromdate",
        required=False,
        default="2005-01-01",
        help="Starting date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--todate",
        required=False,
        default="2006-12-31",
        help="Ending date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--cash",
        required=False,
        action="store",
        type=float,
        default=50000,
        help=("Cash to start with"),
    )

    parser.add_argument(
        "--longonly",
        required=False,
        action="store_true",
        help=("Use the LongOnly sizer"),
    )

    parser.add_argument(
        "--stake",
        required=False,
        action="store",
        type=int,
        default=1,
        help=("Stake to pass to the sizers"),
    )

    parser.add_argument(
        "--period",
        required=False,
        action="store",
        type=int,
        default=15,
        help=("Period for the Simple Moving Average"),
    )

    # Plot options
    parser.add_argument(
        "--plot",
        "-p",
        nargs="?",
        required=False,
        metavar="kwargs",
        const=True,
        help=(
            "Plot the read data applying any kwargs passed\n"
            "\n"
            "For example:\n"
            "\n"
            '  --plot style="candle" (to plot candles)\n'
        ),
    )

    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()


if __name__ == "__main__":
    runstrat()
