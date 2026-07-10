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

import aptrade as bt
import aptrade.feeds as btfeeds
import aptrade.utils.flushfile


class St(bt.Strategy):
    params = (("usepp1", False), ("plot_on_daily", False))

    def __init__(self):
        autoplot = self.p.plot_on_daily
        self.pp = pp = bt.ind.PivotPoint(self.data1, _autoplot=autoplot)

    def next(self):
        if len(self.pp) == 0:
            return

        txt = ",".join(
            [
                "%04d" % len(self),
                "%04d" % len(self.data0),
                "%04d" % len(self.data1),
                self.data.datetime.date(0).isoformat(),
                "%04d" % len(self.pp),
                "%.2f" % self.pp[0],
            ]
        )

        print(txt)


def runstrat():
    args = parse_args()

    cerebro = bt.Cerebro()

    dkwargs = {}
    if args.fromdate:
        dkwargs["fromdate"] = datetime.datetime.strptime(args.fromdate, "%Y-%m-%d")
    if args.todate:
        dkwargs["todate"] = datetime.datetime.strptime(args.todate, "%Y-%m-%d")

    data = btfeeds.BacktraderCSVData(dataname=args.data, **dkwargs)
    cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Months)

    cerebro.addstrategy(St, usepp1=args.usepp1, plot_on_daily=args.plot_on_daily)
    cerebro.run(runonce=False)
    if args.plot:
        cerebro.plot(style="bar")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sample for pivot point and cross plotting",
    )

    parser.add_argument(
        "--data",
        required=False,
        default="../../datas/2005-2006-day-001.txt",
        help="Data to be read in",
    )

    parser.add_argument(
        "--fromdate",
        required=False,
        default="",
        help="Starting date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--todate",
        required=False,
        default="",
        help="Ending date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--plot", required=False, action="store_true", help=("Plot the result")
    )

    parser.add_argument(
        "--plot-on-daily",
        required=False,
        action="store_true",
        help=("Plot the indicator on the daily data"),
    )

    parser.add_argument(
        "--usepp1",
        required=False,
        action="store_true",
        help=("Keep compatibility with the legacy alternate pivot-point flag"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    runstrat()
