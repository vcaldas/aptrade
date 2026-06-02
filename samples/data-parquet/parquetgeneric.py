#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2026 Victor Caldas
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
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime as dt

import aptrade as bt


def runstrat():
    args = parse_args()

    fromdate = dt.datetime.fromisoformat(args.fromdate)
    todate = dt.datetime.fromisoformat(args.todate)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(bt.Strategy)

    # Instantiate the feed factory with datastore root + interval.
    pfeed = bt.feeds.ParquetGeneric(
        path=args.path,
        interval=args.interval,
        fromdate=fromdate,
        todate=todate,
    )

    # Retrieve one stock timeseries from the feed factory.
    data = pfeed.getdata(dataname=args.stock, name=args.stock)
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    results = cerebro.run()

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print("Bars loaded: %d" % len(results[0].datas[0]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal ParquetGeneric sample using stock/<name>/<year>/<month>/<day>/<interval>.parquet layout"
    )

    parser.add_argument(
        "--path",
        required=True,
        help="Datastore root path containing the stock/ folder",
    )
    parser.add_argument("--stock", required=True, help="Stock name folder under stock/")
    parser.add_argument(
        "--fromdate",
        required=True,
        help="Inclusive start datetime/date in ISO format (example: 2026-01-01)",
    )
    parser.add_argument(
        "--todate",
        required=True,
        help="Inclusive end datetime/date in ISO format (example: 2026-01-31)",
    )
    parser.add_argument(
        "--interval",
        default="1-minute",
        choices=["day", "trades", "1-minute", "1m"],
        help="Parquet interval file to load",
    )

    return parser.parse_args()


if __name__ == "__main__":
    runstrat()
