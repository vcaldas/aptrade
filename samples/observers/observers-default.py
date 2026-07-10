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
from pathlib import Path

import aptrade as bt

if __name__ == "__main__":
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(bt.Strategy)

    datapath = Path(__file__).resolve().parents[2] / "datas" / "2006-day-001.txt"
    data = bt.feeds.BacktraderCSVData(dataname=str(datapath))
    cerebro.adddata(data)

    cerebro.run()
    if os.environ.get("APTRADE_SAMPLE_SKIP_PLOT") != "1":
        cerebro.plot()
