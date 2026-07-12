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


import aptrade as bt
import aptrade.indicators as btind
import testcommon

chkdatas = 1


class CurrentTestStrategy(bt.Strategy):
    params = {"main": False}

    def __init__(self):
        btind.SMA()


def test_run(main=False):
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    cerebros = testcommon.runtest(
        datas,
        CurrentTestStrategy,
        main=main,
        plot=main,
        writer=(bt.WriterStringIO, {"csv": True}),
    )

    for cerebro in cerebros:
        writer = cerebro.runwriters[0]
        if main:
            # writer.out.seek(0)
            for ln in writer.out:
                print(ln.rstrip("\r\n"))

        else:
            lines = iter(writer.out)
            ln = next(lines).rstrip("\r\n")
            assert ln == "=" * 79

            count = 0
            while True:
                ln = next(lines).rstrip("\r\n")
                if ln[0] == "=":
                    break
                count += 1

            assert count == 256  # header + 256 lines data


if __name__ == "__main__":
    test_run(main=True)
