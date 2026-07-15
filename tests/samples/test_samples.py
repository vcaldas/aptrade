"""
Run each sample script that has a matching <script-stem>.txt reference file.
A new sample is picked up automatically once its reference file is added.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"


SAMPLE_ENVS = {
    "samples/observers/observers-default-drawdown.py": {
        "APTRADE_SAMPLE_SKIP_PLOT": "1",
    },
    "samples/observers/observers-default.py": {
        "APTRADE_SAMPLE_SKIP_PLOT": "1",
    },
    "samples/observers/observers-orderobserver.py": {
        "APTRADE_SAMPLE_SKIP_PLOT": "1",
    },
    "samples/tradingcalendar/tcal.py": {
        "APTRADE_SAMPLE_SKIP_DURATION": "1",
    },
    "samples/optimization/optimization.py": {
        "APTRADE_SAMPLE_SKIP_DURATION": "1",
    },
}


SAMPLE_ARGS = {
    "samples/calendar-days/calendar-days.py": [
        "--writer",
        "--wrcsv",
    ],
    "samples/btfd/btfd.py": [
        "--offline",
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--strat",
        "approach='closeclose',prorder=True,prdata=True",
    ],
    "samples/order-close/close-daily.py": [
        "--infile",
        "datas/2005-2006-day-001.txt",
        "--seed",
        "0",
    ],
    "samples/order-close/close-minute.py": [
        "--infile",
        "datas/2006-min-005.txt",
    ],
    "samples/order-execution/order-execution.py": [
        "--infile",
        "datas/2006-day-001.txt",
    ],
    "samples/order-history/order-history.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
    ],
    "samples/order_target/order_target.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--target-size",
    ],
    "samples/observer-benchmark/observer-benchmark.py": [
        "--data0",
        "datas/yhoo-1996-2015.txt",
        "--data1",
        "datas/orcl-1995-2014.txt",
        "--printout",
    ],
    "samples/data-filler/data-filler.py": [
        "--data",
        "datas/2006-01-02-volume-min-001.txt",
        "--fromdate",
        "2006-01-02",
        "--todate",
        "2006-01-02",
        "--tstart",
        "09:15",
        "--tend",
        "09:20",
        "--filler",
        "--writer",
        "--wrcsv",
    ],
    "samples/data-multitimeframe/data-multitimeframe.py": [
        "--indicators",
        "--onlydaily",
        "--dataname",
        "datas/2006-week-001.txt",
        "--timeframe",
        "monthly",
    ],
    "samples/data-pandas/data-pandas-optix.py": [
        "--noplot",
    ],
    "samples/data-pandas/data-pandas.py": [
        "--noplot",
    ],
    "samples/data-replay/data-replay.py": [
        "--dataname",
        "datas/2006-day-001.txt",
        "--timeframe",
        "weekly",
        "--noplot",
    ],
    "samples/data-resample/data-resample.py": [
        "--dataname",
        "datas/2006-day-001.txt",
        "--timeframe",
        "weekly",
        "--noplot",
    ],
    "samples/future-spot/future-spot.py": [
        "--data",
        "datas/2006-day-001.txt",
        "--seed",
        "0",
        "--noplot",
    ],
    "samples/gold-vs-sp500/gold-vs-sp500.py": [
        "--offline",
        "--data0",
        "datas/yhoo-1996-2015.txt",
        "--data1",
        "datas/orcl-1995-2014.txt",
    ],
    "samples/weekdays-filler/weekdaysaligner.py": [
        "--data0",
        "datas/yhoo-2014.txt",
        "--data1",
        "datas/orcl-2014.txt",
        "--fromdate",
        "2014-01-01",
        "--todate",
        "2014-01-31",
        "--filler",
        "--fillclose",
    ],
    "samples/stop-trading/stop-loss-approaches.py": [
        "manual",
        "--data0",
        "datas/2005-2006-day-001.txt",
    ],
    "samples/stoptrail/trail.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
    ],
    "samples/strategy-selection/strategy-selection.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
    ],
    "samples/timers/scheduled.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-03",
        "--todate",
        "2005-01-13",
        "--cerebro",
        "runonce=False",
    ],
    "samples/timers/scheduled-min.py": [
        "--data0",
        "datas/2006-min-005.txt",
        "--fromdate",
        "2006-01-02T09:05:00",
        "--todate",
        "2006-01-02T10:30:00",
        "--cerebro",
        "runonce=False",
        "--strat",
        "repeat=datetime.timedelta(minutes=15),cheat=True",
    ],
    "samples/tradingcalendar/tcal.py": [
        "--offline",
        "--data0",
        "datas/yhoo-2014.txt",
        "--fromdate",
        "2014-01-01",
        "--todate",
        "2014-01-31",
        "--timeframe",
        "Weeks",
    ],
    "samples/tradingcalendar/tcal-intra.py": [
        "--data0",
        "datas/2006-min-005.txt",
        "--fromdate",
        "2006-01-02T09:05:00",
        "--todate",
        "2006-01-02T10:00:00",
    ],
    "samples/talib/talibtest.py": [
        "--data0",
        "datas/yhoo-1996-2015.txt",
    ],
    "samples/talib/tablibsartest.py": [
        "--data0",
        "datas/yhoo-1996-2015.txt",
    ],
    "samples/memory-savings/memory-savings.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
    ],
    "samples/mixing-timeframes/mixing-timeframes.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--multi",
    ],
    "samples/multi-copy/multi-copy.py": [
        "--data0",
        "datas/yhoo-1996-2014.txt",
        "--copydata",
        "--fromdate",
        "2005-02-15",
        "--todate",
        "2005-03-31",
        "--runnext",
    ],
    "samples/multi-example/mult-values.py": [
        "--data0",
        "datas/nvda-1999-2014.txt",
        "--data1",
        "datas/yhoo-1996-2014.txt",
        "--data2",
        "datas/orcl-1995-2014.txt",
        "--todate",
        "2002-06-30",
    ],
    "samples/multidata-strategy/multidata-strategy.py": [
        "--data0",
        "datas/orcl-2003-2005.txt",
        "--data1",
        "datas/yhoo-2003-2005.txt",
        "--fromdate",
        "2003-02-01",
        "--todate",
        "2003-04-15",
    ],
    "samples/multidata-strategy/multidata-strategy-unaligned.py": [
        "--data0",
        "datas/orcl-1995-2014.txt",
        "--data1",
        "datas/yhoo-1996-2014.txt",
        "--fromdate",
        "2003-02-01",
        "--todate",
        "2003-04-15",
    ],
    "samples/multitrades/multitrades.py": [
        "--data",
        "datas/2006-day-001.txt",
        "--fromdate",
        "2006-01-01",
        "--todate",
        "2006-04-30",
        "--mtrade",
        "--printout",
    ],
    "samples/kselrsi/ksignal.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2013-01-01",
        "--todate",
        "2013-12-31",
    ],
    "samples/lrsi/lrsi-test.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-03-31",
    ],
    "samples/lineplotter/lineplotter.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-03-31",
    ],
    "samples/macd-settings/macd-settings.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2006-12-31",
    ],
    "samples/partial-plot/partial-plot.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-03-31",
    ],
    "samples/pivot-point/ppsample.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-02-01",
        "--todate",
        "2005-04-15",
    ],
    "samples/plot-same-axis/plot-same-axis.py": [
        "--data",
        "datas/2006-day-001.txt",
        "--fromdate",
        "2006-01-01",
        "--todate",
        "2006-03-31",
        "--noplot",
    ],
    "samples/psar/psar.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-02-15",
    ],
    "samples/psar/psar-intraday.py": [
        "--data0",
        "datas/2006-min-005.txt",
        "--fromdate",
        "2006-01-02T09:05:00",
        "--todate",
        "2006-01-02T10:00:00",
    ],
    "samples/sharpe-timereturn/sharpe-timereturn.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-03-31",
    ],
    "samples/signals-strategy/signals-strategy.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-06-30",
    ],
    "samples/sigsmacross/sigsmacross.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2011-01-01",
        "--todate",
        "2011-12-31",
    ],
    "samples/sigsmacross/sigsmacross2.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2011-01-01",
        "--todate",
        "2011-12-31",
        "--noplot",
    ],
    "samples/sizertest/sizertest.py": [
        "--data0",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-06-30",
    ],
    "samples/slippage/slippage.py": [
        "--data",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-06-30",
    ],
    "samples/relative-volume/relative-volume.py": [
        "--data",
        "datas/2006-01-02-volume-min-001.txt",
        "--fromdate",
        "2006-01-01",
        "--todate",
        "2006-01-03",
        "--writer",
        "--wrcsv",
    ],
    "samples/renko/renko.py": [
        "--data0",
        "datas/2005-2006-day-001.txt",
        "--fromdate",
        "2005-01-01",
        "--todate",
        "2005-03-31",
        "--cerebro",
        "runonce=False",
    ],
    "samples/optimization/optimization.py": [
        "--data",
        "datas/2006-day-001.txt",
        "--fromdate",
        "2006-01-01",
        "--todate",
        "2006-03-31",
        "--maxcpus",
        "1",
        "--ma_low",
        "10",
        "--ma_high",
        "11",
        "--m1_low",
        "12",
        "--m1_high",
        "13",
        "--m2_low",
        "26",
        "--m2_high",
        "27",
        "--m3_low",
        "9",
        "--m3_high",
        "10",
    ],
    "samples/massive-test/massive-test.py": [
        "--data",
        "datas/yhoo-1996-2015.txt",
        "--fromdate",
        "2006-01-01",
        "--todate",
        "2006-03-31",
        "--writer",
        "--wrcsv",
    ],
    "samples/resample-tickdata/resample-tickdata.py": [
        "--dataname",
        "datas/ticksample.csv",
        "--timeframe",
        "seconds",
        "--compression",
        "30",
        "--writer",
        "--wrcsv",
        "--noplot",
    ],
}


SAMPLE_SMOKE_TESTS = {
    "samples/data-resample/data-resample.py",
    "samples/future-spot/future-spot.py",
    "samples/gold-vs-sp500/gold-vs-sp500.py",
    "samples/lineplotter/lineplotter.py",
    "samples/lrsi/lrsi-test.py",
    "samples/multitrades/mtradeobserver.py",
    "samples/observers/observers-default.py",
    "samples/observers/orderobserver.py",
    "samples/partial-plot/partial-plot.py",
    "samples/pivot-point/pivotpoint.py",
    "samples/plot-same-axis/plot-same-axis.py",
    "samples/relative-volume/relvolbybar.py",
    "samples/renko/renko.py",
    "samples/signals-strategy/signals-strategy.py",
    "samples/sigsmacross/sigsmacross2.py",
    "samples/sizertest/sizertest.py",
    "samples/talib/tablibsartest.py",
    "samples/talib/talibtest.py",
    "samples/weekdays-filler/weekdaysfiller.py",
}


PREFERRED_SCRIPT_NAMES = {
    "calmar": "calmar-test.py",
    "data-bid-ask": "bidask.py",
    "data-parquet": "parquetgeneric.py",
    "kselrsi": "ksignal.py",
    "lrsi": "lrsi-test.py",
    "multi-example": "mult-values.py",
    "multitrades": "multitrades.py",
    "order-close": "close-daily.py",
    "pivot-point": "ppsample.py",
    "pyfolio2": "pyfoliotest.py",
    "pyfoliotest": "sample_pyfoliotest.py",
    "stop-trading": "stop-loss-approaches.py",
    "stoptrail": "trail.py",
    "weekdays-filler": "weekdaysfiller.py",
}


def _cases():
    for folder in sorted(path for path in SAMPLES_DIR.iterdir() if path.is_dir()):
        for ref in sorted(folder.glob("*.txt")):
            script = folder / f"{ref.stem}.py"
            if not script.exists():
                continue
            yield pytest.param(script, ref, id=f"{folder.name}:{script.stem}")

    for script_key in sorted(SAMPLE_SMOKE_TESTS):
        script = ROOT_DIR / script_key
        yield pytest.param(script, None, id=script.stem)


@pytest.mark.parametrize("script,reference", list(_cases()))
def test_sample_output(script, reference):
    script_key = script.relative_to(ROOT_DIR).as_posix()
    env = dict(os.environ)
    env.update(SAMPLE_ENVS.get(script_key, {}))
    result = subprocess.run(
        [sys.executable, str(script), *SAMPLE_ARGS.get(script_key, [])],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    if reference is None:
        return

    assert result.stdout == reference.read_text()
