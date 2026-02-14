# pyright: reportGeneralTypeIssues=false

from pathlib import Path

import pandas as pd
import pytest

from typing import Any, cast

import aptrade as bt
from aptrade.analyzers import TradeHistoryAnalyzer

bt = cast(Any, bt)


class _SingleTradeStrategy(bt.Strategy):
    def __init__(self):
        self._entered = False

    def next(self):
        if not self._entered:
            self.buy(size=1)
            self._entered = True
        elif self.position:
            # close position on the bar after entry
            self.sell(size=self.position.size)


@pytest.fixture
def sample_feed():
    index = pd.date_range("2025-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=index,
    )
    data.index.name = "datetime"
    feed = bt.feeds.PandasData(dataname=data)
    return feed


def test_trade_history_records_closed_trade(sample_feed, tmp_path):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.adddata(sample_feed, name="TEST")
    export_dir = tmp_path / "trade_history"
    cerebro.addanalyzer(
        TradeHistoryAnalyzer,
        _name="trade_history",
        export_dir=str(export_dir),
    )
    cerebro.addstrategy(_SingleTradeStrategy)

    results = cerebro.run()
    strat = results[0]
    analysis = strat.analyzers.trade_history.get_analysis()
    trades = analysis.get("trades", []) if isinstance(analysis, dict) else analysis

    assert trades, "TradeHistoryAnalyzer should record at least one closed trade"
    trade = trades[0]
    assert trade["direction"] in {"long", "short"}
    assert "size" in trade
    assert trade["dt_open"] is not None
    assert trade["dt_close"] is not None
    assert trade["data"] == "TEST"

    exported = Path(export_dir) / "trades_dev.csv"
    assert exported.exists(), "Expected CSV export file to be created"
    exported_df = pd.read_csv(exported)
    assert len(exported_df) == len(trades)
