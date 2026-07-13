import importlib
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class FakeLine:
    def __init__(self):
        self.zipped_frames = []
        self.price_lines = []

    def set_zipped(self, frame):
        self.zipped_frames.append(frame)

    def create_price_line(self, **kwargs):
        self.price_lines.append(kwargs)


class FakeChart:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded = False
        self.legend_calls = []
        self.price_scale_calls = []
        self.fit_calls = 0
        self.names = []
        self.synced = 0
        self.trades = []
        self.performance_metrics = []
        self.new_windows = 0
        self.subcharts = []
        self.lines = []
        self.histograms = []
        self.marker_lists = []
        self.zipped_frames = []

    def legend(self, **kwargs):
        self.legend_calls.append(kwargs)

    def price_scale(self, **kwargs):
        self.price_scale_calls.append(kwargs)

    def fit(self):
        self.fit_calls += 1

    def load(self):
        self.loaded = True

    def set_name(self, name):
        self.names.append(name)

    def sync_charts(self):
        self.synced += 1

    def set_trades(self, **kwargs):
        self.trades.append(kwargs)

    def set_performance_metrics(self, metrics, strategy_name):
        self.performance_metrics.append((metrics, strategy_name))

    def new_window(self):
        self.new_windows += 1

    def create_subchart(self, **kwargs):
        subchart = FakeChart(**kwargs)
        self.subcharts.append(subchart)
        return subchart

    def create_line(self, *args, **kwargs):
        line = FakeLine()
        self.lines.append((args, kwargs, line))
        return line

    def create_histogram(self, *args, **kwargs):
        line = FakeLine()
        self.histograms.append((args, kwargs, line))
        return line

    def marker_list(self, markers):
        self.marker_lists.append(markers)

    def set_zipped(self, frame):
        self.zipped_frames.append(frame)


@pytest.fixture
def plot_module(monkeypatch):
    fake_widgets = types.ModuleType("bn_lightweight_charts.widgets")
    fake_widgets.HTMLChart_BN = FakeChart

    fake_package = types.ModuleType("bn_lightweight_charts")
    fake_package.widgets = fake_widgets

    monkeypatch.setitem(sys.modules, "bn_lightweight_charts", fake_package)
    monkeypatch.setitem(sys.modules, "bn_lightweight_charts.widgets", fake_widgets)
    sys.modules.pop("aptrade.plot.plot", None)

    module = importlib.import_module("aptrade.plot.plot")
    return importlib.reload(module)


def test_plot_returns_early_when_filename_missing(plot_module, caplog):
    plotter = plot_module.Plot()

    with caplog.at_level(logging.WARNING):
        result = plotter.plot([], filename=None)

    assert result is None
    assert "plot() called with filename=None" in caplog.text


def test_plot_rejects_non_strategy_instances(plot_module, tmp_path):
    plotter = plot_module.Plot()

    with pytest.raises(TypeError, match="Expected Strategy instance"):
        plotter.plot([object()], filename=tmp_path / "chart.html", show=False)


def test_plot_loads_chart_and_opens_browser(plot_module, monkeypatch, tmp_path):
    plotter = plot_module.Plot()
    chart = FakeChart(filename=str(tmp_path / "chart.html"))
    opened_urls = []

    class DummyStrategy(plot_module.Strategy):
        pass

    strategy = object.__new__(DummyStrategy)

    def fake_plot_one(*args, **kwargs):
        plotter.chart = chart

    monkeypatch.setattr(plotter, "plot_one", fake_plot_one)
    monkeypatch.setattr(plot_module.webbrowser, "open", opened_urls.append)

    plotter.plot([strategy], filename=tmp_path / "chart.html", show=True)

    assert chart.loaded is True
    assert opened_urls == [(tmp_path / "chart.html").resolve().as_uri()]


def test_prepare_trades_list_formats_placeholder_rows(plot_module, monkeypatch):
    plotter = plot_module.Plot()
    trades = pd.DataFrame(
        [
            {
                "ref": np.nan,
                "tradeid": np.nan,
                "commission": np.nan,
                "pnl": np.nan,
                "pnlcomm": np.nan,
                "return_pct": np.nan,
                "dateclose": pd.NaT,
                "size": np.nan,
                "barlen": np.nan,
                "priceopen": np.nan,
                "priceclose": np.nan,
                "dateopen": pd.Timestamp("2024-01-02 09:30:00"),
            }
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "o_ref": 11,
                "o_size": 5,
                "o_datetime": pd.Timestamp("2024-01-02 09:30:00"),
                "o_price": 10.5,
                "o_ordtype": "market",
            }
        ]
    )

    plotter.performance = SimpleNamespace(
        gen_trades=lambda data_name, include_open=True: trades,
        gen_orders=lambda data_name: orders,
    )

    monkeypatch.setattr(
        plot_module.Path, "mkdir", lambda self, parents=False, exist_ok=False: None
    )
    monkeypatch.setattr(plot_module.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        plot_module.pd.DataFrame, "to_csv", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(
        plot_module,
        "format_datetime",
        lambda value: f"FMT:{value.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    result = plotter.prepare_trades_list("AAPL")

    assert result == [
        {
            "type": 0,
            "ref": "--",
            "tradeid": "--",
            "commission": "",
            "pnl": "",
            "pnlcomm": "",
            "return_pct": "",
            "dateopen": "FMT:2024-01-02 09:30:00",
            "dateclose": "--",
            "size": 5,
            "barlen": "",
            "priceopen": 10.5,
            "priceclose": "",
        },
        {
            "type": 1,
            "o_ref": 11,
            "o_size": 5,
            "o_datetime": "FMT:2024-01-02 09:30:00",
            "o_price": 10.5,
            "o_ordtype": "market",
        },
    ]


def test_sortdataindicators_routes_indicators_by_plot_position(plot_module):
    plotter = plot_module.Plot()
    data = object()

    class FakeIndicator:
        def __init__(self, clock, *, subplot, plotabove=False):
            self._clock = clock
            self.plotinfo = SimpleNamespace(
                plot=True,
                plotskip=False,
                subplot=subplot,
                plotabove=plotabove,
                plotforce=False,
                plotmaster=None,
            )
            self.initialized = False

        def _plotinit(self):
            self.initialized = True

    up_indicator = FakeIndicator(data, subplot=True, plotabove=True)
    down_indicator = FakeIndicator(data, subplot=True, plotabove=False)
    over_indicator = FakeIndicator(data, subplot=False)
    strategy = SimpleNamespace(
        datas=[data],
        getindicators=lambda: [up_indicator, down_indicator, over_indicator],
    )

    plotter.sortdataindicators(strategy)

    assert plotter.dplots_up[data] == [up_indicator]
    assert plotter.dplots_down[data] == [down_indicator]
    assert plotter.dplots_over[data] == [over_indicator]
    assert up_indicator.initialized is True
    assert down_indicator.initialized is True
    assert over_indicator.initialized is True


def test_frame_from_series_aligns_to_shortest_length(plot_module):
    frame = plot_module.Plot._frame_from_series(
        {"value": [10, 20], "other": [1, 2, 3]},
        default_times=["t0", "t1", "t2", "t3"],
    )

    assert list(frame["time"]) == ["t2", "t3"]
    assert list(frame["value"]) == [10, 20]
    assert list(frame["other"]) == [2, 3]


def test_show_report_returns_none_when_eq_analyzer_is_missing():
    from aptrade.cerebro import Cerebro

    cerebro = Cerebro()
    strategy = SimpleNamespace(
        analyzers=SimpleNamespace(
            getbyname=lambda name: (_ for _ in ()).throw(ValueError(name))
        )
    )
    cerebro.runstrats = [[strategy]]

    assert cerebro.show_report(show=False) is None


def test_show_report_uses_plot_statistics(monkeypatch):
    from aptrade.cerebro import Cerebro

    calls = {}

    class FakeStatistics:
        def report(self, **kwargs):
            calls.update(kwargs)

    fake_stats_module = types.ModuleType("aptrade.plot.stats")
    fake_stats_module.Statistics = FakeStatistics
    monkeypatch.setitem(sys.modules, "aptrade.plot.stats", fake_stats_module)

    analyzer_holder = SimpleNamespace(getbyname=lambda name: "eq-analyzer")
    strategy = SimpleNamespace(
        analyzers=analyzer_holder, __class__=SimpleNamespace(__name__="FakeStrategy")
    )

    cerebro = Cerebro()
    cerebro.runstrats = [[strategy]]

    result = cerebro.show_report(name="Report", filename="out.html", show=False)

    assert result == "out.html"
    assert calls["name"] == "Report"
    assert calls["performance"] == "eq-analyzer"
    assert calls["strats"] == [strategy]
    assert calls["filename"] == "out.html"
    assert calls["show"] is False
