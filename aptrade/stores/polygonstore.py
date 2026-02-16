#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""Lightweight Polygon REST store for Backtrader.

This provides a minimal API surface used by `feeds/massivedata.py`:
- start(data=...) / stop()
- getTickerQueue(start=False)
- reqHistoricalDataEx(...) -> Queue of bar-like messages, ends with None
- reqHistoricalData(...) -> proxy to reqHistoricalDataEx
- basic stubs for reqMktData/reqRealTimeBars (polling not implemented)

If the `polygon` package (official Polygon Python client) is not
installed the store will raise informative RuntimeErrors when methods
requiring the client are invoked.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    # Try common import name for Polygon REST client
    from polygon import RESTClient as _PolygonRESTClient
except Exception:
    _PolygonRESTClient = None

from aptrade.metabase import MetaParams


class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)

        return cls._singleton


class PolygonStore(metaclass=MetaSingleton):
    """A minimal store backed by Polygon REST client.

    Notes:
    - This implementation focuses on historical data retrieval used by
      `MassiveData`/`MassiveStore` feed logic. Real-time streaming is left as
      a no-op/stub (returns an empty queue) because Polygon realtime needs
      a websocket client and a different architecture.
    """

    BrokerCls = None
    DataCls = None

    params = (
        ("api_key", ""),
        ("_debug", False),
        ("timeout", 10.0),
    )

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Return the registered DataCls with args/kwargs.

        Provides a helpful error if DataCls was not registered yet.
        """
        if cls.DataCls is None:
            try:
                import importlib

                importlib.import_module("backtrader.feeds.polygondata")
            except Exception:
                pass

            if cls.DataCls is None:
                raise RuntimeError(
                    (
                        "No DataCls is registered for store {name}. "
                        "Import 'backtrader.feeds.polygondata' (or the correct feed) "
                        "before calling {name}.getdata(), or set {name}.DataCls manually."
                    ).format(name=cls.__name__)
                )

        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Return an instance of the registered BrokerCls."""
        if cls.BrokerCls is None:
            raise RuntimeError(
                "No BrokerCls is registered for store {}".format(cls.__name__)
            )
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self, **kwargs):
        super(PolygonStore, self).__init__()

        self.debug = True
        self.datas = list()

        # Polygon REST client (may be None if package not installed)
        if _PolygonRESTClient is None:
            self.client = None
        else:
            try:
                # RESTClient usually accepts api_key as first positional arg
                self.client = _PolygonRESTClient(self.p.api_key)
            except TypeError:
                # Some versions may use keyword
                self.client = _PolygonRESTClient(api_key=self.p.api_key)

    def start(self, data=None):
        """Start the store and optionally register a data feed."""
        if data is not None:
            self.datas.append(data)

    def get_notifications(self):
        """Stub method to get notifications (not implemented)."""
        return ""

    def get_granularity(self, timeframe, compression):
        return "minute"

    def fetch_ohlcv(self, symbol, timeframe, since, limit, params={}):
        aggs = []
        for a in self.client.list_aggs(
            "AAPL",
            1,
            "day",
            "2025-01-09",
            "2025-02-10",
            adjusted="true",
            sort="asc",
            limit=120,
        ):
            aggs.append(a)

        print(aggs)

        if self.debug:
            print(
                "Fetching: {}, TF: {}, Since: {}, Limit: {}".format(
                    symbol, timeframe, since, limit
                )
            )
        return self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit, params=params
        )
