#!/usr/bin/env python
###############################################################################
#
# Copyright (C) 2015-2021 Daniel Rodriguez
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
"""IBStore — singleton IB connection and data/order coordination hub.

This module is the backbone of the library.  Everything that touches the IB
API goes through the ``IBStore`` singleton:

* Connection lifecycle — ``ib_insync.IB.connectAsync()`` is wrapped to be
  compatible with Python 3.14's strict asyncio task requirements.
* Historical data — ``reqHistoricalDataEx`` maps a (begindate, enddate) pair
  to one or more IB ``reqHistoricalData`` calls, automatically choosing the
  correct duration/bar-size combination from IB's published limits.
* Live data — ``reqRealTimeBars`` (5-second RTBars) and ``reqMktData``
  (RTVolume ticks or BBO for CASH/CFD instruments).
* Orders — ``placeOrder`` and ``cancelOrder`` proxy to ``ib_insync``.
* Account — ``updateAccountValue`` / ``get_acc_cash`` / ``get_acc_value``
  read account data from ``ib.accountValues()``.

Duration / bar-size lookup tables
----------------------------------
IB imposes strict limits on which bar sizes are valid for a given request
duration.  The class-level ``_durations`` dict encodes these limits as::

    duration_string → tuple of valid bar size strings

``__init__`` inverts this into ``self.revdur``::

    (TimeFrame, compression) → sorted list of valid duration strings

This lets ``reqHistoricalDataEx`` instantly find the longest duration that
fits the requested date range without making multiple IB round-trips.
"""

import asyncio
import collections
import logging
import queue
import random
import threading
from copy import copy
from datetime import datetime, timedelta
from importlib import import_module

from ib_async import IB, Contract, StartupFetch, util

from aptrade.dataseries import TimeFrame
from aptrade.metabase import MetaParams
from aptrade.position import Position
from aptrade.utils import AutoDict

logger = logging.getLogger(__name__)


def _ts2dt(tstamp=None):
    """Convert an IB millisecond timestamp to a UTC ``datetime``.

    IB RTVolume tick strings encode the time as milliseconds since the Unix
    epoch.  This helper splits the value into seconds and microseconds and
    returns a timezone-naive UTC ``datetime``.

    Args:
        tstamp: Integer or string millisecond timestamp, or ``None``.

    Returns:
        datetime: UTC datetime.  If ``tstamp`` is falsy (None, 0, empty
        string) the current UTC time is returned.
    """
    if not tstamp:
        return datetime.utcnow()

    sec, msec = divmod(int(tstamp), 1000)
    usec = msec * 1000
    return datetime.utcfromtimestamp(sec).replace(microsecond=usec)


class RTVolume:
    """Parse a tickString tickType 48 (RTVolume) event from the IB API.

    IB delivers RTVolume as a semicolon-separated string with six fields::

        price;size;timestamp_ms;cumulative_volume;vwap;single_trade_flag

    Example::

        '150.25;100;1609459200000;50000;150.10;true'

    The ``_fields`` list defines the field name and conversion function for
    each position.  The parsed values are stored as instance attributes.

    Attributes:
        price (float): Last trade price.
        size (int): Last trade size.
        datetime (datetime): UTC datetime of the tick (from ``_ts2dt``).
        volume (int): Cumulative volume for the session.
        vwap (float): Volume-weighted average price.
        single (bool): True if this is a single-lot trade.

    The class can also be instantiated with ``rtvol=''`` and a ``price``
    keyword to simulate a tick from a tickPrice event (e.g. for CASH assets
    that do not emit RTVolume).
    """

    _fields = [
        ("price", float),
        ("size", int),
        ("datetime", _ts2dt),
        ("volume", int),
        ("vwap", float),
        ("single", bool),
    ]

    def __init__(self, rtvol="", price=None, tmoffset=None):
        """Parse an RTVolume string into attributes.

        Args:
            rtvol (str): Semicolon-separated RTVolume string, or ``''`` to
                create an empty/default instance.
            price (float, optional): Override the parsed price.  Useful when
                simulating RTVolume from a tickPrice event.
            tmoffset (timedelta, optional): Server-to-local time offset added
                to the parsed datetime.
        """
        # Split the string and iterate; if empty, call each func with no args
        # to get a sensible default (0, 0.0, utcnow, etc.).
        tokens = iter(rtvol.split(";"))
        for name, func in self._fields:
            setattr(self, name, func(next(tokens)) if rtvol else func())

        if price is not None:
            self.price = price  # override with caller-supplied price

        if tmoffset is not None:
            self.datetime += tmoffset  # apply server time offset


async def _update_ticks_once(ib, contract, ticks, q_ticks):
    """Subscribe to market data and deliver the first batch of ticks.

    Calls ``ib.reqMktData`` with the given tick type string, then waits
    for the first ``pendingTickersEvent`` and puts all received ``Ticker``
    objects onto ``q_ticks``.

    Args:
        ib (ib_insync.IB): Live IB connection.
        contract (ib_insync.Contract): The instrument to subscribe to.
        ticks (str): Tick type string, e.g. ``'233'`` for RTVolume.
            Pass ``''`` for CASH markets (BBO ticks only).
        q_ticks (queue.Queue): Destination queue for ``Ticker`` objects.
    """
    ib.reqMktData(contract, ticks)

    async for tickers in ib.pendingTickersEvent:
        for ticker in tickers:
            q_ticks.put(ticker)
        return


class MetaSingleton(MetaParams):
    """Metaclass that enforces the singleton pattern on ``IBStore``.

    The first call to ``IBStore(...)`` creates the instance and caches it in
    ``cls._singleton``.  All subsequent calls — regardless of the arguments —
    return the same cached instance.

    This guarantees that only one ``ib_insync.IB`` connection exists per
    Python process, which is required because TWS/Gateway enforces unique
    ``clientId`` values per connection.
    """

    def __init__(self, name, bases, dct):
        super().__init__(name, bases, dct)
        self._singleton = None  # initialise the cache slot

    def __call__(self, *args, **kwargs):
        if self._singleton is None:
            # First call — create and cache the instance.
            self._singleton = super().__call__(*args, **kwargs)
        return self._singleton


class IBStore(metaclass=MetaSingleton):
    """Singleton that owns the ``ib_insync.IB`` connection and coordinates
    all data and order requests.

    There is exactly **one** ``IBStore`` instance per Python process.  All
    ``IBData`` feeds and the ``IBBroker`` share it.  Pass connection
    parameters the first time the class is instantiated; subsequent calls
    return the cached instance unchanged.

    The parameters can also be specified in the classes which use this store,
    like ``IBData`` and ``IBBroker``

    Params:

      - ``host`` (default:``127.0.0.1``): where IB TWS or IB Gateway are
        actually running. And although this will usually be the localhost, it
        must not be

      - ``port`` (default: ``7496``): port to connect to. The demo system uses
        ``7497``

      - ``clientId`` (default: ``None``): which clientId to use to connect to
        TWS.

        ``None``: generates a random id between 1 and 65535
        An ``integer``: will be passed as the value to use.

      - ``_debug`` (default: ``False``)

        Print all messages received from TWS to standard output

      - ``timeout`` (default: ``3.0``)

                Connection timeout in seconds

      - ``timeoffset`` (default: ``True``)

        If True, the time obtained from ``reqCurrentTime`` (IB Server time)
        will be used to calculate the offset to localtime and this offset will
        be used for the price notifications (tickPrice events, for example for
        CASH markets) to modify the locally calculated timestamp.

        The time offset will propagate to other parts of the ``backtrader``
        ecosystem like the **resampling** to align resampling timestamps using
        the calculated offset.

      - ``timerefresh`` (default: ``60.0``)

        Time in seconds: how often the time offset has to be refreshed

      - ``indcash`` (default: ``True``)

        Manage IND codes as if they were cash for price retrieval

      - ``readonly`` (default: ``False``)

        Set to ``True`` when the IB API is configured in read-only mode
        (account data and market data only, no order placement).

      - ``account`` (default: ``''``)

        Main account identifier to subscribe to.  Empty string means the
        first account returned by ``managedAccounts()``.
    """

    # Base ID for historical/realtime data request IDs.  Using a high base
    # (0x01000000 = 16 777 216) ensures data request IDs never collide with
    # the low-numbered order IDs assigned by TWS.
    REQIDBASE = 0x01000000

    # These class attributes are populated by the MetaIBBroker and MetaIBData
    # metaclasses when those classes are first imported.  They allow IBStore
    # to instantiate the broker and data feed without a circular import.
    BrokerCls = None  # set by MetaIBBroker when IBBroker class is created
    DataCls = None  # set by MetaIBData when IBData class is created

    params = (
        ("host", "127.0.0.1"),
        ("port", 7496),
        ("clientId", None),  # None → random 1–65535
        ("_debug", False),
        ("timeout", 3.0),  # connection timeout in seconds
        ("timeoffset", True),  # adjust timestamps by server-local offset
        ("timerefresh", 60.0),  # how often (seconds) to re-sync server time
        ("indcash", True),  # treat IND security type as CASH for data
        ("readonly", False),  # read-only IB API mode
        ("account", ""),  # account for updates ('' = first managed)
    )

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Instantiate and return a ``DataCls`` (i.e. ``IBData``) object.

        Convenience factory so callers can do ``store.getdata(...)`` without
        importing ``IBData`` directly.
        """
        if cls.DataCls is None:
            import_module("aptrade.feeds.ibdata")
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Instantiate and return a ``BrokerCls`` (i.e. ``IBBroker``) object.

        Convenience factory so callers can do ``store.getbroker(...)`` without
        importing ``IBBroker`` directly.
        """
        if cls.BrokerCls is None:
            import_module("aptrade.brokers.ibbroker")
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self):
        """Initialise in-memory state for the singleton store.

        Network connection is intentionally deferred to ``connect()`` and is
        triggered from ``start()`` so instance creation is side-effect free.
        """
        super().__init__()

        self._init_runtime_state()
        self._init_connection_config()
        self._build_duration_index()

    def _init_runtime_state(self):
        self._env = None
        self.broker = None
        self.datas = []

        self.iscash = {}
        self.acc_cash = AutoDict()
        self.acc_value = AutoDict()
        self.acc_upds = AutoDict()

        self.positions = collections.defaultdict(Position)
        self.orderid = None
        self.managed_accounts = []
        self.notifs = queue.Queue()

        self.last_tick = None

        self.tmoffset = timedelta(0)
        self._lock_tmoffset = threading.Lock()
        self._time_refresh_stop = threading.Event()
        self._time_refresh_thread = None

        self.ib = IB()
        self._connected = False
        self._asyncio_initialized = False

    def _init_connection_config(self):
        self.clientId = (
            random.randint(1, pow(2, 16) - 1)
            if self.p.clientId is None
            else self.p.clientId
        )
        self.timeout = 2 if self.p.timeout is None else self.p.timeout
        self.readonly = False if self.p.readonly is None else self.p.readonly
        self.account = "" if self.p.account is None else self.p.account

    def _build_duration_index(self):
        def keyfn(x):
            n, t = x.split()
            tf, comp = self._sizes[t]
            return (tf, int(n) * comp)

        def key2fn(x):
            n, d = x.split()
            tf = self._dur2tf[d]
            return (tf, int(n))

        self.revdur = collections.defaultdict(list)
        for duration, barsizes in self._durations.items():
            for barsize in barsizes:
                self.revdur[keyfn(barsize)].append(duration)

        for barsize in self.revdur:
            self.revdur[barsize].sort(key=key2fn)

    def _ensure_asyncio(self):
        if self._asyncio_initialized:
            return

        if self.p._debug:
            util.logToConsole(level=logging.DEBUG)

        util.patchAsyncio()
        util.startLoop()
        self._asyncio_initialized = True

    def connect(self):
        if self._connected and self.ib.isConnected():
            return

        self._ensure_asyncio()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            asyncio.ensure_future(
                self.ib.connectAsync(
                    host=self.p.host,
                    port=self.p.port,
                    clientId=self.clientId,
                    timeout=self.timeout,
                    readonly=self.readonly,
                    account=self.account,
                    # Keep startup sync lightweight; full account refresh is
                    # triggered explicitly by broker bootstrap.
                    fetchFields=(StartupFetch.POSITIONS | StartupFetch.ORDERS_OPEN),
                )
            )
        )
        self._connected = True

    def _start_time_refresh(self):
        if not self.p.timeoffset:
            return

        if self._time_refresh_thread and self._time_refresh_thread.is_alive():
            return

        self._time_refresh_stop.clear()
        self._time_refresh_thread = threading.Thread(
            target=self._time_refresh_loop,
            name="ibstore-time-refresh",
            daemon=True,
        )
        self._time_refresh_thread.start()

    def _stop_time_refresh(self):
        self._time_refresh_stop.set()
        if self._time_refresh_thread and self._time_refresh_thread.is_alive():
            self._time_refresh_thread.join(timeout=2)

    def _time_refresh_loop(self):
        # Prime offset immediately before waiting for the periodic refresh.
        self._safe_req_current_time()

        while not self._time_refresh_stop.wait(self.p.timerefresh):
            self._safe_req_current_time()

    def start(self, data=None, broker=None):
        """Register a data feed or broker with the store.

        Called by ``IBData.start()`` and ``IBBroker.start()`` respectively.

        Args:
            data (IBData, optional): Data feed to register.  A new queue is
                created, primed with ``None``, and returned so the feed can
                begin its FSM loop.
            broker (IBBroker, optional): Broker to register.

        Returns:
            queue.Queue or None: If ``data`` is provided, returns the initial
            ticker queue.  Otherwise returns ``None``.
        """
        self.connect()

        if data is not None:
            self._env = data._env
            self.datas.append(data)
            # A queue pre-loaded with None kick-starts the data feed's _load loop.
            return self.getTickerQueue(start=True)

        elif broker is not None:
            self.broker = broker
            self._start_time_refresh()

    def stop(self):
        """Disconnect from TWS/Gateway.

        Safe to call even if the connection was never established.
        """
        self._stop_time_refresh()

        try:
            self.ib.disconnect()
            self._connected = False
        except AttributeError:
            pass

    def get_notifications(self):
        """Drain and return all pending store-level notifications.

        Pushes a ``None`` sentinel onto the queue so we know where to stop
        reading (the background connection thread may still be adding items).

        Returns:
            list: All notification items queued since the last call.
        """
        self.notifs.put(None)  # sentinel marks the end of this batch
        notifs = []
        while True:
            notif = self.notifs.get()
            if notif is None:  # sentinel reached
                break
            notifs.append(notif)
        return notifs

    def managedAccounts(self):
        """Fetch and cache the list of accounts managed by this TWS session.

        This is the first message in the TWS login stream.  Also triggers a
        ``reqCurrentTime()`` call to begin the server time-offset calculation.
        """
        logger.info("[ibstore-checkpoint] managedAccounts:request")
        self.managed_accounts = self.ib.managedAccounts()
        logger.info(
            "[ibstore-checkpoint] managedAccounts:received count=%s accounts=%s",
            len(self.managed_accounts),
            self.managed_accounts,
        )

        if self.p.timeoffset:
            logger.info("[ibstore-checkpoint] managedAccounts:start_time_refresh")
            self._start_time_refresh()
        else:
            logger.info("[ibstore-checkpoint] managedAccounts:timeoffset_disabled")

    def _safe_req_current_time(self):
        try:
            logger.info("[ibstore-checkpoint] reqCurrentTime:thread_start")
            self.reqCurrentTime()
            logger.info("[ibstore-checkpoint] reqCurrentTime:thread_end")
        except Exception:
            logger.exception("[ibstore-checkpoint] reqCurrentTime:thread_failed")

    def currentTime(self, msg):
        """Handle a server time response and update the local time offset.

        If ``timeoffset`` is enabled, computes the difference between the IB
        server clock and the local clock and schedules a refresh timer.

        Args:
            msg: A message object with a ``time`` attribute (Unix timestamp).
        """
        if not self.p.timeoffset:
            return

        curtime = datetime.fromtimestamp(float(msg.time))
        with self._lock_tmoffset:
            self.tmoffset = curtime - datetime.now()

    def timeoffset(self):
        """Return the current server-to-local time offset as a ``timedelta``.

        Returns:
            timedelta: Offset to add to local time to obtain server time.
        """
        with self._lock_tmoffset:
            return self.tmoffset

    def reqCurrentTime(self):
        """Ask TWS for its current time.

        Updates ``self.tmoffset`` immediately from the IB response.
        """
        logger.info("[ibstore-checkpoint] reqCurrentTime:before_ib_call")
        response = self.ib.reqCurrentTime()

        if response is None:
            logger.info("[ibstore-checkpoint] reqCurrentTime:no_response")
            return

        if isinstance(response, datetime):
            curtime = response
        else:
            curtime = datetime.fromtimestamp(float(getattr(response, "time", response)))

        with self._lock_tmoffset:
            self.tmoffset = curtime - datetime.now()

        logger.info("[ibstore-checkpoint] reqCurrentTime:after_ib_call")

    def nextOrderId(self):
        """Return the next available order ID from TWS.

        Each call to ``ib.client.getReqId()`` increments an internal TWS
        counter and returns a unique integer suitable for use as an IB order ID.

        Returns:
            int: Next order ID.
        """
        self.orderid = self.ib.client.getReqId()
        return self.orderid

    def getTickerQueue(self, start=False):
        """Create and return a new ``queue.Queue`` for data delivery.

        Args:
            start (bool): If ``True``, pre-load the queue with a single
                ``None`` item.  This kick-starts an ``IBData._load`` loop that
                is waiting on the queue before the first real data arrives.

        Returns:
            queue.Queue: An empty queue, or one pre-loaded with ``None``.
        """
        q = queue.Queue()
        if start:
            q.put(None)
        return q

    def getContractDetails(self, contract, maxcount=None):
        """Fetch full contract details from IB for a preliminary contract.

        Resolves the IB ``conId`` and all exchange/currency/tick-size details
        for the given skeletal ``Contract`` object.

        Args:
            contract (ib_insync.Contract): Skeletal contract (from
                ``makecontract``).
            maxcount (int, optional): Maximum number of matches to accept.
                If the IB API returns more than ``maxcount`` matches (ambiguous
                contract), returns ``None`` and enqueues an error notification.

        Returns:
            list of ContractDetails or None: List of matching contract details,
            or ``None`` if the contract is not found / ambiguous.
        """
        cds = self.ib.reqContractDetails(contract)

        if not cds:
            err = "Ambiguous contract: none/multiple answers received"
            self.notifs.put((err, cds, {}))
            return None

        if maxcount and len(cds) > maxcount:
            # Multiple matches — use the one with the lowest conId (primary listing).
            cds = sorted(cds, key=lambda cd: cd.contract.conId)[:maxcount]

        return cds

    def reqHistoricalDataEx(
        self,
        contract,
        enddate,
        begindate,
        timeframe,
        compression,
        what=None,
        useRTH=False,
        tz="",
        sessionend=None,
    ):
        """Request historical data for a date range, splitting if necessary.

        Unlike ``reqHistoricalData`` (which requires a fixed IB duration
        string), this method accepts two datetimes and automatically selects
        the best duration from the ``_durations`` table.

        If the range fits within a single IB request, one call is made.  If
        the range is too large (no single duration covers it), the maximum
        available duration is used and a single request is made up to that
        limit — further splitting is left to the caller (``IBData._st_start``
        handles retries via queue draining).

        For CASH/CFD contracts the default ``what`` is forced to ``'BID'``
        because IB does not return TRADES data for those instrument types.

        Args:
            contract (ib_insync.Contract): Resolved IB contract.
            enddate (datetime or None): End of the requested range.  ``None``
                means "up to now".
            begindate (datetime or None): Start of the requested range.
                ``None`` means "use the maximum available duration".
            timeframe: Backtrader ``TimeFrame`` constant.
            compression (int): Backtrader bar compression.
            what (str, optional): Data to show: ``'TRADES'``, ``'BID'``,
                ``'ASK'``, ``'MIDPOINT'``, etc.
            useRTH (bool): ``True`` = regular trading hours only.
            tz: Timezone object with a ``.zone`` attribute (e.g. pytz).
            sessionend: Session end time (passed to IB API).

        Returns:
            queue.Queue: Queue pre-loaded with ``ib_insync.BarData`` objects.
        """
        # Preserve local variables for error reporting.
        kwargs = locals().copy()
        kwargs.pop("self", None)

        if timeframe < TimeFrame.Seconds:
            # IB historical data API does not support sub-second granularity.
            return self.getTickerQueue(start=True)

        if enddate is None:
            enddate = datetime.now()

        if begindate is None:
            # No start date — use the maximum duration for this bar size.
            duration = self.getmaxduration(timeframe, compression)
            if duration is None:
                err = (
                    "No duration for historical data request for timeframe/compresison"
                )
                self.notifs.put((err, (), kwargs))
                return self.getTickerQueue(start=True)
            barsize = self.tfcomp_to_size(timeframe, compression)
            if barsize is None:
                err = (
                    "No supported barsize for historical data request for "
                    "timeframe/compresison"
                )
                self.notifs.put((err, (), kwargs))
                return self.getTickerQueue(start=True)

            return self.reqHistoricalData(
                contract=contract,
                enddate=enddate,
                duration=duration,
                barsize=barsize,
                what=what,
                useRTH=useRTH,
                tz=tz,
                sessionend=sessionend,
            )

        # Find the shortest valid duration that covers (begindate → enddate).
        durations = self.getdurations(timeframe, compression)

        duration = None
        for dur in durations:
            intdate = self.dt_plus_duration(begindate, dur)
            if intdate >= enddate:
                # This duration covers the full range in one request.
                intdate = enddate
                duration = dur
                break

        if duration is None:
            # No single duration spans the full range — use the largest
            # available and let the caller handle the remainder.
            duration = durations[-1]

        barsize = self.tfcomp_to_size(timeframe, compression)

        return self._request_historical_data(
            contract=contract,
            enddate=intdate,
            duration=duration,
            barsize=barsize,
            what=what,
            useRTH=useRTH,
            tz=tz,
        )

    def reqHistoricalData(
        self,
        contract,
        enddate,
        duration,
        barsize,
        what=None,
        useRTH=False,
        tz="",
        sessionend=None,
    ):
        """Request historical data with an explicit IB duration string.

        Lower-level wrapper around ``ib.reqHistoricalData``.  The result bars
        are placed onto a new queue and returned synchronously (``ib_insync``
        makes this call blocking via its patched event loop).

        For CASH/CFD contracts ``what`` defaults to ``'BID'`` because TRADES
        data is not available for those instruments.

        Args:
            contract (ib_insync.Contract): Resolved IB contract.
            enddate (datetime): End datetime of the request.
            duration (str): IB duration string, e.g. ``'1 D'``, ``'3600 S'``.
            barsize (str): IB bar size string, e.g. ``'5 mins'``, ``'1 hour'``.
            what (str, optional): Data type (TRADES, BID, ASK, MIDPOINT, …).
            useRTH (bool): Regular trading hours only.
            tz: Timezone with ``.zone`` attribute.
            sessionend: Session end time.

        Returns:
            queue.Queue: Queue containing ``ib_insync.BarData`` objects.
        """
        return self._request_historical_data(
            contract=contract,
            enddate=enddate,
            duration=duration,
            barsize=barsize,
            what=what,
            useRTH=useRTH,
            tz=tz,
        )

    def _request_historical_data(
        self, contract, enddate, duration, barsize, what, useRTH, tz
    ):
        q = self.getTickerQueue()
        resolved_what = self._resolve_historical_what(contract, what)
        end_dt_with_tz = self._format_ib_datetime(enddate, tz)

        histdata = self.ib.reqHistoricalData(
            contract,
            end_dt_with_tz,
            duration,
            barsize,
            resolved_what,
            useRTH,
            2,
        )
        for msg in histdata:
            q.put(msg)

        return q

    def _resolve_historical_what(self, contract, what):
        if contract.secType in ["CASH", "CFD"] and not what:
            return "BID"

        if contract.secType in ["IND"] and self.p.indcash and not what:
            return "BID"

        return what or "TRADES"

    def _format_ib_datetime(self, value, tz):
        tz_name = ""
        if isinstance(tz, str):
            tz_name = tz
        elif hasattr(tz, "zone"):
            tz_name = tz.zone
        elif hasattr(tz, "key"):
            tz_name = tz.key

        suffix = f" {tz_name}" if tz_name else ""
        return value.strftime("%Y%m%d %H:%M:%S") + suffix

    def reqRealTimeBars(self, contract, useRTH=False, duration=5):
        """Subscribe to 5-second real-time bars and return one batch.

        Calls ``ib.reqRealTimeBars`` and then sleeps for ``duration`` seconds
        to allow at least one bar to arrive, then drains the bar list into a
        queue.

        Note: ``reqRealTimeBars`` always returns 5-second MIDPOINT bars
        regardless of the ``duration`` parameter (IB API limitation).

        Args:
            contract (ib_insync.Contract): The instrument to subscribe to.
            useRTH (bool): Regular trading hours only.
            duration (int): Seconds to wait for bars (default 5).

        Returns:
            queue.Queue: Queue containing ``ib_insync.RealTimeBar`` objects.
        """
        q = self.getTickerQueue()

        rtb = self.ib.reqRealTimeBars(contract, duration, "MIDPOINT", useRTH=useRTH)
        self.ib.sleep(duration)  # wait for at least one bar to arrive
        for bar in rtb:
            q.put(bar)
        return q

    def reqMktData(self, contract, what=None):
        """Subscribe to market data ticks and return the first batch.

        For equity instruments, requests RTVolume (tick type 233), which
        delivers last-sale price, size, timestamp, cumulative volume, and VWAP.

        For CASH/CFD instruments, RTVolume is not available; IB sends BBO
        (bid/ask) ticks instead.

        Args:
            contract (ib_insync.Contract): The instrument to subscribe to.
            what (str, optional): ``'ASK'`` to use ask price for CASH
                instruments (default is bid).

        Returns:
            queue.Queue: Queue containing ``ib_insync.Ticker`` tick objects.
        """
        q = self.getTickerQueue()
        ticks = "233"  # RTVolume tick type: last, lastSize, rtVolume, rtTime, vwap

        if contract.secType in ["CASH", "CFD"]:
            # CASH markets do not produce RTVolume; IB sends BBO ticks instead.
            ticks = ""
            if what == "ASK":
                pass  # will receive ask price in the BBO tick

        # Run the async tick subscription coroutine synchronously.
        q_ticks = queue.Queue()
        util.run(_update_ticks_once(self.ib, contract, ticks, q_ticks))

        # Drain the raw Ticker objects and forward individual ticks to the
        # output queue, deduplicating against the last seen tick.
        while not q_ticks.empty():
            ticker = q_ticks.get()
            for tick in ticker.ticks:
                if tick != self.last_tick:
                    self.last_tick = tick
                    q.put(tick)

        return q

    # -------------------------------------------------------------------------
    # Duration / bar-size lookup tables
    # -------------------------------------------------------------------------
    # IB imposes strict pairing rules between request duration strings and
    # bar sizes.  _durations maps each valid duration to the bar sizes it
    # supports.  This is used in __init__ to build the inverted table
    # self.revdur, and by reqHistoricalDataEx to select the right duration.

    _durations = {
        "60 S": ("1 secs", "5 secs", "10 secs", "15 secs", "30 secs", "1 min"),
        # 120 seconds - 2 mins
        "120 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
        ),
        # 180 seconds - 3 mins
        "180 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
        ),
        # 300 seconds - 5 mins
        "300 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
        ),
        # 600 seconds - 10 mins
        "600 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
        ),
        # 900 seconds - 15 mins
        "900 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
        ),
        # 1200 seconds - 20 mins
        "1200 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
        ),
        # 1800 seconds - 30 mins
        "1800 S": (
            "1 secs",
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
        ),
        # 3600 seconds - 1 hour
        "3600 S": (
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
        ),
        # 7200 seconds - 2 hours
        "7200 S": (
            "5 secs",
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
        ),
        # 10800 seconds - 3 hours
        "10800 S": (
            "10 secs",
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
        ),
        # 14400 seconds - 4 hours
        "14400 S": (
            "15 secs",
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
        ),
        # 28800 seconds - 8 hours
        "28800 S": (
            "30 secs",
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
        ),
        # 1 day
        "1 D": (
            "1 min",
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
            "1 day",
        ),
        # 2 days
        "2 D": (
            "2 mins",
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
            "1 day",
        ),
        # 1 week
        "1 W": (
            "3 mins",
            "5 mins",
            "10 mins",
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
            "1 day",
            "1 W",
        ),
        # 2 weeks
        "2 W": (
            "15 mins",
            "20 mins",
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
            "1 day",
            "1 W",
        ),
        # 1 month
        "1 M": (
            "30 mins",
            "1 hour",
            "2 hours",
            "3 hours",
            "4 hours",
            "8 hours",
            "1 day",
            "1 W",
            "1 M",
        ),
        # 2 – 11 months (only daily, weekly, and monthly bars are valid)
        "2 M": ("1 day", "1 W", "1 M"),
        "3 M": ("1 day", "1 W", "1 M"),
        "4 M": ("1 day", "1 W", "1 M"),
        "5 M": ("1 day", "1 W", "1 M"),
        "6 M": ("1 day", "1 W", "1 M"),
        "7 M": ("1 day", "1 W", "1 M"),
        "8 M": ("1 day", "1 W", "1 M"),
        "9 M": ("1 day", "1 W", "1 M"),
        "10 M": ("1 day", "1 W", "1 M"),
        "11 M": ("1 day", "1 W", "1 M"),
        # 1 year
        "1 Y": ("1 day", "1 W", "1 M"),
    }

    # Maps IB bar size unit strings to (Backtrader TimeFrame, multiplier).
    # The multiplier converts the numeric part of a bar size string to a
    # Backtrader compression value.  e.g. '2 hours' → (Minutes, 60) → 120.
    _sizes = {
        "secs": (TimeFrame.Seconds, 1),
        "min": (TimeFrame.Minutes, 1),
        "mins": (TimeFrame.Minutes, 1),
        "hour": (TimeFrame.Minutes, 60),  # 1 hour = 60-minute compression
        "hours": (TimeFrame.Minutes, 60),
        "day": (TimeFrame.Days, 1),
        "W": (TimeFrame.Weeks, 1),
        "M": (TimeFrame.Months, 1),
    }

    # Maps IB duration dimension characters to Backtrader TimeFrame constants.
    _dur2tf = {
        "S": TimeFrame.Seconds,
        "D": TimeFrame.Days,
        "W": TimeFrame.Weeks,
        "M": TimeFrame.Months,
        "Y": TimeFrame.Years,
    }

    def getdurations(self, timeframe, compression):
        """Return the list of valid IB duration strings for a bar size.

        Uses the inverted ``self.revdur`` table built in ``__init__``.

        Args:
            timeframe: Backtrader ``TimeFrame`` constant.
            compression (int): Backtrader bar compression.

        Returns:
            list of str: Valid IB duration strings (e.g. ``['60 S', '120 S',
            …, '1 Y']``), sorted from shortest to longest.  Empty list if
            the timeframe/compression combination is not supported.
        """
        key = (timeframe, compression)
        if key not in self.revdur:
            return []
        return self.revdur[key]

    def getmaxduration(self, timeframe, compression):
        """Return the longest valid IB duration for a bar size.

        Convenience wrapper over ``getdurations`` that returns only the last
        (longest) element.

        Args:
            timeframe: Backtrader ``TimeFrame`` constant.
            compression (int): Backtrader bar compression.

        Returns:
            str or None: Longest duration string (e.g. ``'1 Y'``), or
            ``None`` if the combination is not supported.
        """
        key = (timeframe, compression)
        try:
            return self.revdur[key][-1]
        except (KeyError, IndexError):
            pass
        return None

    def tfcomp_to_size(self, timeframe, compression):
        """Convert a Backtrader timeframe/compression pair to an IB bar size.

        The IB API uses strings like ``'5 mins'``, ``'1 hour'``, ``'1 day'``
        etc.  This method derives the correct string from the Backtrader
        parameters.

        Special cases:
        - Compression that is a multiple of 7 days is expressed as weeks.
        - Compression that is a multiple of 60 minutes is expressed as hours.

        Args:
            timeframe: Backtrader ``TimeFrame`` constant.
            compression (int): Backtrader bar compression (number of units).

        Returns:
            str or None: IB bar size string, or ``None`` for unsupported
            timeframes (Microseconds, Ticks).
        """
        if timeframe == TimeFrame.Months:
            return f"{compression} M"

        if timeframe == TimeFrame.Weeks:
            return f"{compression} W"

        if timeframe == TimeFrame.Days:
            if not compression % 7:
                # Compress whole weeks: 7 days → '1 W', 14 days → '2 W', etc.
                return f"{compression // 7} W"
            return f"{compression} day"

        if timeframe == TimeFrame.Minutes:
            if not compression % 60:
                # Express as hours: 60 mins → '1 hour', 120 mins → '2 hours'.
                hours = compression // 60
                return (f"{hours} hour") + ("s" * (hours > 1))
            return (f"{compression} min") + ("s" * (compression > 1))

        if timeframe == TimeFrame.Seconds:
            return f"{compression} secs"

        return None  # Microseconds or Ticks — not supported

    def dt_plus_duration(self, dt, duration):
        """Add an IB duration string to a ``datetime``.

        Used by ``reqHistoricalDataEx`` to check whether a given duration
        covers the full requested range.

        Args:
            dt (datetime): Start datetime.
            duration (str): IB duration string, e.g. ``'3600 S'``, ``'1 D'``,
                ``'2 W'``, ``'6 M'``, ``'1 Y'``.

        Returns:
            datetime: ``dt`` advanced by the given duration.
        """
        size, dim = duration.split()
        size = int(size)

        if dim == "S":
            return dt + timedelta(seconds=size)
        if dim == "D":
            return dt + timedelta(days=size)
        if dim == "W":
            return dt + timedelta(days=size * 7)
        if dim == "M":
            # Month arithmetic: handle year rollovers correctly.
            month = dt.month - 1 + size  # shift to 0-based
            years, month = divmod(month, 12)
            return dt.replace(year=dt.year + years, month=month + 1)
        if dim == "Y":
            return dt.replace(year=dt.year + size)

        return dt  # unknown dimension — return unchanged

    # -------------------------------------------------------------------------
    # Commented-out histduration method (retained for reference)
    # -------------------------------------------------------------------------
    # def histduration(self, dt1, dt2):
    #     # Given two dates, compute the smallest IB duration string that covers
    #     # the range.  This was the original approach before the _durations
    #     # table was introduced.  It is superseded by reqHistoricalDataEx but
    #     # kept here as a reference implementation.
    #     ...

    def makecontract(
        self, symbol, sectype, exch, curr, expiry="", strike=0.0, right="", mult=1
    ):
        """Construct an ``ib_insync.Contract`` from individual components.

        Does not validate the parameters against IB — that is done later by
        ``getContractDetails``.

        Args:
            symbol (str): Ticker symbol (e.g. ``'AAPL'``, ``'EUR'``).
            sectype (str): IB security type (``'STK'``, ``'CASH'``, ``'FUT'``,
                ``'OPT'``, ``'FOP'``, ``'IND'``, ``'CFD'``).
            exch (str): Exchange (e.g. ``'SMART'``, ``'CME'``, ``'IDEALPRO'``).
            curr (str): Currency (e.g. ``'USD'``, ``'EUR'``).
            expiry (str): Expiry date string: ``'YYYYMM'`` for futures,
                ``'YYYYMMDD'`` for options.  Empty for non-expiring instruments.
            strike (float): Option strike price.
            right (str): Option right: ``'C'`` (call) or ``'P'`` (put).
            mult (int or str): Contract multiplier (e.g. ``50`` for ES futures).

        Returns:
            ib_insync.Contract: Partially-specified IB contract.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sectype
        contract.exchange = exch
        if curr:
            contract.currency = curr
        if sectype in ["FUT", "OPT", "FOP"]:
            contract.lastTradeDateOrContractMonth = expiry
        if sectype in ["OPT", "FOP"]:
            contract.strike = strike
            contract.right = right
        if mult:
            contract.multiplier = mult
        return contract

    def cancelOrder(self, order):
        """Send a cancel request to TWS.

        ``ib_async.IB.cancelOrder`` expects an order object, but parts of the
        codebase still pass a raw integer order ID.  Accept both for
        compatibility and resolve IDs to the live order object when possible.

        Args:
            order: An order-like object with ``orderId`` or an ``int`` order ID.
        """
        if hasattr(order, "orderId"):
            self.ib.cancelOrder(order)
            return

        orderid = int(order)

        # Resolve an order id to the corresponding live order object.
        # ib_async keeps these in Trade wrappers.
        for trade in self.ib.openTrades():
            ib_order = getattr(trade, "order", None)
            if ib_order is not None and getattr(ib_order, "orderId", None) == orderid:
                self.ib.cancelOrder(ib_order)
                return

        logger.warning(
            "cancelOrder could not resolve order id %s in open trades; "
            "sending raw client cancel as fallback",
            orderid,
        )
        self.ib.client.cancelOrder(orderid, "")

    def placeOrder(self, orderid, contract, order):
        """Place an order via TWS and return the trade handle immediately.

        Args:
            orderid (int): IB order ID (must match ``order.orderId``).
            contract (ib_insync.Contract): The trading contract.
            order (IBOrder): The order object to submit.

        Returns:
            ib_insync.Trade: The trade object with the final order status.
        """
        return self.ib.placeOrder(contract, order)

    def reqTrades(self):
        """Return the list of currently open and recently completed trades.

        Returns:
            list of ib_insync.Trade: All known trades for this session.
        """
        return self.ib.openTrades()

    def reqPositions(self):
        """Fetch current open positions from IB.

        Returns:
            list of ib_insync.Position: Open positions for all accounts.
        """
        return self.ib.positions()

    def getposition(self, contract, clone=False):
        """Return the Backtrader ``Position`` for a given IB contract.

        Positions are keyed by IB ``conId`` (numeric contract identifier).
        A clone is returned by default to avoid external mutation.

        Args:
            contract (ib_insync.Contract): Contract whose position to look up.
                Must have a valid ``conId`` (i.e. be a resolved contract).
            clone (bool): If ``True``, return a copy (default ``False``).

        Returns:
            backtrader.Position: Current size and average cost basis.
        """
        position = self.positions[contract.conId]
        if clone:
            return copy(position)
        return position

    def _resolve_account(self, account=None):
        if account is not None:
            return account

        if not self.managed_accounts:
            self.managedAccounts()

        if not self.managed_accounts:
            return None

        if len(self.managed_accounts) > 1:
            return "__ALL__"

        return self.managed_accounts[0]

    def reqAccountUpdates(self, account=None):
        """Fetch account data from IB and populate the account dictionaries.

        If ``account`` is ``None``, uses the configured account selection rules.

        Args:
            account (str, optional): Specific account to fetch.
        """
        logger.info("[ibstore-checkpoint] reqAccountUpdates:start account=%s", account)
        resolved_account = self._resolve_account(account)
        if resolved_account is None:
            logger.warning("[ibstore-checkpoint] reqAccountUpdates:no_managed_accounts")
            return

        logger.info(
            "[ibstore-checkpoint] reqAccountUpdates:using_account=%s",
            resolved_account,
        )

        self.updateAccountValue()
        logger.info("[ibstore-checkpoint] reqAccountUpdates:complete")

    def updateAccountValue(self):
        """Fetch all account values from IB and update internal caches.

        Calls ``ib.accountValues()`` which returns a flat list of
        ``AccountValue`` named tuples with fields:
        ``(account, tag, value, currency)``.

        Caches:
        - ``acc_upds[account][tag][currency]`` — all values (nested dict).
        - ``acc_value[account]`` — NetLiquidation in BASE currency.
        - ``acc_cash[account]`` — TotalCashBalance in BASE currency.
        """
        logger.info("[ibstore-checkpoint] updateAccountValue:request")
        ret = self.ib.accountValues()
        logger.info(
            "[ibstore-checkpoint] updateAccountValue:received rows=%s",
            len(ret),
        )

        for msg in ret:
            try:
                value = float(msg.value)
            except ValueError:
                # Some tags return non-numeric strings (e.g. account name).
                value = msg.value

            # Store the raw value in the nested dict.
            self.acc_upds[msg.account][msg.tag][msg.currency] = value

            if msg.tag == "NetLiquidation":
                # Tracks total portfolio value (cash + positions at market).
                self.acc_value[msg.account] = value
            elif msg.tag == "TotalCashBalance" and msg.currency == "BASE":
                # BASE currency cash balance across all sub-currencies.
                self.acc_cash[msg.account] = value

    def get_acc_values(self, account=None):
        """Return all account value info received from TWS.

        Returns the full nested dict of ``{tag: {currency: value}}`` for the
        given account, or for all accounts if ``account`` is ``None`` and
        multiple accounts exist.

        Args:
            account (str, optional): Account ID.  ``None`` means "first (or
                only) managed account"; returns all accounts if multiple.

        Returns:
            dict: ``{tag: {currency: value}}`` or ``{account: {tag: ...}}``.
        """
        resolved_account = self._resolve_account(account)
        if resolved_account in (None, "__ALL__"):
            return self.acc_upds.copy()

        if resolved_account in self.acc_upds:
            return self.acc_upds[resolved_account].copy()

        return self.acc_upds.copy()

    def _get_account_metric(self, source_map, account=None):
        resolved_account = self._resolve_account(account)

        if resolved_account is None:
            return 0.0

        if resolved_account == "__ALL__":
            return sum(source_map.values())

        if resolved_account in source_map:
            return source_map[resolved_account]

        return 0.0

    def get_acc_value(self, account=None):
        """Return the net liquidation value for the given account.

        If multiple accounts are managed, returns the sum across all of them.

        Args:
            account (str, optional): Account ID.  ``None`` = first managed
                account (or sum of all if multiple).

        Returns:
            float: Net liquidation value, or ``0.0`` if not yet available.
        """
        return self._get_account_metric(self.acc_value, account)

    def get_acc_cash(self, account=None):
        """Return the total cash balance for the given account.

        If multiple accounts are managed, returns the sum across all of them.

        Args:
            account (str, optional): Account ID.  ``None`` = first managed
                account (or sum of all if multiple).

        Returns:
            float: Total cash balance (BASE currency), or ``0.0`` if not yet
            available.
        """
        return self._get_account_metric(self.acc_cash, account)
