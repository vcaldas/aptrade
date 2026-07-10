#!/usr/bin/env python
###############################################################################
#
# Copyright (C) 2015-2020 Daniel Rodriguez
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
# Source obtained from https://github.com/ultra1971/backtrader_ib_insync

###############################################################################
"""IBData — Backtrader data feed for Interactive Brokers.

This module provides a single public class, ``IBData``, which implements
Backtrader's ``DataBase`` interface and delivers OHLCV bars from IB to a
Cerebro strategy.

Data flow overview
------------------
1. ``IBData.__init__`` parses the ``dataname`` string into a preliminary
   ``ib_insync.Contract`` object (``precontract``).
2. ``IBData.start`` asks ``IBStore`` for full contract details (resolves the
   conId), decides whether to use RTBars or RTVolume for live data, and
   kicks off the first data request via ``_st_start``.
3. ``IBData._load`` is the Backtrader hook called once per bar.  It drives a
   simple finite state machine:

   - **_ST_HISTORBACK** — draining a queue of historical bars that were
     pre-fetched by ``_st_start`` (or backfill-on-reconnect).  When the
     queue empties the feed transitions to ``_ST_LIVE``.
   - **_ST_LIVE** — on every ``_load`` call, issues a fresh ``reqMktData``
     (RTVolume ticks) or ``reqRealTimeBars`` (5-second bars) request and
     processes one message from the result queue.
   - **_ST_FROM** — draining an external ``backfill_from`` data source before
     switching to IB historical/live.
   - **_ST_OVER** — terminal state, returns ``False`` to signal end-of-data.

Live data modes
---------------
``rtbar=False`` (default)
    Tick data via ``reqMktData`` tick type 233 (RTVolume for equities) or
    plain BBO ticks for CASH/CFD.  Each tick populates a complete bar with
    open = high = low = close = tick price.

``rtbar=True``
    5-second real-time bars via ``reqRealTimeBars``.  Only available for
    timeframe/compression ≥ Seconds/5.  Uses MIDPOINT pricing.
"""

import datetime
import time

import aptrade as bt
from aptrade import TimeFrame, date2num, num2date
from aptrade.feed import DataBase
from aptrade.stores import IBStore


class MetaIBData(DataBase.__class__):
    """Metaclass that auto-registers ``IBData`` with ``IBStore``.

    When Python finishes creating the ``IBData`` class, this metaclass fires
    and stores a reference to the class in ``IBStore.DataCls``.  This lets
    ``IBStore.getdata()`` instantiate the data feed without a circular import.
    """

    def __init__(cls, name, bases, dct):
        """Register the newly created data class with the store."""
        super().__init__(name, bases, dct)
        IBStore.DataCls = cls


class IBData(DataBase, metaclass=MetaIBData):
    """Interactive Brokers Data Feed.

    Supports the following contract specifications in parameter ``dataname``:

          - TICKER  # Stock type and SMART exchange
          - TICKER-STK  # Stock and SMART exchange
          - TICKER-STK-EXCHANGE  # Stock
          - TICKER-STK-EXCHANGE-CURRENCY  # Stock

          - TICKER-CFD  # CFD and SMART exchange
          - TICKER-CFD-EXCHANGE  # CFD
          - TICKER-CDF-EXCHANGE-CURRENCY  # Stock

          - TICKER-IND-EXCHANGE  # Index
          - TICKER-IND-EXCHANGE-CURRENCY  # Index

          - TICKER-YYYYMM-EXCHANGE  # Future
          - TICKER-YYYYMM-EXCHANGE-CURRENCY  # Future
          - TICKER-YYYYMM-EXCHANGE-CURRENCY-MULT  # Future
          - TICKER-FUT-EXCHANGE-CURRENCY-YYYYMM-MULT # Future

          - TICKER-YYYYMM-EXCHANGE-CURRENCY-STRIKE-RIGHT  # FOP
          - TICKER-YYYYMM-EXCHANGE-CURRENCY-STRIKE-RIGHT-MULT  # FOP
          - TICKER-FOP-EXCHANGE-CURRENCY-YYYYMM-STRIKE-RIGHT # FOP
          - TICKER-FOP-EXCHANGE-CURRENCY-YYYYMM-STRIKE-RIGHT-MULT # FOP

          - CUR1.CUR2-CASH-IDEALPRO  # Forex

          - TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT  # OPT
          - TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT-MULT  # OPT
          - TICKER-OPT-EXCHANGE-CURRENCY-YYYYMMDD-STRIKE-RIGHT # OPT
          - TICKER-OPT-EXCHANGE-CURRENCY-YYYYMMDD-STRIKE-RIGHT-MULT # OPT

    Params:

      - ``sectype`` (default: ``STK``)

        Default value to apply as *security type* if not provided in the
        ``dataname`` specification

      - ``exchange`` (default: ``SMART``)

        Default value to apply as *exchange* if not provided in the
        ``dataname`` specification

      - ``currency`` (default: ``''``)

        Default value to apply as *currency* if not provided in the
        ``dataname`` specification

      - ``historical`` (default: ``False``)

        If set to ``True`` the data feed will stop after doing the first
        download of data.

        The standard data feed parameters ``fromdate`` and ``todate`` will be
        used as reference.

        The data feed will make multiple requests if the requested duration is
        larger than the one allowed by IB given the timeframe/compression
        chosen for the data.

      - ``what`` (default: ``None``)

        If ``None`` the default for different assets types will be used for
        historical data requests:

          - 'BID' for CASH assets
          - 'TRADES' for any other

        Use 'ASK' for the Ask quote of cash assets

        Check the IB API docs if another value is wished

      - ``rtbar`` (default: ``False``)

        If ``True`` the ``5 Seconds Realtime bars`` provided by Interactive
        Brokers will be used as the smallest tick. According to the
        documentation they correspond to real-time values (once collated and
        curated by IB)

        If ``False`` then the ``RTVolume`` prices will be used, which are based
        on receiving ticks. In the case of ``CASH`` assets (like for example
        EUR.JPY) ``RTVolume`` will always be used and from it the ``bid`` price
        (industry de-facto standard with IB according to the literature
        scattered over the Internet)

        Even if set to ``True``, if the data is resampled/kept to a
        timeframe/compression below Seconds/5, no real time bars will be used,
        because IB doesn't serve them below that level

      - ``qcheck`` (default: ``0.5``)

        Time in seconds to wake up if no data is received to give a chance to
        resample/replay packets properly and pass notifications up the chain

      - ``backfill_start`` (default: ``True``)

        Perform backfilling at the start. The maximum possible historical data
        will be fetched in a single request.

      - ``backfill`` (default: ``True``)

        Perform backfilling after a disconnection/reconnection cycle. The gap
        duration will be used to download the smallest possible amount of data

      - ``backfill_from`` (default: ``None``)

        An additional data source can be passed to do an initial layer of
        backfilling. Once the data source is depleted and if requested,
        backfilling from IB will take place. This is ideally meant to backfill
        from already stored sources like a file on disk, but not limited to.

      - ``latethrough`` (default: ``False``)

        If the data source is resampled/replayed, some ticks may come in too
        late for the already delivered resampled/replayed bar. If this is
        ``True`` those ticks will be let through in any case.

        Check the Resampler documentation to see how to take those ticks into
        account.

        This can happen especially if ``timeoffset`` is set to ``False`` in
        the ``IBStore`` instance and the TWS server time is not in sync with
        that of the local computer

      - ``tradename`` (default: ``None``)

        Useful for some specific cases like ``CFD`` in which prices are offered
        by one asset and trading happens in a different one:

        - SPY-STK-SMART-USD -> SP500 ETF (will be specified as ``dataname``)
        - SPY-CFD-SMART-USD -> which is the corresponding CFD which offers not
          price tracking but in this case will be the trading asset (specified
          as ``tradename``)

      - ``hist_tzo`` (default: ``0``)

        Timezone offset in hours to apply to historical bar timestamps.
        Backtrader expects UTC; if the IB server returns bars in local time,
        set this to the server's UTC offset (e.g. ``-5`` for US Eastern
        Standard Time) so bars are shifted back to UTC.

    The default values allow a bare ``TICKER`` string, to which ``sectype``
    (default: ``STK``) and ``exchange`` (default: ``SMART``) are applied.

    Some assets like ``AAPL`` need an explicit currency::

        IBData(dataname='AAPL', currency='USD')
        # equivalent to dataname='AAPL-STK-SMART-USD'
    """

    params = (
        ("sectype", "STK"),  # security type default
        ("exchange", "SMART"),  # exchange default
        ("currency", ""),  # currency default (empty = IB default)
        ("rtbar", False),  # True = use 5-second RTBars; False = RTVolume ticks
        ("historical", False),  # True = download only, then stop
        ("what", None),  # what to show: TRADES, BID, ASK, MIDPOINT, …
        ("useRTH", False),  # True = regular trading hours only
        ("qcheck", 0.5),  # seconds to wait for queue data before yielding
        ("backfill_start", True),  # backfill at startup
        ("backfill", True),  # backfill after reconnection
        ("backfill_from", None),  # alternative data source for initial backfill
        ("latethrough", False),  # allow late ticks through the resampler
        ("tradename", None),  # separate contract for order placement
        ("hist_tzo", 0),  # timezone offset (hours) for historical bars
    )

    _store = IBStore

    # The minimum timeframe/compression supported by IB real-time bars.
    # Anything finer must use RTVolume ticks instead.
    RTBAR_MINSIZE = (TimeFrame.Seconds, 5)

    # States for the finite state machine (FSM) inside ``_load``.
    # The integer values are assigned left-to-right by range(5).
    _ST_FROM = 0  # consuming an external backfill_from data source
    _ST_START = 1  # initial state — about to call _st_start()
    _ST_LIVE = 2  # live data mode (reqMktData or reqRealTimeBars)
    _ST_HISTORBACK = 3  # draining a historical data queue
    _ST_OVER = 4  # terminal — feed is finished

    def _gettz(self):
        """Resolve the timezone to use for bar timestamps.

        Priority:
        1. If ``self.p.tz`` is a non-string object (e.g. a pytz timezone),
           wrap it in a Backtrader ``Localizer`` and return it.
        2. If ``self.p.tz`` is a string, treat it as a pytz timezone name.
        3. Otherwise fall back to the ``timeZoneId`` reported in the IB
           contract details.

        Returns:
            A pytz timezone object, a Backtrader Localizer, or ``None`` if
            the timezone cannot be determined (pytz not installed, unknown
            zone, no contract details yet).
        """
        tzstr = isinstance(self.p.tz, str)
        if self.p.tz is not None and not tzstr:
            # Caller passed a tzinfo object directly — wrap it.
            return bt.utils.date.Localizer(self.p.tz)

        if self.contractdetails is None:
            return None  # contract details not yet available

        try:
            import pytz  # optional dependency — keep the import local
        except ImportError:
            return None  # pytz not installed

        # Use the explicit tz string if provided, otherwise use the IB zone.
        tzs = self.p.tz if tzstr else self.contractdetails.timeZoneId

        if tzs == "CST":
            # TWS reports 'CST' but pytz only knows 'CST6CDT' for US Central.
            tzs = "CST6CDT"

        try:
            tz = pytz.timezone(tzs)
        except pytz.UnknownTimeZoneError:
            return None

        return tz

    def islive(self):
        """Tell Cerebro whether this feed delivers live data.

        Returns ``True`` (live) when ``historical=False``, which causes
        Cerebro to disable preloading and ``runonce`` mode so that the
        strategy's ``next()`` is called bar-by-bar in real time.
        """
        return not self.p.historical

    def __init__(self, **kwargs):
        """Initialise the data feed and parse the contract string.

        Stores a reference to the ``IBStore`` singleton and pre-parses both
        ``dataname`` and ``tradename`` into preliminary ``Contract`` objects.
        Full contract resolution (conId, etc.) happens in ``start()`` once the
        IB connection is available.

        Args:
            **kwargs: Forwarded to ``IBStore`` and to the ``DataBase`` params
                system.  Includes ``dataname``, ``tradename``, ``sectype``,
                ``exchange``, ``currency``, ``rtbar``, ``historical``, etc.
        """
        self.ibstore = self._store(**kwargs)
        # Parse the dataname/tradename strings into skeletal Contract objects.
        # These will be resolved against the live IB API in start().
        self.precontract = self.parsecontract(self.p.dataname)
        self.pretradecontract = self.parsecontract(self.p.tradename)

    def setenvironment(self, env):
        """Register the shared ``IBStore`` with the Cerebro environment.

        Called by Cerebro when the data feed is added.  Ensures the store is
        known to Cerebro so that ``start()`` / ``stop()`` lifecycle calls reach
        it correctly.

        Args:
            env: The Cerebro instance (Backtrader environment).
        """
        super().setenvironment(env)
        env.addstore(self.ibstore)

    def parsecontract(self, dataname):
        """Parse a ``dataname`` string into a preliminary IB ``Contract``.

        The dataname format is a dash-separated string.  Supported formats::

            TICKER                                    → STK, SMART exchange
            TICKER-STK-EXCHANGE-CURRENCY              → stock
            CUR1.CUR2-CASH-IDEALPRO                   → forex
            TICKER-YYYYMM-EXCHANGE-CURRENCY-MULT      → future
            TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT-MULT → option
            TICKER-FUT-EXCHANGE-CURRENCY-YYYYMM-MULT  → future (explicit)
            TICKER-FOP-EXCHANGE-CURRENCY-YYYYMM-STRIKE-RIGHT-MULT → FOP
            TICKER-OPT-EXCHANGE-CURRENCY-YYYYMMDD-STRIKE-RIGHT-MULT → option

        The returned ``Contract`` is "pre-resolved" — it has enough fields for
        ``IBStore.getContractDetails()`` to look up the full IB contract
        (including the numeric ``conId``).

        Args:
            dataname (str or None): Contract identifier string, or ``None``.

        Returns:
            ib_insync.Contract or None: Skeletal contract, or ``None`` if
            ``dataname`` is ``None``.
        """
        if dataname is None:
            return None

        # Start with the defaults from params; override as tokens are parsed.
        exch = self.p.exchange
        curr = self.p.currency
        expiry = ""
        strike = 0.0
        right = ""
        mult = ""

        # Split the dash-separated string into a token iterator.
        tokens = iter(dataname.split("-"))

        # First token is always the symbol (e.g. 'AAPL', 'EUR.USD').
        symbol = next(tokens)
        try:
            sectype = next(tokens)
        except StopIteration:
            # Only a bare ticker was given — use the default security type.
            sectype = self.p.sectype

        # A pure-digit second token means an expiry date was provided instead
        # of an explicit security type — infer the type from the digit count.
        if sectype.isdigit():
            expiry = sectype  # save the expiry string
            if len(sectype) == 6:  # YYYYMM format → future
                sectype = "FUT"
            else:  # YYYYMMDD format → option
                sectype = "OPT"

        if sectype == "CASH":
            # Forex pairs are encoded as 'BASE.QUOTE'; split them apart.
            symbol, curr = symbol.split(".")

        try:
            exch = next(tokens)  # exchange (e.g. 'SMART', 'CME')
            curr = next(tokens)  # currency (e.g. 'USD', 'EUR')

            if sectype == "FUT":
                if not expiry:
                    expiry = next(tokens)  # YYYYMM expiry
                mult = next(tokens)  # contract multiplier

                # If a 'right' token (C/P) follows the multiplier, this is
                # actually a Futures on Options (FOP) contract, not a future.
                right = next(tokens)
                sectype = "FOP"
                strike, mult = float(mult), ""  # mult was actually the strike
                mult = next(tokens)  # try to read the real multiplier

            elif sectype == "OPT":
                if not expiry:
                    expiry = next(tokens)  # YYYYMMDD expiry
                strike = float(next(tokens))  # strike price
                right = next(tokens)  # 'C' or 'P'
                mult = next(tokens)  # contract multiplier

        except StopIteration:
            # Not all optional tokens were present — use whatever was parsed.
            pass

        # Build the ib_insync Contract object from the parsed components.
        precon = self.ibstore.makecontract(
            symbol=symbol,
            sectype=sectype,
            exch=exch,
            curr=curr,
            expiry=expiry,
            strike=strike,
            right=right,
            mult=mult,
        )

        return precon

    def start(self):
        """Connect to IB, resolve the contract, and initiate data requests.

        Called by Cerebro before the first bar.  Steps:

        1. Kicks off the ``IBStore`` (creates the data queue).
        2. Decides whether to use RTVolume (``_usertvol=True``) or RTBars.
        3. Resolves ``precontract`` and ``pretradecontract`` to full IB
           contracts with ``conId`` via ``IBStore.getContractDetails()``.
        4. Calls ``_st_start()`` to issue the first historical / live request.

        Sends ``CONNECTED`` on success or ``DISCONNECTED`` if the contract
        cannot be resolved.
        """
        super().start()

        # Register with the store and get the initial queue.
        self.q = self.ibstore.start(data=self)

        # Decide live data mode.  RTBars are only supported at Seconds/5+.
        self._usertvol = not self.p.rtbar
        tfcomp = (self._timeframe, self._compression)
        if tfcomp < self.RTBAR_MINSIZE:
            # The requested bar size is too small for RTBars — fall back to
            # RTVolume ticks regardless of the rtbar parameter.
            self._usertvol = True

        # These are set after contract resolution below.
        self.contract = None
        self.contractdetails = None
        self.tradecontract = None
        self.tradecontractdetails = None

        if self.p.backfill_from is not None:
            # An external backfill source was provided — start there.
            self._state = self._ST_FROM
            self.p.backfill_from.setenvironment(self._env)
            self.p.backfill_from._start()
        else:
            self._state = self._ST_START  # normal startup path

        self.put_notification(self.CONNECTED)

        # Resolve the data contract to its full IB representation (gets conId,
        # exchange details, timezone, etc.).
        cds = self.ibstore.getContractDetails(self.precontract, maxcount=1)
        if cds is not None:
            cdetails = cds[0]
            self.contract = cdetails.contract
            self.contractdetails = cdetails
        else:
            # Contract not found or ambiguous.
            self.put_notification(self.DISCONNECTED)
            return

        if self.pretradecontract is None:
            # No separate trading asset — data and trading use the same contract.
            self.tradecontract = self.contract
            self.tradecontractdetails = self.contractdetails
        else:
            # A tradename was given (e.g. SPY-CFD while tracking SPY-STK).
            # Resolve the trading contract separately.
            cds = self.ibstore.getContractDetails(self.pretradecontract, maxcount=1)
            if cds is not None:
                cdetails = cds[0]
                self.tradecontract = cdetails.contract
                self.tradecontractdetails = cdetails
            else:
                self.put_notification(self.DISCONNECTED)
                return

        if self._state == self._ST_START:
            # No external backfill source — proceed to first IB request.
            self._start_finish()
            self._st_start()

    def stop(self):
        """Stop the data feed and disconnect from IB."""
        super().stop()
        self.ibstore.stop()

    def haslivedata(self):
        """Return ``True`` if the data queue exists and is non-empty.

        Used by Cerebro to decide whether to call ``_load`` again immediately
        or to yield control to the event loop.
        """
        return bool(self.q)

    def _load(self):
        """Load the next bar into Backtrader's line buffers.

        This is the core FSM loop called by Cerebro on every bar cycle.

        FSM transitions
        ---------------
        Any state ≠ _ST_LIVE
            Drain the pre-fetched historical queue ``self.q`` one bar at a
            time.  When the queue is empty, transition to ``_ST_LIVE`` and
            notify LIVE.

        _ST_LIVE
            Issue a fresh market data / RTBar request on every call and
            process one message from the result queue.

        Returns:
            True if a bar was successfully loaded into the line buffers.
            False if the feed has ended (``_ST_OVER`` or no contract).
            None (implicit) if no data was available this cycle.
        """
        if self.contract is None or self._state == self._ST_OVER:
            return False  # no contract or explicitly finished

        while True:
            # --- Historical / backfill phase ---
            if self._state != self._ST_LIVE:
                if not self.q.empty():
                    # A bar is waiting in the queue — load it.
                    msg = self.q.get()
                    ret = self._load_rtbar(msg, hist=True, hist_tzo=self.p.hist_tzo)
                    if ret:
                        return True
                else:
                    # Queue exhausted — switch to live mode.
                    self.put_notification(self.LIVE)
                    self._state = self._ST_LIVE

            # --- Live data phase ---
            if self._state == self._ST_LIVE:
                # Issue a fresh data request each cycle.
                if self._usertvol:
                    # RTVolume / tick data (default for most instruments).
                    self.q = self.ibstore.reqMktData(self.contract, self.p.what)
                else:
                    # 5-second real-time bars.
                    self.q = self.ibstore.reqRealTimeBars(self.contract)

                if not self.q.empty():
                    msg = self.q.get()

                    if self._usertvol:
                        # Tick-based bar: open=high=low=close=tick price.
                        ret = self._load_rtvolume(msg)
                    else:
                        # OHLCV bar from reqRealTimeBars.
                        ret = self._load_rtbar(
                            msg, hist=False, hist_tzo=self.p.hist_tzo
                        )
                    if ret:
                        return True

    def _st_start(self):
        """Issue the first data request after the contract is resolved.

        Two paths depending on ``self.p.historical``:

        Historical mode (``historical=True``)
            Sends a ``reqHistoricalDataEx`` request spanning ``fromdate`` →
            ``todate``, puts the feed in ``_ST_HISTORBACK``, and returns.
            ``_load`` will drain the result queue bar by bar; the feed ends
            when the queue is empty.

        Live mode (``historical=False``)
            If ``backfill_start=True``, sends a historical request first to
            fill the gap from the beginning of data to now, then transitions
            to ``_ST_LIVE`` once that queue is drained.
            If ``backfill_start=False``, goes straight to ``_ST_LIVE``.

        Returns:
            True in all cases (signals to the caller to continue the loop).
        """
        if self.p.historical:
            # Historical-only mode.
            self.put_notification(self.DELAYED)

            dtend = None
            if self.todate < float("inf"):
                dtend = num2date(self.todate)

            dtbegin = None
            if self.fromdate > float("-inf"):
                dtbegin = num2date(self.fromdate)

            # Request the full historical range.  IBStore.reqHistoricalDataEx
            # automatically splits it into multiple IB requests if needed.
            self.q = self.ibstore.reqHistoricalDataEx(
                contract=self.contract,
                enddate=dtend,
                begindate=dtbegin,
                timeframe=self._timeframe,
                compression=self._compression,
                what=self.p.what,
                useRTH=self.p.useRTH,
                tz=self._tz,
                sessionend=self.p.sessionend,
            )

            self._state = self._ST_HISTORBACK
            return True

        else:
            # Live mode — optionally backfill first.
            self._statelivereconn = self.p.backfill_start
            if self.p.backfill_start:
                dtend = None
                if self.p.backfill_from is not None:
                    # Backfill from the end of the external source.
                    dtbegin = num2date(self.p.backfill_from)
                    self._state = self._ST_FROM
                else:
                    dtbegin = None  # fetch max available history
                    self._state = self._ST_START

                # Fetch historical bars to backfill up to the current moment.
                self.q = self.ibstore.reqHistoricalDataEx(
                    contract=self.contract,
                    enddate=dtend,
                    begindate=dtbegin,
                    timeframe=self._timeframe,
                    compression=self._compression,
                    what=self.p.what,
                    useRTH=self.p.useRTH,
                    tz=self._tz,
                    sessionend=self.p.sessionend,
                )
                self.put_notification(self.DELAYED)
            else:
                # Skip backfill — go live immediately.
                self._state = self._ST_LIVE

        return True

    def _load_rtbar(self, rtbar, hist=False, hist_tzo=None):
        """Load a single OHLCV bar (RTBar or historical bar) into line buffers.

        IB historical bars use a ``date`` attribute while live RTBars use
        ``time``.  Both provide open, high, low, close, and volume.

        Args:
            rtbar: An ``ib_insync.BarData`` object (historical or RTBar).
            hist (bool): ``True`` for historical bars (use ``rtbar.date``),
                ``False`` for live RTBars (use ``rtbar.time``).
            hist_tzo: Timezone offset in hours for historical bars, or
                ``None`` to auto-detect from the system timezone.

        Returns:
            True if the bar was loaded successfully.
            False if the bar is older than the last delivered bar and
            ``latethrough`` is False (i.e. the bar is discarded).
        """
        if hist:
            if hist_tzo is None:
                # Fall back to the system local UTC offset (seconds → hours).
                hist_tzo = time.timezone / 3600
                rtbar.date = rtbar.date + datetime.timedelta(hours=hist_tzo)
            dt = date2num(rtbar.date)
        else:
            dt = date2num(rtbar.time)

        # Reject out-of-order bars unless latethrough is enabled.
        if dt < self.lines.datetime[-1] and not self.p.latethrough:
            return False

        self.lines.datetime[0] = dt

        # Populate the OHLCV lines.  ``open`` is a Python keyword in some
        # contexts, so ib_insync may expose it as ``open_``.
        try:
            self.lines.open[0] = rtbar.open
        except AttributeError:
            self.lines.open[0] = rtbar.open_  # ib_insync ≥ 0.9.70 uses open_
        self.lines.high[0] = rtbar.high
        self.lines.low[0] = rtbar.low
        self.lines.close[0] = rtbar.close
        self.lines.volume[0] = rtbar.volume
        self.lines.openinterest[0] = 0  # IB does not provide OI via RTBars

        return True

    def _load_rtvolume(self, rtvol):
        """Load a single RTVolume tick into line buffers as a flat bar.

        Because a tick is a single price point, open = high = low = close =
        tick price.  Volume is the size reported with the tick.

        Args:
            rtvol: An ``ib_insync.Ticker`` object (from ``reqMktData``).
                Expected to have ``time`` and ``price`` and ``size``
                attributes from the latest tick.

        Returns:
            True if the tick was loaded successfully.
            False if the tick is older than the last bar and ``latethrough``
            is False.
        """
        dt = date2num(rtvol.time)
        if dt < self.lines.datetime[-1] and not self.p.latethrough:
            return False  # discard late tick

        self.lines.datetime[0] = dt

        # All OHLC values are set to the single tick price.
        tick = rtvol.price
        self.lines.open[0] = tick
        self.lines.high[0] = tick
        self.lines.low[0] = tick
        self.lines.close[0] = tick
        self.lines.volume[0] = rtvol.size
        self.lines.openinterest[0] = 0

        return True
