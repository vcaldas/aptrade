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
"""IBBroker — Backtrader broker implementation for Interactive Brokers.

This module contains three public classes:

``IBOrder``
    A dual-inheritance class that simultaneously satisfies Backtrader's
    ``OrderBase`` interface and ``ib_async``'s ``order.Order`` structure.
    It translates Backtrader order parameters (exec type, price, TIF, …) into
    the fields that ib_async expects to send to TWS.

``IBCommInfo``
    A minimal ``CommInfoBase`` subclass used to attach commission information
    to orders so that Backtrader's internal trade accounting keeps working.
    Actual commissions are reported by IB and are not recalculated here.

``IBBroker``
    Implements ``BrokerBase``.  Delegates connection management to the shared
    ``IBStore`` singleton and polls ``ib_insync`` trade objects on every
    Cerebro ``next()`` tick to drive the Backtrader order state machine.
"""

import collections
import logging
import threading
import uuid
from datetime import date, datetime, timedelta

import ib_async

from aptrade.broker import BrokerBase
from aptrade.comminfo import CommInfoBase
from aptrade.order import Order, OrderBase
from aptrade.position import Position
from aptrade.stores.ibstore import IBStore
from aptrade.utils.dateintern import num2date
from aptrade.utils.py3 import queue

logger = logging.getLogger(__name__)


class IBOrder(OrderBase, ib_async.order.Order):
    """Dual-base order class that bridges Backtrader and ib_async.

    Inherits from both ``backtrader.OrderBase`` (which carries the strategy
    ref, size, price, exec type, validity, etc.) and from
    ``ib_async.order.Order`` (which carries the IB-specific fields TWS needs:
    ``orderType``, ``lmtPrice``, ``auxPrice``, ``tif``, etc.).

    ``OrderBase.__init__`` runs first via ``super()`` and populates all
    Backtrader-side attributes.  The body of ``__init__`` then maps those
    attributes onto the corresponding IB fields.

    Extra IB-only parameters (e.g. ``orderType='LIT'``) can be passed as
    **kwargs and are applied directly to the IB order object::

        order = IBOrder('BUY', orderType='LIT', lmtPrice=10.0, auxPrice=9.8)

    This produces a *Limit If Touched* order regardless of the Backtrader
    exec-type mapping — useful for IB order types not natively supported by
    Backtrader.
    """

    def __str__(self):
        """Return a human-readable summary combining Backtrader and IB fields."""
        basetxt = super().__str__()
        tojoin = [basetxt]
        tojoin.append(f"Ref: {self.ref}")
        tojoin.append(f"orderId: {self.orderId}")
        tojoin.append(f"Action: {self.action}")
        tojoin.append(f"Size (ib): {self.totalQuantity}")
        tojoin.append(f"Lmt Price: {self.lmtPrice}")
        tojoin.append(f"Aux Price: {self.auxPrice}")
        tojoin.append(f"OrderType: {self.orderType}")
        tojoin.append(f"Tif (Time in Force): {self.tif}")
        tojoin.append(f"GoodTillDate: {self.goodTillDate}")
        return "\n".join(tojoin)

    # Mapping from Backtrader execution-type constants to IB order type strings.
    # None (unspecified) defaults to a plain market order.
    _IBOrdTypes = {
        None: "MKT",  # default — plain market order
        Order.Market: "MKT",  # market order
        Order.Limit: "LMT",  # limit order
        Order.Close: "MOC",  # market-on-close
        Order.Stop: "STP",  # stop order (becomes market when triggered)
        Order.StopLimit: "STPLMT",  # stop-limit order
        Order.StopTrail: "TRAIL",  # trailing stop
        Order.StopTrailLimit: "TRAIL LIMIT",  # trailing stop-limit
    }

    def __init__(self, action, **kwargs):
        """Initialise the order, map Backtrader fields to IB fields.

        Args:
            action (str): ``'BUY'`` or ``'SELL'``.
            **kwargs: Passed through to ``OrderBase`` (e.g. ``size``,
                ``price``, ``pricelimit``, ``exectype``, ``valid``,
                ``tradeid``, ``clientId``, ``orderId``) and any extra
                IB-specific attributes applied directly to the order object.
        """
        # Flag set to True when an openOrder with PendingCancel/Cancelled is
        # received — this signals an order expiry rather than a user cancel.
        self._willexpire = False

        # Tell OrderBase whether this is a buy or sell so it sets ordtype.
        self.ordtype = self.Buy if action == "BUY" else self.Sell

        # Let OrderBase process size, price, exectype, valid, etc.
        super().__init__()
        # Initialise the ib_async Order fields to their defaults.
        ib_async.order.Order.__init__(self)

        # --- Map execution type to IB order type string ---
        self.orderType = self._IBOrdTypes[self.exectype]

        self.permid = 0  # IB permanent order ID (assigned by TWS after submission)

        # IB uses 'BUY' / 'SELL' strings, not integers.
        self.action = action

        # Price fields: lmtPrice is the limit price; auxPrice is the stop /
        # trail-amount price.  Both default to 0.0 (unused).
        self.lmtPrice = 0.0
        self.auxPrice = 0.0

        if self.exectype == self.Market:
            # Market order: no prices needed.
            pass
        elif self.exectype == self.Close:
            # Market-on-Close: no explicit price.
            pass
        elif self.exectype == self.Limit:
            # Limit order: execute at lmtPrice or better.
            self.lmtPrice = self.price
        elif self.exectype == self.Stop:
            # Stop order: triggers a market order when auxPrice is touched.
            self.auxPrice = self.price  # stop trigger price
        elif self.exectype == self.StopLimit:
            # Stop-Limit: triggers a limit order at lmtPrice when auxPrice is
            # touched.  auxPrice = stop trigger; lmtPrice = limit after trigger.
            self.lmtPrice = self.pricelimit  # limit price after trigger
            self.auxPrice = self.price  # stop trigger price
        elif self.exectype == self.StopTrail:
            # Trailing stop: trail by a fixed amount or percentage.
            if self.trailamount is not None:
                self.auxPrice = self.trailamount  # fixed dollar trail
            elif self.trailpercent is not None:
                # IB expects trailingPercent in percentage points (e.g. 2.0
                # for 2 %), but Backtrader uses a fraction (e.g. 0.02).
                self.trailingPercent = self.trailpercent * 100.0
        elif self.exectype == self.StopTrailLimit:
            # Trailing stop-limit: combines a trailing stop with a limit price.
            self.trailStopPrice = self.lmtPrice = self.price
            # The limit offset is expressed relative to the trail stop price.
            self.lmtPrice = self.pricelimit
            if self.trailamount is not None:
                self.auxPrice = self.trailamount
            elif self.trailpercent is not None:
                self.trailingPercent = self.trailpercent * 100.0

        # IB requires a positive quantity; direction is encoded in ``action``.
        self.totalQuantity = abs(self.size)

        self.transmit = self.transmit  # whether to transmit immediately to TWS
        if self.parent is not None:
            # Bracket / attached orders reference their parent by IB order ID.
            self.parentId = self.parent.orderId

        # --- Time-In-Force ---
        # Backtrader ``valid`` field can be:
        #   None          → GTC  (Good Till Cancelled — IB default)
        #   datetime/date → GTD  (Good Till Date)
        #   timedelta     → DAY if equal to Order.DAY, else GTD
        #   0             → DAY
        #   numeric       → GTD  (Backtrader date number)
        if self.valid is None:
            tif = "GTC"  # no expiry specified — good till cancelled
        elif isinstance(self.valid, (datetime, date)):
            tif = "GTD"
            self.m_goodTillDate = self.valid.strftime("%Y%m%d %H:%M:%S")
        elif isinstance(self.valid, (timedelta,)):
            if self.valid == self.DAY:
                tif = "DAY"
            else:
                tif = "GTD"
                valid = datetime.now() + self.valid
                self.goodTillDate = valid.strftime("%Y%m%d %H:%M:%S")
        elif self.valid == 0:
            tif = "DAY"
        else:
            tif = "GTD"
            valid = num2date(self.valid)
            self.goodTillDate = valid.strftime("%Y%m%d %H:%M:%S")

        self.tif = tif

        # OCA (One-Cancels-All) type 1: cancel all remaining orders in the
        # group when one is filled.  The actual ocaGroup UUID is assigned by
        # IBBroker.submit().
        self.ocaType = 1

        # Apply any extra IB-specific kwargs directly to the order object.
        # If the attribute already exists on ib_async.Order, use it directly;
        # otherwise fall back to the legacy 'm_' prefixed name.
        for k in kwargs:
            setattr(self, (not hasattr(self, k)) * "m_" + k, kwargs[k])


class IBCommInfo(CommInfoBase):
    """Minimal commission-info object for IB orders.

    IB calculates actual commissions server-side and reports them back after
    execution.  Backtrader's ``Strategy`` and ``Trade`` objects still need a
    ``CommInfo`` attached to each order so that the internal accounting
    (``getvalue``, ``getoperationcost``) can run without crashing.

    Both methods here return a simple ``abs(size) * price`` approximation.
    The values are close enough for P&L tracking purposes, even though the
    real IB commission structure is more complex (per-share tiered, minimum
    fees, etc.).

    Note: margin is not pre-calculated here (it would require an
    ``OrderState`` round-trip).  ``getvaluesize`` approximates it as full
    notional value.
    """

    def getvaluesize(self, size, price):
        """Return the notional value of a position (size * price).

        Used by Backtrader to estimate margin/value.  For IB, margin is
        determined server-side, so this is an approximation.
        """
        return abs(size) * price

    def getoperationcost(self, size, price):
        """Return the cash cost of opening/closing a position.

        Returns:
            float: ``abs(size) * price`` — the gross value of the trade,
            ignoring commissions (which are added by IB separately).
        """
        return abs(size) * price


class MetaIBBroker(BrokerBase.__class__):
    """Metaclass that auto-registers ``IBBroker`` with ``IBStore``.

    When Python finishes creating the ``IBBroker`` class body, this metaclass
    fires and stores a reference to ``IBBroker`` in ``IBStore.BrokerCls``.
    This allows ``IBStore.getbroker()`` to instantiate the broker without a
    hard import dependency between the two modules.
    """

    def __init__(self, name, bases, dct):
        """Register the newly created broker class with the store."""
        super().__init__(name, bases, dct)
        IBStore.BrokerCls = self


class IBBroker(BrokerBase, metaclass=MetaIBBroker):
    """Backtrader ``BrokerBase`` implementation for Interactive Brokers.

    Responsibilities
    ----------------
    * Delegates the IB connection to the ``IBStore`` singleton.
    * Translates ``buy()`` / ``sell()`` calls into ``IBOrder`` objects and
      submits them via ``IBStore.placeOrder()``.
    * On every Cerebro ``next()`` cycle, polls ``ib_async`` trade objects to
      drive order status transitions: Submitted → Accepted → Completed /
      Cancelled / Rejected.
    * Reports cash and portfolio value from the IB account.

    Limitations
    -----------
    * ``tradeid`` is not accurately supported because P&L comes directly from
      IB (FIFO basis) rather than being recalculated locally.
    * Positions opened outside this session (e.g. from another client or a
      manual trade) are not reflected in Backtrader's internal position
      tracking until a ``reqPositions()`` update is received.
    """

    params = ()

    def __init__(self, **kwargs):
        """Initialise the broker and connect to IBStore.

        Args:
            **kwargs: Forwarded to ``IBStore`` (``host``, ``port``,
                ``clientId``, ``account``, etc.).  See ``IBStore`` params for
                the full list.
        """
        super().__init__()

        # Obtain (or create) the IBStore singleton with the given connection
        # parameters.  All IBData feeds share this same store.
        self.ibstore = IBStore(**kwargs)

        # Cash and portfolio value are fetched lazily from IB on each call to
        # getcash() / getvalue().  Initialise to 0 until start() pulls the
        # real figures.
        self.startingcash = self.cash = 0.0
        self.startingvalue = self.value = 0.0

        # List of orders that have been submitted but not yet reached a
        # terminal state (Filled / Cancelled / Rejected / Expired).
        self.open_orders = []

        # Queue of cloned order objects waiting to be consumed by Cerebro's
        # notify_order() machinery.
        self.notifs = queue.Queue()

        # Deque of order IDs that need to be notified (used internally).
        self.tonotify = collections.deque()

        # Account bootstrap background thread handle.
        self._account_bootstrap_thread = None

    def start(self):
        """Called by Cerebro when the run starts.

        Registers this broker with the store, then fetches the initial account
        cash and portfolio value so that Backtrader's starting balance is
        correct from bar zero.
        """
        super().start()
        logger.info("[ibbroker-checkpoint] start:begin")

        # Register the broker with the store so it can receive order updates.
        logger.info("[ibbroker-checkpoint] start:before_store_start")
        self.ibstore.start(broker=self)
        logger.info("[ibbroker-checkpoint] start:after_store_start")

        # Temporarily zero out while we wait for the IB account data.
        self.startingcash = self.cash = 0.0
        self.startingvalue = self.value = 0.0

        # Run account bootstrap asynchronously so broker.start does not block
        # cerebro strategy instantiation when IB account APIs are slow.
        self._account_bootstrap_thread = threading.Thread(
            target=self._bootstrap_account_snapshot,
            name="ibbroker-account-bootstrap",
            daemon=False,
        )
        self._account_bootstrap_thread.start()
        logger.info("[ibbroker-checkpoint] start:account_bootstrap_dispatched")
        logger.info("[ibbroker-checkpoint] start:complete")

    def _bootstrap_account_snapshot(self):
        try:
            logger.info("[ibbroker-checkpoint] bootstrap:before_req_account_updates")
            self.ibstore.reqAccountUpdates()
            logger.info("[ibbroker-checkpoint] bootstrap:after_req_account_updates")

            self.startingcash = self.cash = self.ibstore.get_acc_cash()
            self.startingvalue = self.value = self.ibstore.get_acc_value()
            logger.info(
                "[ibbroker-checkpoint] bootstrap:account_snapshot cash=%s value=%s",
                self.startingcash,
                self.startingvalue,
            )
        except Exception as exc:
            logger.warning(
                "IB account bootstrap failed in background thread; "
                "continuing with cash/value at 0.0. Error: %s",
                exc,
                exc_info=True,
            )

    def stop(self):
        """Called by Cerebro when the run finishes.  Disconnects from IB."""
        super().stop()
        if (
            self._account_bootstrap_thread is not None
            and self._account_bootstrap_thread.is_alive()
        ):
            self._account_bootstrap_thread.join(timeout=5)
        self.ibstore.stop()

    def getcash(self):
        """Return the current total cash balance from IB.

        Fetches a fresh value from the store on every call.  Non-blocking —
        returns the last known value if no IB response is available yet.
        """
        self.cash = self.ibstore.get_acc_cash()
        return self.cash

    def getvalue(self, datas=None):
        """Return the current net liquidation value of the IB account.

        Args:
            datas: Ignored — IB reports portfolio-level value, not per-data.

        Returns:
            float: Net liquidation value (cash + open positions at market).
        """
        self.value = self.ibstore.get_acc_value()
        return self.value

    def getposition(self, data, clone=True):
        """Return the current position for the given data feed.

        Looks up the position by the IB contract ID of the trading contract
        associated with the data feed.

        Args:
            data: A Backtrader data feed whose ``tradecontract`` attribute
                identifies the IB instrument.
            clone (bool): Return a copy of the position (default ``True``).

        Returns:
            backtrader.Position: Current size and average price.
        """
        contract = getattr(data, "tradecontract", None)
        if contract is None:
            logger.warning(
                "Data feed %s has no tradecontract; returning flat position",
                getattr(data, "_name", "<unnamed>"),
            )
            return Position()

        # Massive live feeds may carry an unresolved contract (conId=0).
        # IBStore positions are keyed by conId, so unresolved contracts cannot
        # be matched reliably. Treat as flat here to avoid cross-symbol bleed
        # from shared conId=0 lookups.
        if not getattr(contract, "conId", 0):
            return Position()

        return self.ibstore.getposition(contract, clone=clone)

    def cancel(self, order):
        """Cancel a live order.

        Args:
            order (IBOrder): The order to cancel.  Its ``orderId`` must match
                a live order at TWS.
        """
        return self.ibstore.cancelOrder(order)

    def orderstatus(self, order):
        """Return the current status string of an order.

        Args:
            order (IBOrder): The order to query.

        Returns:
            str: One of the IB status strings (e.g. ``'Submitted'``).
        """
        return order.status

    def submit(self, order):
        """Submit an order to IB and handle the immediate response.

        Steps:
        1. Mark the order as submitted in Backtrader.
        2. Assign an OCA group UUID (or inherit from an ``oco`` order).
        3. Place the order via ``IBStore.placeOrder()`` and wait for TWS to
           acknowledge it.
        4. If TWS reports the order as already filled, complete it immediately.
           Otherwise, add it to ``open_orders`` for polling in ``next()``.

        Args:
            order (IBOrder): A fully constructed order ready to send.

        Returns:
            IBOrder: The same order object (status updated).
        """
        order.submit(self)

        # Assign OCA group: a fresh UUID for standalone orders, or inherit the
        # group from the linked ``oco`` order so they cancel each other.
        if order.oco is None:
            order.ocaGroup = uuid.uuid4()
        else:
            order.ocaGroup = self.orderbyid[order.oco.orderId].ocaGroup

        # Place the order and block until TWS confirms it has been received.
        trade = self.ibstore.placeOrder(order.orderId, order.data.tradecontract, order)

        # Notify the strategy that the order has been submitted.
        self.notify(order)

        if trade.orderStatus.status == self.FILLED:
            # Order filled immediately (e.g. market order during open market).
            order.completed()
            self.notify(order)
        else:
            # Order is live — track it for status updates.
            self.open_orders.append(order)

        return order

    def getcommissioninfo(self, data):
        """Build an ``IBCommInfo`` object for the given data feed's contract.

        The multiplier and stocklike flag drive Backtrader's internal value /
        cost calculations.  For futures/options the contract multiplier
        converts price to notional; for equities it is 1.

        Args:
            data: Backtrader data feed with a ``tradecontract`` attribute.

        Returns:
            IBCommInfo: Commission-info object attached to new orders.
        """
        contract = data.tradecontract
        try:
            mult = float(contract.multiplier)
        except (ValueError, TypeError):
            mult = 1.0  # equities, forex — no multiplier

        # stocklike=True disables margin-based position sizing in Backtrader.
        stocklike = contract.secType not in (
            "FUT",
            "OPT",
            "FOP",
        )

        return IBCommInfo(mult=mult, stocklike=stocklike)

    def _makeorder(
        self,
        action,
        owner,
        data,
        size,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        **kwargs,
    ):
        """Create an ``IBOrder`` without submitting it.

        Fetches the next available IB order ID from TWS and attaches
        commission info before returning the order.

        Args:
            action (str): ``'BUY'`` or ``'SELL'``.
            owner: The Backtrader strategy instance that owns this order.
            data: The data feed the order is for.
            size (float): Number of units to trade.
            price (float, optional): Limit / stop price.
            plimit (float, optional): Limit price for stop-limit orders.
            exectype: Backtrader exec type constant (e.g. ``Order.Limit``).
            valid: Expiry specification (see ``IBOrder.__init__``).
            tradeid (int): Backtrader trade grouping ID (not used by IB).
            **kwargs: Extra IB order fields passed through to ``IBOrder``.

        Returns:
            IBOrder: Constructed but not yet submitted order.
        """
        order = IBOrder(
            action,
            owner=owner,
            data=data,
            size=size,
            price=price,
            pricelimit=plimit,
            exectype=exectype,
            valid=valid,
            tradeid=tradeid,
            clientId=self.ibstore.clientId,
            orderId=self.ibstore.nextOrderId(),
            **kwargs,
        )

        order.addcomminfo(self.getcommissioninfo(data))
        return order

    def buy(
        self,
        owner,
        data,
        size,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        **kwargs,
    ):
        """Create and submit a buy order.

        Args:
            owner: The calling strategy.
            data: Data feed identifying the instrument.
            size (float): Number of units to buy.
            price (float, optional): Order price (limit / stop level).
            plimit (float, optional): Limit price for stop-limit orders.
            exectype: Backtrader exec type (default ``Order.Market``).
            valid: Order validity / TIF (see ``IBOrder.__init__``).
            tradeid (int): Strategy-internal trade ID.
            **kwargs: Extra IB order fields.

        Returns:
            IBOrder: Submitted buy order.
        """
        order = self._makeorder(
            "BUY", owner, data, size, price, plimit, exectype, valid, tradeid, **kwargs
        )

        return self.submit(order)

    def sell(
        self,
        owner,
        data,
        size,
        price=None,
        plimit=None,
        exectype=None,
        valid=None,
        tradeid=0,
        **kwargs,
    ):
        """Create and submit a sell order.

        Args:
            owner: The calling strategy.
            data: Data feed identifying the instrument.
            size (float): Number of units to sell.
            price (float, optional): Order price (limit / stop level).
            plimit (float, optional): Limit price for stop-limit orders.
            exectype: Backtrader exec type (default ``Order.Market``).
            valid: Order validity / TIF (see ``IBOrder.__init__``).
            tradeid (int): Strategy-internal trade ID.
            **kwargs: Extra IB order fields.

        Returns:
            IBOrder: Submitted sell order.
        """
        order = self._makeorder(
            "SELL", owner, data, size, price, plimit, exectype, valid, tradeid, **kwargs
        )

        return self.submit(order)

    def notify(self, order):
        """Enqueue a clone of ``order`` for delivery to the strategy.

        Backtrader's notification machinery calls ``get_notification()`` each
        bar to drain this queue and call ``strategy.notify_order()``.

        Args:
            order (IBOrder): The order whose current state should be notified.
        """
        self.notifs.put(order.clone())

    def get_notification(self):
        """Dequeue and return the next pending order notification.

        Returns:
            IBOrder or None: The next cloned order, or ``None`` if the queue
            is empty.
        """
        try:
            return self.notifs.get(False)
        except queue.Empty:
            pass

        return None

    # IB trade status strings returned by ``trade.orderStatus.status``.
    # Used as string constants throughout ``next()`` to avoid typos.
    (
        SUBMITTED,
        FILLED,
        CANCELLED,
        INACTIVE,
        PENDINGSUBMIT,
        PENDINGCANCEL,
        PRESUBMITTED,
    ) = (
        "Submitted",
        "Filled",
        "Cancelled",
        "Inactive",
        "PendingSubmit",
        "PendingCancel",
        "PreSubmitted",
    )

    def next(self):
        """Poll open orders and drive the Backtrader order state machine.

        Called by Cerebro once per bar.  Fetches the current trade list from
        IB and matches each ``open_order`` against a live ``ib_async.Trade``
        by ``orderId``.  Transitions the Backtrader order through the
        following states based on the IB trade status:

        +-----------------+--------------------------------------------------+
        | IB status       | Backtrader action                                |
        +=================+==================================================+
        | Submitted (0    | ``order.accept()`` — order acknowledged by TWS   |
        | fills so far)   | but not yet matched                              |
        +-----------------+--------------------------------------------------+
        | Cancelled       | ``order.expire()`` if ``_willexpire`` is set     |
        |                 | (GTD/DAY expiry), otherwise ``order.cancel()``   |
        +-----------------+--------------------------------------------------+
        | PendingCancel   | ``order.cancel()`` — cancellation in progress    |
        +-----------------+--------------------------------------------------+
        | Inactive        | ``order.reject()`` — typically a margin / config |
        |                 | rejection from TWS                               |
        +-----------------+--------------------------------------------------+
        | Submitted or    | ``order.completed()`` — fully filled             |
        | Filled (non-0   |                                                  |
        | fills)          |                                                  |
        +-----------------+--------------------------------------------------+
        | PendingSubmit,  | ``notify()`` only — no state change yet          |
        | PreSubmitted    |                                                  |
        +-----------------+--------------------------------------------------+

        A ``None`` sentinel is pushed onto ``self.notifs`` at the start to
        mark the boundary of this bar's notifications.
        """
        # Push a None boundary marker so Cerebro knows which notifications
        # belong to this bar vs the next.
        self.notifs.put(None)

        if len(self.open_orders) == 0:
            return  # nothing to do

        # Fetch the current live trades list from TWS (one round-trip).
        trades = self.ibstore.reqTrades()

        for order in self.open_orders:
            for trade in trades:
                if order.orderId != trade.order.orderId:
                    continue  # not this order

                status = trade.orderStatus.status

                if status == self.SUBMITTED and trade.filled == 0:
                    # TWS has acknowledged the order but no fill has arrived.
                    if order.status != order.Accepted:
                        order.accept(self)
                        self.notify(order)

                elif status == self.CANCELLED:
                    # Duplicate-detection: skip if already in a terminal state.
                    if order.status in [order.Cancelled, order.Expired]:
                        pass
                    elif order._willexpire:
                        # An openOrder with PendingCancel/Cancelled was seen
                        # earlier, indicating a GTD/DAY time-based expiry.
                        order.expire()
                    else:
                        # Direct user cancellation (no prior openOrder signal).
                        order.cancel()
                    self.open_orders.remove(order)
                    self.notify(order)

                elif status == self.PENDINGCANCEL:
                    # Documented as an internal-only status but seen in practice.
                    if order.status == order.Cancelled:
                        pass  # already handled
                    else:
                        order.cancel()
                    self.open_orders.remove(order)
                    self.notify(order)

                elif status == self.INACTIVE:
                    # Usually means the order was rejected (e.g. insufficient
                    # margin).  According to the IB docs it can also mean the
                    # order is temporarily inactive and may be reactivated, but
                    # treating it as a rejection is the safe default.
                    if order.status == order.Rejected:
                        pass  # duplicate
                    order.reject(self)
                    self.open_orders.remove(order)
                    self.notify(order)

                elif status in [self.SUBMITTED, self.FILLED]:
                    # A non-zero fill count with Submitted or a full Fill
                    # status means the order is done.
                    order.completed()
                    self.open_orders.remove(order)
                    self.notify(order)

                elif status in [self.PENDINGSUBMIT, self.PRESUBMITTED]:
                    # Transient states — just notify so the strategy can log
                    # them.  No state transition yet.
                    self.notify(order)

                else:
                    # Unknown / unexpected status — do nothing.
                    pass
        return

    # ---------------------------------------------------------------------------
    # The methods below (push_execution, push_commissionreport, push_portupdate,
    # push_ordererror, push_orderstate) are commented out.  They represent the
    # callback-based approach used by the original ibpy library.  With ib_async
    # the equivalent logic is handled by polling ib.trades() in next() above.
    # They are retained as a reference for future implementation.
    # ---------------------------------------------------------------------------

    # def push_execution(self, ex):
    #     self.executions[ex.m_execId] = ex

    # def push_commissionreport(self, cr):
    #     with self._lock_orders:
    #         ex = self.executions.pop(cr.m_execId)
    #         oid = ex.m_orderId
    #         order = self.orderbyid[oid]
    #         ostatus = self.ordstatus[oid].pop(ex.m_cumQty)

    #         position = self.getposition(order.data, clone=False)
    #         pprice_orig = position.price
    #         size = ex.m_shares if ex.m_side[0] == 'B' else -ex.m_shares
    #         price = ex.m_price
    #         # use pseudoupdate and let the updateportfolio do the real update?
    #         psize, pprice, opened, closed = position.update(size, price)

    #         # split commission between closed and opened
    #         comm = cr.m_commission
    #         closedcomm = comm * closed / size
    #         openedcomm = comm - closedcomm

    #         comminfo = order.comminfo
    #         closedvalue = comminfo.getoperationcost(closed, pprice_orig)
    #         openedvalue = comminfo.getoperationcost(opened, price)

    #         # default in m_pnl is MAXFLOAT
    #         pnl = cr.m_realizedPNL if closed else 0.0

    #         # The internal broker calc should yield the same result
    #         # pnl = comminfo.profitandloss(-closed, pprice_orig, price)

    #         # Use the actual time provided by the execution object
    #         # The report from TWS is in actual local time, not the data's tz
    #         dt = date2num(datetime.strptime(ex.m_time, '%Y%m%d  %H:%M:%S'))

    #         # Need to simulate a margin, but it plays no role, because it is
    #         # controlled by a real broker. Let's set the price of the item
    #         margin = order.data.close[0]

    #         order.execute(dt, size, price,
    #                       closed, closedvalue, closedcomm,
    #                       opened, openedvalue, openedcomm,
    #                       margin, pnl,
    #                       psize, pprice)

    #         if ostatus.status == self.FILLED:
    #             order.completed()
    #             self.ordstatus.pop(oid)  # nothing left to be reported
    #         else:
    #             order.partial()

    #         if oid not in self.tonotify:  # Lock needed
    #             self.tonotify.append(oid)

    # def push_portupdate(self):
    #     # If the IBStore receives a Portfolio update, then this method will be
    #     # indicated. If the execution of an order is split in serveral lots,
    #     # updatePortfolio messages will be intermixed, which is used as a
    #     # signal to indicate that the strategy can be notified
    #     with self._lock_orders:
    #         while self.tonotify:
    #             oid = self.tonotify.popleft()
    #             order = self.orderbyid[oid]
    #             self.notify(order)

    # def push_ordererror(self, msg):
    #     with self._lock_orders:
    #         try:
    #             order = self.orderbyid[msg.id]
    #         except (KeyError, AttributeError):
    #             return  # no order or no id in error

    #         if msg.errorCode == 202:
    #             if not order.alive():
    #                 return
    #             order.cancel()

    #         elif msg.errorCode == 201:  # rejected
    #             if order.status == order.Rejected:
    #                 return
    #             order.reject()

    #         else:
    #             order.reject()  # default for all other cases

    #         self.notify(order)

    # def push_orderstate(self, msg):
    #     with self._lock_orders:
    #         try:
    #             order = self.orderbyid[msg.orderId]
    #         except (KeyError, AttributeError):
    #             return  # no order or no id in error

    #         if msg.orderState.m_status in ['PendingCancel', 'Cancelled',
    #                                        'Canceled']:
    #             # This is most likely due to an expiration]
    #             order._willexpire = True
