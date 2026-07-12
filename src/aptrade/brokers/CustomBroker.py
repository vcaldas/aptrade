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
"""Custom broker starting point.

This module intentionally mirrors the default aptrade/backtrader broker
behavior by subclassing ``BackBroker`` without changing any runtime logic.
It provides a clean place to debug broker behavior from first principles
without affecting the rest of the codebase.
"""

import queue

from aptrade import BrokerBase
from aptrade.stores.ibstore import IBStore

__all__ = ["CustomBroker", "BrokerCustom"]


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


class CustomBroker(BrokerBase, metaclass=MetaIBBroker):
    """Exact default broker behavior, isolated under a new name."""

    params = (
        ("cash", 10000.0),
        ("checksubmit", True),
        ("eosbar", False),
        ("filler", None),
        # slippage options
        ("slip_perc", 0.0),
        ("slip_fixed", 0.0),
        ("slip_open", False),
        ("slip_match", True),
        ("slip_limit", True),
        ("slip_out", False),
        ("coc", False),
        ("coo", False),
        ("int2pnl", True),
        ("shortcash", True),
        ("fundstartval", 100.0),
        ("fundmode", False),
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Obtain (or create) the IBStore singleton with the given connection
        # parameters.  All IBData feeds share this same store.
        self.ibstore = IBStore(**kwargs)
        self._userhist = []
        self._fundhist = []
        # share_value, net asset value
        self._fhistlast = [float("NaN"), float("NaN")]

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


BrokerCustom = CustomBroker
