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

import collections

from aptrade.metabase import MetaParams


class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(self, name, bases, dct):
        super().__init__(name, bases, dct)
        self._singleton = None

    def __call__(self, *args, **kwargs):
        if self._singleton is None:
            self._singleton = super().__call__(*args, **kwargs)

        return self._singleton


class Store(metaclass=MetaSingleton):
    """Base class for all Stores"""

    _started = False

    params = ()

    def getdata(self, *args, **kwargs):
        """Returns ``DataCls`` with args, kwargs"""
        data = self.DataCls(*args, **kwargs)
        data._store = self
        return data

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Returns broker with *args, **kwargs from registered ``BrokerCls``"""
        broker = cls.BrokerCls(*args, **kwargs)
        broker._store = cls
        return broker

    BrokerCls = None  # broker class will autoregister
    DataCls = None  # data class will auto register

    def start(self, data=None, broker=None):
        if not self._started:
            self._started = True
            self.notifs = collections.deque()
            self.datas = []
            self.broker = None

        if data is not None:
            self._cerebro = self._env = data._env
            self.datas.append(data)

            if self.broker is not None:
                if hasattr(self.broker, "data_started"):
                    self.broker.data_started(data)

        elif broker is not None:
            self.broker = broker

    def stop(self):
        pass

    def put_notification(self, msg, *args, **kwargs):
        self.notifs.append((msg, args, kwargs))

    def get_notifications(self):
        """Return the pending "store" notifications"""
        self.notifs.append(None)  # put a mark / threads could still append
        return list(iter(self.notifs.popleft, None))
