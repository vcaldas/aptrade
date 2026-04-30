#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# The modules below should/must define __all__ with the objects wishes
# or prepend an "_" (underscore) to private classes/variables
from abc import ABC, abstractmethod
from types import SimpleNamespace

from aptrade.metabase import MetaParams


class PositionSizer(ABC):
    @abstractmethod
    def compute_size(
        self,
        portfolio_value: float,
        cash: float,
        price: float,
        commission_per_unit: float,
        is_buy: bool,
    ) -> int:
        """
        Compute position size.

        Args:
            portfolio_value: Total portfolio value (cash + positions)
            cash: Available cash
            price: Asset price
            commission_per_unit: Commission added per unit
            is_buy: True for buy, False for sell

        Returns:
            Integer size to execute.
        """
        raise NotImplementedError


class AbstractSizer(ABC):
    """
    The Abstract Factory interface declares a set of methods that return
    different abstract products. These products are called a family and are
    related by a high-level theme or concept. Products of one family are usually
    able to collaborate among themselves. A family of products may have several
    variants, but the products of one variant are incompatible with products of
    another.
    """

    strategy = None
    broker = None

    def getsizing(self, data, isbuy: bool):
        comminfo = self.broker.getcommissioninfo(data)
        return self._getsizing(comminfo, self.broker.getcash(), data, isbuy)

    def __init__(self, *args, **kwargs):
        """Initialize params for sizers that declare a `params` tuple.

        Supports positional and keyword parameters to remain compatible
        with the previous `MetaParams`-based behavior used by sizers.
        """
        params_def = getattr(self.__class__, "params", ()) or ()
        # Start with defaults
        params_obj = SimpleNamespace()
        param_names = [name for name, _ in params_def]
        for name, default in params_def:
            setattr(params_obj, name, default)

        # Apply positional args in order
        for name, val in zip(param_names, args):
            setattr(params_obj, name, val)

        # Apply keyword overrides
        for name in param_names:
            if name in kwargs:
                setattr(params_obj, name, kwargs.pop(name))

        # expose as both `params` and `p` for compatibility
        self.params = params_obj
        self.p = params_obj

    @abstractmethod
    def _getsizing(self, comminfo, cash, data, isbuy: bool) -> int:
        """This method has to be overriden by subclasses of Sizer to provide
        the sizing functionality

        Params:
          - ``comminfo``: The CommissionInfo instance that contains
            information about the commission for the data and allows
            calculation of position value, operation cost, commision for the
            operation

          - ``cash``: current available cash in the *broker*

          - ``data``: target of the operation

          - ``isbuy``: will be ``True`` for *buy* operations and ``False``
            for *sell* operations

        The method has to return the actual size (an int) to be executed. If
        ``0`` is returned nothing will be executed.

        The absolute value of the returned value will be used

        """
        raise NotImplementedError

    def set(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker

from .fixedsize import *
from .percents_sizer import *
from .simple import *
