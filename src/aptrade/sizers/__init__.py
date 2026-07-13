#!/usr/bin/env python
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
# Copyright (C) 2025-2026 Victor Caldas
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

@dataclass(slots=True, frozen=True)
class EmptyParams:
    """Default parameter set."""

    pass


class AbstractSizer(ABC):
    """Base class for all position sizers."""

    Parameters: ClassVar[type] = EmptyParams

    strategy: Any | None = None
    broker: Any | None = None

    def __init__(self, **kwargs):
        self.p = self.Parameters(**kwargs)

    @property
    def params(self):
        """Backward compatibility."""
        return self.p

    def getsizing(self, data, isbuy: bool):
        comminfo = self.broker.getcommissioninfo(data)
        return self._getsizing(
            comminfo,
            self.broker.getcash(),
            data,
            isbuy,
        )

    @abstractmethod
    def _getsizing(self, comminfo, cash, data, isbuy: bool) -> int:
        raise NotImplementedError

    def set(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker


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


from .fixedsize import *  # noqa: F403, F401, E402
from .percents_sizer import *  # noqa: F403, F401, E402
from .simple import *  # noqa: F403, F401, E402
