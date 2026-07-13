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
from dataclasses import dataclass

from aptrade.sizers import AbstractSizer

__all__ = ["PercentSizer", "AllInSizer", "PercentSizerInt", "AllInSizerInt"]


@dataclass(slots=True, frozen=True)
class PercentSizerParams:
    """Parameters for PercentSizer."""

    percents: float = 20.0
    retint: bool = False


@dataclass(slots=True, frozen=True)
class AllInSizerParams(PercentSizerParams):
    """Parameters for AllInSizer."""

    percents: float = 100.0


@dataclass(slots=True, frozen=True)
class PercentSizerIntParams(PercentSizerParams):
    """Parameters for PercentSizerInt."""

    retint: bool = True


@dataclass(slots=True, frozen=True)
class AllInSizerIntParams(PercentSizerIntParams):
    """Parameters for AllInSizerInt."""

    percents: float = 100.0


class PercentSizer(AbstractSizer):
    """Sizer that allocates a percentage of available cash."""

    Parameters = PercentSizerParams

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)

        if not position:
            size = cash / data.close[0] * (self.p.percents / 100.0)
        else:
            size = position.size

        return int(size) if self.p.retint else size


class AllInSizer(PercentSizer):
    """Sizer that allocates 100% of available cash."""

    Parameters = AllInSizerParams


class PercentSizerInt(PercentSizer):
    """Sizer that allocates a percentage of cash and returns an integer size."""

    Parameters = PercentSizerIntParams


class AllInSizerInt(PercentSizerInt):
    """Sizer that allocates all available cash and returns an integer size."""

    Parameters = AllInSizerIntParams
