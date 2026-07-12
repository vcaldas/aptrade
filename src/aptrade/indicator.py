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

from .lineiterator import IndicatorBase, LineIterator
from .lineseries import Lines
from .metabase import AutoInfoClass


class MetaIndicator(IndicatorBase.__class__):
    _refname = "_indcol"
    _indcol = {}

    _icache = {}
    _icacheuse = False

    @classmethod
    def cleancache(cls):
        cls._icache = {}

    @classmethod
    def usecache(cls, onoff):
        cls._icacheuse = onoff

    # Object cache deactivated on 2016-08-17. If the object is being used
    # inside another object, the minperiod information carried over
    # influences the first usage when being modified during the 2nd usage

    def __call__(self, *args, **kwargs):
        if not self._icacheuse:
            return super().__call__(*args, **kwargs)

        # implement a cache to avoid duplicating lines actions
        ckey = (self, tuple(args), tuple(kwargs.items()))  # tuples hashable
        try:
            return self._icache[ckey]
        except TypeError:  # something not hashable
            return super().__call__(*args, **kwargs)
        except KeyError:
            pass  # hashable but not in the cache

        _obj = super().__call__(*args, **kwargs)
        return self._icache.setdefault(ckey, _obj)

    def __init__(self, name, bases, dct):
        """
        Class has already been created ... register subclasses
        """
        # Initialize the class
        super().__init__(name, bases, dct)

        if not self.aliased and name != "Indicator" and not name.startswith("_"):
            refattr = getattr(self, self._refname)
            refattr[name] = self

        # Check if next and once have both been overridden
        next_over = self.next != IndicatorBase.next
        once_over = self.once != IndicatorBase.once

        if next_over and not once_over:
            # No -> need pointer movement to once simulation via next
            self.once = self.once_via_next
            self.preonce = self.preonce_via_prenext
            self.oncestart = self.oncestart_via_nextstart


class Indicator(IndicatorBase, metaclass=MetaIndicator):
    _ltype = LineIterator.IndType

    csv = False

    def advance(self, size=1):
        # Need intercepting this call to support datas with
        # different lengths (timeframes)
        if len(self) < len(self._clock):
            self.lines.advance(size=size)

    def preonce_via_prenext(self, start, end):
        # generic implementation if prenext is overridden but preonce is not
        for _i in range(start, end):
            for data in self.datas:
                data.advance()

            for indicator in self._ind_iterator:
                indicator.advance()

            self.advance()
            self.prenext()

    def oncestart_via_nextstart(self, start, end):
        # nextstart has been overriden, but oncestart has not and the code is
        # here. call the overriden nextstart
        for _i in range(start, end):
            for data in self.datas:
                data.advance()

            for indicator in self._ind_iterator:
                indicator.advance()

            self.advance()
            self.nextstart()

    def once_via_next(self, start, end):
        # Not overridden, next must be there ...
        for _i in range(start, end):
            for data in self.datas:
                data.advance()

            for indicator in self._ind_iterator:
                indicator.advance()

            self.advance()
            self.next()


class MtLinePlotterIndicator(Indicator.__class__):
    def donew(self, *args, **kwargs):
        lname = kwargs.pop("name")
        name = self.__name__

        lines = getattr(self, "lines", Lines)
        self.lines = lines._derive(name, (lname,), 0, [])

        plotlines = AutoInfoClass
        newplotlines = {}
        newplotlines.setdefault(lname, {})
        self.plotlines = plotlines._derive(name, newplotlines, [], recurse=True)

        # Create the object and set the params in place
        _obj, args, kwargs = super().donew(*args, **kwargs)

        _obj.owner = _obj.data.owner._clock
        _obj.data.lines[0].addbinding(_obj.lines[0])

        # Return the object and arguments to the chain
        return _obj, args, kwargs


class LinePlotterIndicator(Indicator, metaclass=MtLinePlotterIndicator):
    pass
