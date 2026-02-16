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
from __future__ import absolute_import, division, print_function, unicode_literals

# Load contributed indicators and studies
import numpy

import aptrade.commissions as commissions
import aptrade.sizer as sizer
from aptrade.analyzers import *
from aptrade.commissions import *

# from . import analyzers as analyzers
from . import broker as broker
from . import brokers as brokers
from . import errors as errors
from . import feeds as feeds
from . import filters as filters
from . import indicators as indicators
from . import observers as observers
from . import stores as stores
from . import talib as talib
from . import timer as timer
from . import utils as utils
from .aptsignal import *
from .broker import *
from .cerebro import *
from .dataseries import *
from .errors import *
from .feed import *
from .flt import *
from .functions import *
from .indicator import *
from .linebuffer import *
from .lineiterator import *
from .lineseries import *
from .observer import *
from .order import *
from .position import *
from .resamplerfilter import *
from .sizer import *
from .strategy import *
from .timer import *
from .trade import *
from .writer import *


def hello(n: int) -> str:
    """Greet the sum from 0 to n (exclusive end)."""
    sum_n = numpy.arange(n).sum()
    return f"Hello {sum_n}!"
