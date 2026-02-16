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

import queue as queue
import sys

try:
    import winreg
except ImportError:
    winreg = None

MAXINT = sys.maxsize
MININT = -sys.maxsize - 1

# MAXFLOAT = sys.float_info.max
# MINFLOAT = sys.float_info.min


def cmp(a, b):
    return (a > b) - (a < b)


def iterkeys(d):
    return iter(d.keys())


def itervalues(d):
    return iter(d.values())


def iteritems(d):
    return iter(d.items())


def keys(d):
    return list(d.keys())


def values(d):
    return list(d.values())


def items(d):
    return list(d.items())
