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
import datetime
from zoneinfo import ZoneInfo


def format_datetime(dt: datetime, tz: str | ZoneInfo = None) -> str:
    if tz is None:
        # tz = ZoneInfo(get_localzone_name())
        return dt.strftime("%Y-%m-%d %H:%M")
    elif isinstance(tz, str):
        tz = ZoneInfo(tz)
    # If dt does not contain tzinfo, assume it is in the specified zone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        # Convert datetime to the required timezone
        dt = dt.astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M GMT%z")


from .autodict import *  # noqa: F403, F401
from .date import *  # noqa: F403, F401
from .ordereddefaultdict import *  # noqa: F403, F401
