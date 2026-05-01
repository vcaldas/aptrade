#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2026 Victor Caldas
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

import datetime as _dt
from pathlib import Path

import aptrade as bt


class ParquetStore(bt.Store):
    """Store for local parquet time series files.

    Expected directory layout:

      - stock/{stock_name}/{year}/{month}/{day}/day.parquet
      - stock/{stock_name}/{year}/{month}/{day}/trades.parquet
      - stock/{stock_name}/{year}/{month}/{day}/1-minute.parquet

    Params:
      - ``path`` (default: ``None``): datastore root path
    """

    params = (("path", None),)

    _INTERVAL_FILES = {
        "day": "day.parquet",
        "trades": "trades.parquet",
        "1-minute": "1-minute.parquet",
        "1m": "1-minute.parquet",
    }

    def __init__(self):
        self._path = Path(self.p.path).expanduser().resolve() if self.p.path else None

    def set_path(self, path):
        """Set/replace datastore root path."""
        self._path = Path(path).expanduser().resolve()

    def get_path(self):
        """Return configured datastore root path as string or ``None``."""
        return None if self._path is None else str(self._path)

    def get_timeseries(self, stock_name, start_date, end_date, interval):
        """Return a DataFrame concatenated across the requested date range.

        Args:
          - stock_name (str): stock identifier in the path.
          - start_date (date|datetime|str): inclusive start date.
          - end_date (date|datetime|str): inclusive end date.
          - interval (str): one of ``day``, ``trades``, ``1-minute`` (or ``1m``).
        """
        if self._path is None:
            raise ValueError("ParquetStore path is not configured. Pass params path=... or call set_path(...)")

        interval_key = str(interval).strip().lower()
        try:
            interval_file = self._INTERVAL_FILES[interval_key]
        except KeyError:
            supported = ", ".join(sorted(self._INTERVAL_FILES.keys()))
            raise ValueError("Unsupported interval {!r}. Supported: {}".format(interval, supported))

        start = self._as_date(start_date)
        end = self._as_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to read parquet data. Install pandas and a parquet engine (for example pyarrow)."
            )

        frames = []
        day = start
        one_day = _dt.timedelta(days=1)

        while day <= end:
            parquet_path = self._find_daily_file(stock_name=stock_name, day=day, filename=interval_file)
            if parquet_path is not None:
                frames.append(pd.read_parquet(parquet_path))

            day += one_day

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return self._filter_datetime(result, start_date=start_date, end_date=end_date)

    @staticmethod
    def _as_date(value):
        if isinstance(value, _dt.datetime):
            return value.date()
        if isinstance(value, _dt.date):
            return value

        text = str(value).strip()
        try:
            return _dt.date.fromisoformat(text)
        except ValueError:
            return _dt.datetime.fromisoformat(text).date()

    def _find_daily_file(self, stock_name, day, filename):
        candidates = self._candidate_paths(stock_name=stock_name, day=day, filename=filename)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _candidate_paths(self, stock_name, day, filename):
        # Support both zero-padded and non-zero-padded month/day directories.
        year = str(day.year)
        month_padded = "{:02d}".format(day.month)
        day_padded = "{:02d}".format(day.day)
        month_plain = str(day.month)
        day_plain = str(day.day)

        base = self._path / "stock" / str(stock_name) / year
        return [
            base / month_padded / day_padded / filename,
            base / month_plain / day_plain / filename,
        ]

    @staticmethod
    def _filter_datetime(df, start_date, end_date):
        try:
            import pandas as pd
        except ImportError:
            return df

        if df.empty:
            return df

        datetime_col = None
        for col in ("datetime", "Datetime", "timestamp", "Timestamp", "date", "Date"):
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            return df

        sdt = pd.Timestamp(start_date)
        edt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        dt_series = pd.to_datetime(df[datetime_col], errors="coerce")
        mask = (dt_series >= sdt) & (dt_series <= edt)
        return df.loc[mask].reset_index(drop=True)
