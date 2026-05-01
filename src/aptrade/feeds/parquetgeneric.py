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

import aptrade as bt
from aptrade.feed import DataBase
from aptrade.stores import parquetstore
from aptrade.utils.dateintern import date2num


class MetaParquetData(DataBase.__class__):
	def __init__(cls, name, bases, dct):
		"""Class has already been created ... register"""
		super(MetaParquetData, cls).__init__(name, bases, dct)
		setattr(parquetstore.ParquetStore, "DataCls", cls)


class ParquetData(DataBase, metaclass=MetaParquetData):
	"""Parquet-backed data feed.

	Uses ``ParquetStore`` to read files with layout:

	  - stock/{stock_name}/{year}/{month}/{day}/day.parquet
	  - stock/{stock_name}/{year}/{month}/{day}/trades.parquet
	  - stock/{stock_name}/{year}/{month}/{day}/1-minute.parquet

	Parameters:
	  - ``path``: datastore root path
	  - ``interval``: parquet interval key (``day``, ``trades``, ``1-minute``, ``1m``)
	  - ``start_date``: explicit start date (optional, falls back to ``fromdate``)
	  - ``end_date``: explicit end date (optional, falls back to ``todate``)
	  - ``nocase``: case-insensitive column lookup
	  - ``datetime``, ``open``, ``high``, ``low``, ``close``, ``volume``, ``openinterest``:
		column mappings (same semantics as ``PandasData``)
	"""

	params = (
		("path", None),
		("interval", "1-minute"),
		("start_date", None),
		("end_date", None),
		("nocase", True),
		("datetime", None),
		("open", -1),
		("high", -1),
		("low", -1),
		("close", -1),
		("volume", -1),
		("openinterest", -1),
	)

	_store = parquetstore.ParquetStore

	def __init__(self, **kwargs):
		self.parquet = self._store(path=self.p.path)
		self._df = None
		self._idx = -1
		self._colmapping = dict()

	def setenvironment(self, env):
		super(ParquetData, self).setenvironment(env)
		env.addstore(self.parquet)

	def start(self):
		super(ParquetData, self).start()
		self.parquet.start(data=self)

		start_date = self.p.start_date if self.p.start_date is not None else self.p.fromdate
		end_date = self.p.end_date if self.p.end_date is not None else self.p.todate

		if start_date is None or end_date is None:
			raise ValueError(
				"ParquetData requires start/end date. Pass start_date/end_date or fromdate/todate."
			)

		self._df = self.parquet.get_timeseries(
			stock_name=self.p.dataname,
			start_date=start_date,
			end_date=end_date,
			interval=self.p.interval,
		)

		self._idx = -1
		self._build_colmapping()

	def _build_colmapping(self):
		self._colmapping = dict()

		if self._df is None:
			return

		colnames = list(self._df.columns.values)
		for datafield in self.getlinealiases():
			defmapping = getattr(self.params, datafield)

			if isinstance(defmapping, int) and defmapping < 0:
				for colname in colnames:
					if not isinstance(colname, str):
						continue

					if self.p.nocase:
						found = datafield.lower() == colname.lower()
					else:
						found = datafield == colname

					if found:
						self._colmapping[datafield] = colname
						break

				if datafield not in self._colmapping:
					self._colmapping[datafield] = None
					continue
			else:
				self._colmapping[datafield] = defmapping

		if self.p.nocase:
			normalized = [x.lower() if isinstance(x, str) else x for x in self._df.columns.values]
		else:
			normalized = [x for x in self._df.columns.values]

		for key, value in self._colmapping.items():
			if value is None:
				continue

			if isinstance(value, str):
				try:
					if self.p.nocase:
						value = normalized.index(value.lower())
					else:
						value = normalized.index(value)
				except ValueError:
					defmap = getattr(self.params, key)
					if isinstance(defmap, int) and defmap < 0:
						value = None
					else:
						raise

			self._colmapping[key] = value

	def _to_datetime_num(self, value):
		try:
			dt = value.to_pydatetime()
		except AttributeError:
			import pandas as pd

			dt = pd.Timestamp(value).to_pydatetime()

		return date2num(dt)

	def _load(self):
		if self._df is None or self._df.empty:
			return False

		self._idx += 1
		if self._idx >= len(self._df):
			return False

		for datafield in self.getlinealiases():
			if datafield == "datetime":
				continue

			colindex = self._colmapping.get(datafield)
			line = getattr(self.lines, datafield)

			if colindex is None:
				if datafield == "openinterest":
					line[0] = 0.0
				continue

			line[0] = self._df.iloc[self._idx, colindex]

		coldtime = self._colmapping.get("datetime")
		if coldtime is None:
			tstamp = self._df.index[self._idx]
		else:
			tstamp = self._df.iloc[self._idx, coldtime]

		self.lines.datetime[0] = self._to_datetime_num(tstamp)
		return True


class ParquetGeneric(bt.FeedBase):
	"""Feed factory for ``ParquetData``."""

	DataCls = ParquetData
	params = DataCls.params._gettuple()

