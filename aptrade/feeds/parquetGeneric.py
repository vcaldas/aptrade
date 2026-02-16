# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false

from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .. import TimeFrame, feed
from ..utils import date2num


class GenericParquetData(feed.DataBase):
    """Generic Parquet reader with column-to-line mapping like :class:`GenericCSVData`."""

    params = (
        ("nullvalue", float("NaN")),
        ("dtformat", "%Y-%m-%d %H:%M:%S"),
        ("tmformat", "%H:%M:%S"),
        ("datetime", 0),
        ("time", -1),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", 6),
    )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Parquet reader / tables cannot be pickled; rebuild after unpickling
        state["_parquet"] = None
        state["_current_table"] = None
        state["_current_columns"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._parquet = None
        self._current_table = None
        self._current_columns = None

    def _open_parquet(self):
        dataname = self.p.dataname
        if isinstance(dataname, Path):
            dataname = str(dataname)

        self._parquet = pq.ParquetFile(dataname)

    def start(self):
        super(GenericParquetData, self).start()

        self._dtstr = False
        self._dtconvert = None  # type: ignore[assignment]
        if isinstance(self.p.dtformat, str):
            self._dtstr = True
        elif isinstance(self.p.dtformat, int):
            idt = int(self.p.dtformat)
            if idt == 1:
                self._dtconvert = lambda x: datetime.utcfromtimestamp(int(x))
            elif idt == 2:
                self._dtconvert = lambda x: datetime.utcfromtimestamp(float(x))
        else:
            self._dtconvert = self.p.dtformat

        self._open_parquet()
        self._row_group = 0
        self._row_in_group = 0
        self._row_count = 0
        self._current_table = None
        self._current_columns: Optional[List[Any]] = None

        schema = self._parquet.schema_arrow
        names = list(schema.names) if schema.names is not None else []
        self._name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(names)
        }

        def _resolve(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, int):
                return value if value >= 0 else None
            if isinstance(value, str):
                return self._name_to_index.get(value)
            return None

        self._datetime_idx = _resolve(self.p.datetime)

        if self._datetime_idx is None:
            raise ValueError("datetime column must be specified for GenericParquetData")

        self._time_idx = _resolve(self.p.time)

        self._line_indices = {}
        for linefield in (x for x in self.getlinealiases() if x != "datetime"):
            self._line_indices[linefield] = _resolve(getattr(self.p, linefield))

    def _ensure_row(self) -> bool:
        if self._parquet is None:
            self._open_parquet()

        while True:
            if self._current_table is not None and self._row_in_group < self._row_count:
                return True
            if self._row_group >= self._parquet.num_row_groups:
                return False

            table = self._parquet.read_row_group(self._row_group)
            self._row_group += 1
            if table is None or table.num_rows == 0:
                continue

            table = table.combine_chunks()
            self._current_table = table
            self._current_columns = [table.column(i) for i in range(table.num_columns)]
            self._row_count = table.num_rows
            self._row_in_group = 0
            if self._row_count:
                return True

    def _get_value(self, column_index: int, row_index: int) -> Any:
        if self._current_columns is None:
            return None

        column = self._current_columns[column_index]
        value = column[row_index].as_py()
        if value is None:
            return None
        return value

    def _convert_datetime(self, dtvalue: Any, time_value: Any) -> datetime:
        if dtvalue is None:
            raise ValueError("Parquet row missing datetime value")

        if isinstance(dtvalue, datetime):
            dt = dtvalue
        elif isinstance(dtvalue, date) and isinstance(time_value, time):
            dt = datetime.combine(dtvalue, time_value)
        elif self._dtstr:
            dtfield = str(dtvalue)
            dtformat = self.p.dtformat
            if isinstance(time_value, time):
                base = datetime.strptime(dtfield, dtformat)
                dt = base.replace(
                    hour=time_value.hour,
                    minute=time_value.minute,
                    second=time_value.second,
                    microsecond=time_value.microsecond,
                )
            elif time_value is not None:
                time_str = str(time_value)
                dt = datetime.strptime(
                    dtfield + "T" + time_str,
                    dtformat + "T" + self.p.tmformat,
                )
            else:
                dt = datetime.strptime(dtfield, dtformat)
        else:
            if self._dtconvert is not None:
                dt = self._dtconvert(dtvalue)
            elif isinstance(dtvalue, date):
                dt = datetime.combine(dtvalue, time.min)
            elif isinstance(dtvalue, (int, float)):
                dt = datetime.utcfromtimestamp(dtvalue)
            else:
                dt = datetime.utcfromtimestamp(float(dtvalue))
            if isinstance(time_value, time):
                dt = datetime.combine(dt.date(), time_value)

        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, time.min)

        return dt

    def _load(self):
        if not self._ensure_row():
            return False

        row_index = self._row_in_group
        self._row_in_group += 1

        time_value = None
        if self._time_idx is not None:
            time_value = self._get_value(self._time_idx, row_index)

        dtvalue = self._get_value(self._datetime_idx, row_index)
        dt = self._convert_datetime(dtvalue, time_value)

        if self.p.timeframe >= TimeFrame.Days:
            if self._tzinput:
                dtin = self._tzinput.localize(dt)
            else:
                dtin = dt

            dtnum = date2num(dtin)

            dteos = datetime.combine(dt.date(), self.p.sessionend)
            dteosnum = self.date2num(dteos)

            if dteosnum > dtnum:
                self.lines.datetime[0] = dteosnum
            else:
                self.l.datetime[0] = date2num(dt) if self._tzinput else dtnum
        else:
            self.lines.datetime[0] = date2num(dt)

        for linefield in (x for x in self.getlinealiases() if x != "datetime"):
            col_index = self._line_indices.get(linefield)
            if col_index is None:
                value = self.p.nullvalue
            else:
                value = self._get_value(col_index, row_index)
                if value is None or value == "":
                    value = self.p.nullvalue

            line = getattr(self.lines, linefield)
            line[0] = float(value)

        return True


class GenericParquet(feed.FeedBase):
    params = (("basepath", ""),) + feed.FeedBase.params._gettuple()

    DataCls = GenericParquetData

    def _getdata(self, dataname, **kwargs):
        basepath = Path(self.p.basepath)
        if basepath:
            dataname = basepath / dataname
        return super(GenericParquet, self)._getdata(str(dataname), **kwargs)
