from typing import Optional

from .base import Analyzer


class ProgressTracker(Analyzer):
    """Report strategy progress back to a shared dictionary."""

    params = (
        ("progress_dict", None),
        ("total_map", None),
        ("report_every", 0),
    )

    def __init__(self):
        self._data = None
        self._total = 0
        self._key: Optional[str] = None
        self._report_every = 1
        self._last_reported = 0
        self._name: str = ""

    def start(self):
        data_index = getattr(self.strategy.params, "data_index", 0)
        data_name = getattr(self.strategy.params, "data_name", "")
        datas = list(self.strategy.datas)
        if 0 <= data_index < len(datas):
            self._data = datas[data_index]
        elif datas:
            self._data = datas[0]

        if data_name:
            self._name = str(data_name)
        else:
            self._name = f"strategy_{os.getpid()}"

        totals = self.params.total_map or {}
        self._total = int(totals.get(data_name, 0) or 0)
        if (not self._total or self._total <= 0) and self._data is not None:
            data_label = getattr(self._data, "_name", "") or getattr(
                self._data, "_dataname", ""
            )
            if data_label:
                self._total = int(totals.get(str(data_label), 0) or 0)
            if not data_name and data_label:
                self._name = str(data_label)

        report_every_param = int(self.params.report_every or 0)
        if report_every_param <= 0 and self._total:
            report_every_param = max(1, self._total // 20)
        self._report_every = max(1, report_every_param or 1)
        self._last_reported = -self._report_every

        progress_dict = self.params.progress_dict
        if progress_dict is not None:
            self._key = f"{os.getpid()}::{self._name}::{id(self)}"
            progress_dict[self._key] = {
                "name": self._name,
                "processed": 0,
                "total": self._total,
            }

    def next(self):
        if self._data is None:
            return
        processed = len(self._data)
        if processed - self._last_reported >= self._report_every:
            self._record(processed)

    def stop(self):
        if self._data is None:
            return
        self._record(len(self._data))

    def _record(self, processed: int) -> None:
        processed = int(processed)
        if self._total:
            processed = min(processed, self._total)
        self._last_reported = processed

        progress_dict = self.params.progress_dict
        if progress_dict is not None and self._key is not None:
            progress_dict[self._key] = {
                "name": self._name,
                "processed": processed,
                "total": self._total,
            }
