import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Analyzer


class TradeHistoryAnalyzer(Analyzer):
    """Capture closed trade details for later export."""

    params = (("export_dir", None),)

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._counter = 1

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        # print(self.strategy.current_day, self.strategy.prev_close)
        # print(self.strategy.__dict__)
        data = trade.data
        dataname = getattr(data, "_name", None) or getattr(data, "_dataname", "")
        history = getattr(trade, "history", []) or []
        entry_snapshot = history[0].status if history else None
        exit_event = getattr(history[-1], "event", None) if history else None
        strategy_label = getattr(self.strategy.params, "data_name", "")

        def _to_iso(dt_value: Optional[Any]) -> Optional[str]:
            if dt_value is None:
                return None
            try:
                return dt_value.isoformat()
            except AttributeError:
                return None

        entry_size = getattr(entry_snapshot, "size", trade.size)
        entry_price = getattr(entry_snapshot, "price", trade.price)
        exit_price = getattr(exit_event, "price", trade.value)
        direction = "long" if entry_size and entry_size > 0 else "short"

        record = {
            "trade_id": self._counter,
            "data": str(dataname),
            "direction": direction,
            "size": entry_size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": trade.pnl,
            "pnl_comm": trade.pnlcomm,
            "bar_open": trade.baropen,
            "bar_close": trade.barclose,
            "bar_len": trade.barlen,
            "dt_open": _to_iso(trade.open_datetime()),
            "dt_close": _to_iso(trade.close_datetime()),
            "strategy": strategy_label,
        }

        self._records.append(record)
        self._counter += 1

    def get_analysis(self):
        return {"trades": self._records}

    def stop(self):
        export_dir = getattr(self.params, "export_dir", None)
        if not export_dir or not self._records:
            return

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        outfile = export_path / f"trades_{os.getpid()}.csv"
        outfile = export_path / "trades_dev.csv"

        fieldnames = sorted({key for row in self._records for key in row.keys()})
        write_header = not outfile.exists()

        with outfile.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in self._records:
                writer.writerow(row)
