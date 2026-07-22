#!/usr/bin/env python
"""ATR / volatility-based sizer
=================================

``ATRSizer`` sizes positions based on recent Average True Range (ATR),
allocating a fixed percentage of portfolio value to the trade and
computing the size so that expected dollar risk per unit (based on ATR)
matches the target allocation. This helps achieve volatility-based
risk parity across instruments.

Parameters
----------
- ``percents``: percent of portfolio value to risk (default: 1.0 = 1%)
- ``atr_period``: lookback period to compute ATR (default: 14)
- ``multiplier``: scale factor applied to ATR (default: 1.0)
- ``retint``: return integer size when True (default: True)
- ``max_size``: optional cap on returned size (default: None)
"""

from dataclasses import dataclass
from typing import Optional

from aptrade.sizers import AbstractSizer


@dataclass(slots=True, frozen=True)
class ATRSizerParams:
    """Parameters for ATRSizer."""

    percents: float = 1.0
    atr_period: int = 14
    multiplier: float = 1.0
    retint: bool = True
    max_size: int | None = None


class ATRSizer(AbstractSizer):
    """Sizer that sizes positions using ATR (average true range).

    Algorithm (simplified):

    - Compute ATR over ``atr_period`` bars using true range definition.
    - Compute target dollar allocation = ``percents`` (percentage) of portfolio value.
    - Estimate dollar risk per unit = ATR * multiplier + per-unit commission.
    - size = target_allocation / dollar_risk_per_unit

    Notes
    -----
    - This implementation uses the ``data`` object values (``high``, ``low``, ``close``)
      and expects at least ``atr_period + 1`` bars of history.
    - Rounds down to integer if ``retint`` is True and applies ``max_size`` cap.
    """

    Parameters = ATRSizerParams

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Defensive defaults
        ap = max(1, int(self.p.atr_period))

        # Compute ATR (simple moving average of true range)
        trs = []
        for i in range(ap):
            try:
                high = float(data.high[-i])
                low = float(data.low[-i])
                prev_close = float(data.close[-i - 1])
            except Exception:
                # Not enough history; fallback to a single-bar range
                try:
                    high = float(data.high[0])
                    low = float(data.low[0])
                    prev_close = float(data.close[0])
                except Exception:
                    return 0

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        atr = sum(trs) / len(trs) if trs else 0.0

        # Dollar allocation to risk
        portfolio_value = self.broker.getvalue()
        target_dollars = (self.p.percents / 100.0) * portfolio_value

        # Commission per unit (best-effort). Many brokers put commission per trade;
        # try to read a per-unit commission if available
        commission_per_unit = 0.0
        try:
            commission_per_unit = float(comminfo.p.commission)
        except Exception:
            commission_per_unit = 0.0

        per_unit_risk = atr * float(self.p.multiplier) + commission_per_unit
        if per_unit_risk <= 0:
            return 0

        raw_size = target_dollars / per_unit_risk

        if self.p.retint:
            size = int(raw_size)
        else:
            size = raw_size

        if self.p.max_size is not None:
            try:
                size = min(size, int(self.p.max_size))
            except Exception:
                pass

        return max(0, size)
