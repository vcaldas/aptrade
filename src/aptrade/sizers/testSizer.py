"""Test sizer utilities for documentation and examples.

`testSizer` provides a minimal, deterministic sizer used in samples and
tests. It is intentionally simple and prints sizing decisions to stdout so
you can observe the sizer being called during backtests.

Usage example::

    from aptrade.sizers.testSizer import TestSizer
    cerebro.addsizer(TestSizer, stake=2)

The sizer multiplies the configured `stake` by 1 for buys and by 2 for
sells (demonstrating different behavior for entry vs exit).
"""

from dataclasses import dataclass

from aptrade.sizers import AbstractSizer


@dataclass(slots=True, frozen=True)
class TestSizerParameters:
    """Parameters for :class:`TestSizer`.

    Attributes:
        stake: base stake size used for buys; sells use a multiplier.
    """

    stake: int = 1


class TestSizer(AbstractSizer):
    """A tiny example sizer used in samples and tests.

    Behavior:
      - For buy orders returns ``stake``.
      - For sell orders returns ``2 * stake``.

    The sizer prints a log line with the current strategy date, data name,
    order type and computed size. It is useful to verify that the sizer is
    wired correctly in `Cerebro` during examples.
    """

    Parameters = TestSizerParameters

    def _getsizing(self, comminfo, cash, data, isbuy) -> int:
        """Compute the size and print a diagnostic line.

        Args:
            comminfo: commission information object provided by broker
            cash: available cash (not used in this trivial sizer)
            data: data feed object
            isbuy: True for buy, False for sell

        Returns:
            int: computed position size
        """

        dt = self.strategy.datetime.date()
        size = self.p.stake * (1 + (not isbuy))
        order_type = "buy" if isbuy else "sell"
        print(
            f"{dt} Data {getattr(data, '_name', '<data>')} OType {order_type} Sizing to {size}"
        )
        return size
