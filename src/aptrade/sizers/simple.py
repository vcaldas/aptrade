import math

"""Simple portfolio-percentage sizer
===================================

`SimpleSizer` allocates a fixed percentage of the current portfolio value
to compute an integer order size. It performs lightweight validation on the
``percents`` parameter.
"""


from dataclasses import dataclass

from aptrade.sizers import AbstractSizer


@dataclass(slots=True)
class SimpleSizerParams:
    """Parameters for SimpleSizer."""

    percents: float = 95.0

    def __post_init__(self):
        if not 0.1 <= self.percents <= 100:
            raise ValueError(
                f"'percents' must be between 0.1 and 100, got {self.percents}"
            )


class SimpleSizer(AbstractSizer):
    """Position sizer that uses a fixed percentage of portfolio value.

    Parameters:
        percents: Percentage of portfolio to use. Default: 95
    """

    # Backwards-compatible attribute name used historically
    Params = SimpleSizerParams
    # Provide a canonical `Parameters` name for Sphinx autodoc consistency
    Parameters = SimpleSizerParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _getsizing(self, comminfo, cash, data, isbuy) -> int:
        value = self.broker.getvalue()
        price = data.close[0] + comminfo.p.commission
        size = (self.p.percents / 100.0) * value / price
        return math.floor(size)


# class SimpleSizer(AbstractSizer):
#     """Position sizer that uses a fixed percentage of portfolio value.

#     Parameters:
#         percents: Percentage of portfolio to use (0.1-100). Default: 99
#     """

#     def __init__(self, percents: float = 95):
#         """Initialize with validated parameters.

#         Args:
#             percents: Percentage of portfolio to use

#         Raises:
#             ValidationError: If percents is not between 0.1 and 100
#         """
#         self.p = SimpleSizerParams(percents=percents)

#     def _getsizing(self, comminfo, cash, data, isbuy) -> int:
#         value = self.broker.getvalue()
#         price = data.close[0] + comminfo.p.commission
#         size = (self.p.percents / 100) * value / price
#         return math.floor(size)
