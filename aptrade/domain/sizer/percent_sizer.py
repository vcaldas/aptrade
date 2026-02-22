from . import Sizer
import aptrade as bt


class PercentTargetSizer(Sizer):
    def size(self, price: float, equity: float, target_percent: float) -> int:
        if price <= 0:
            return 0
        target_value = equity * (target_percent / 100)
        size = target_value / price
        return int(size)


class BacktraderPercentTargetSizer(bt.AbstractSizer):
    """Adapter: bridges Backtrader sizer interface to domain PercentTargetSizer."""

    params = (("percents", 90.0),)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._domain_sizer = PercentTargetSizer()

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Called by Backtrader framework. Delegates to domain sizer."""
        equity = float(self.broker.getvalue())
        price = float(data.close[0]) + float(comminfo.p.commission)

        return self._domain_sizer.size(
            price=price, equity=equity, target_percent=float(self.params.percents)
        )
