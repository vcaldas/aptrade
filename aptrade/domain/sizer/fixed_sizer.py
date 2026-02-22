from . import Sizer


class PercentTargetSizer(Sizer):
    def size(self, price: float, equity: float, target_percent: float) -> int:
        if price <= 0:
            return 0
        target_value = equity * (target_percent / 100)
        size = target_value / price
        return int(size)
