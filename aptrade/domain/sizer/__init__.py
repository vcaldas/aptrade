from abc import ABC, abstractmethod


class Sizer(ABC):
    @abstractmethod
    def size(self, price: float, equity: float, target_percent: float) -> int:
        """Calculate order size based on price, equity, and target allocation."""
        pass


from .percent_sizer import PercentTargetSizer, BacktraderPercentTargetSizer
