from abc import ABC, abstractmethod
from typing import Iterator
from aptrade.domain.models.market_event import MarketEvent


class DataPort(ABC):
    """Port for historical or live market data."""

    @abstractmethod
    def get_events(self) -> Iterator[MarketEvent]:
        """Return a stream of MarketEvent objects."""
        raise NotImplementedError
