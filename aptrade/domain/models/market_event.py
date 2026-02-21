from dataclasses import dataclass


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    price: float
    timestamp: float | None = None  # optional
