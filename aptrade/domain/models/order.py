# domain/models/order.py

from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: int
    is_buy: bool
    price: float
