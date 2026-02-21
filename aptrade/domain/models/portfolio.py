from dataclasses import dataclass


@dataclass
class MarketEvent:
    symbol: str
    price: float


@dataclass
class Order:
    symbol: str
    quantity: int
    is_buy: bool
    price: float


@dataclass
class Position:
    quantity: int = 0


class Portfolio:
    def __init__(self, cash: float):
        self.cash = cash
        self.positions = {}

    def get_position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position())

    def total_value(self, market_prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            value += pos.quantity * market_prices.get(symbol, 0)
        return value

    def apply_fill(self, order: Order, commission_per_unit: float):
        pos = self.get_position(order.symbol)
        cost = order.quantity * (order.price + commission_per_unit)

        if order.is_buy:
            self.cash -= cost
            pos.quantity += order.quantity
        else:
            self.cash += order.quantity * order.price
            pos.quantity -= order.quantity
