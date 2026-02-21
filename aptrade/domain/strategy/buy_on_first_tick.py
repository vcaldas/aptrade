from ..models.market_event import MarketEvent


class BuyOnFirstTickStrategy:
    """Buys once on the first market event."""

    def __init__(self):
        self.has_bought = False

    def on_event(self, event: MarketEvent):
        if not self.has_bought:
            self.has_bought = True
            return {"action": "buy"}
        return None
