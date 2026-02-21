class BuyOnFirstTickStrategy:
    def __init__(self):
        self.has_bought = False

    def on_event(self, event: MarketEvent):
        if not self.has_bought:
            self.has_bought = True
            return {"action": "buy"}
        return None
