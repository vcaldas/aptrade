from aptrade.domain.models.order import Order
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Engine")


class ExecutionEngine:
    def __init__(
        self, portfolio, strategy, sizer, commission_per_unit=0.0, data_port=None
    ):
        self.portfolio = portfolio
        self.strategy = strategy
        self.sizer = sizer
        self.commission = commission_per_unit
        self.market_prices = {}
        self.data_port = data_port  # optional

    def run(self):
        if self.data_port is None:
            raise ValueError("Data port not provided")
        for event in self.data_port.get_events():
            self.on_market_event(event)

    def on_market_event(self, event):
        self.market_prices[event.symbol] = event.price
        logger.info(f"MarketEvent: {event.symbol} price={event.price}")

        decision = self.strategy.on_event(event)
        if not decision:
            logger.info("Strategy: no action")
            return
        logger.info(f"Strategy decision: {decision}")

        portfolio_value = self.portfolio.total_value(self.market_prices)

        size = self.sizer.compute_size(
            portfolio_value=portfolio_value,
            cash=self.portfolio.cash,
            price=event.price,
            commission_per_unit=self.commission,
            is_buy=True,
        )
        logger.info(f"Sizer computed size: {size}")

        if size <= 0:
            logger.info("Sizer returned 0, skipping order")

            return

        order = Order(
            symbol=event.symbol,
            quantity=size,
            is_buy=True,
            price=event.price,
        )
        self.portfolio.apply_fill(order, self.commission)
        logger.info(f"Order executed: {order}")
        logger.info(
            f"Portfolio cash: {self.portfolio.cash}, position: {self.portfolio.get_position(event.symbol).quantity}\n"
        )
