# aptrade/application/builder.py

from aptrade.domain.models.portfolio import Portfolio
from aptrade.domain.sizing.simple import SimpleSizer
from aptrade.engine import ExecutionEngine
from aptrade.ports.data_port import DataPort


class EngineBuilder:
    def __init__(self):
        self._initial_cash = 100_000
        self._strategy = None
        self._sizer = None
        self._commission = 0.0
        self._data_port = None

    def with_data_port(self, data_port: DataPort):
        self._data_port = data_port
        return self

    def with_cash(self, amount: float):
        self._initial_cash = amount
        return self

    def with_strategy(self, strategy):
        self._strategy = strategy
        return self

    def with_sizer(self, sizer):
        self._sizer = sizer
        return self

    def with_commission(self, commission: float):
        self._commission = commission
        return self

    def build(self):
        if self._strategy is None:
            raise ValueError("Strategy must be provided")

        if self._sizer is None:
            self._sizer = SimpleSizer()

        portfolio = Portfolio(cash=self._initial_cash)

        engine = ExecutionEngine(
            portfolio=portfolio,
            strategy=self._strategy,
            sizer=self._sizer,
            commission_per_unit=self._commission,
            data_port=self._data_port,
        )
        return engine
