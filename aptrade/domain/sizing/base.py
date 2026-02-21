from abc import ABC, abstractmethod


class PositionSizer(ABC):
    """
    Pure domain base class for position sizing.

    No broker.
    No strategy.
    No hidden state.
    """

    @abstractmethod
    def compute_size(
        self,
        *,
        portfolio_value: float,
        cash: float,
        price: float,
        commission_per_unit: float,
        is_buy: bool,
    ) -> int:
        """
        Compute the quantity to trade.

        All required state must be passed explicitly.
        """
        raise NotImplementedError
