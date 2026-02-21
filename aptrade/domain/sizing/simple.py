import math
from pydantic import BaseModel, ConfigDict, Field

from .base import PositionSizer


class SimpleSizerParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    percents: float = Field(
        default=95,
        ge=0.1,
        le=100,
        description="Percentage of portfolio value to allocate",
    )


class SimpleSizer(PositionSizer):
    def __init__(self, percents: float = 95):
        self.params = SimpleSizerParams(percents=percents)

    def compute_size(
        self,
        *,
        portfolio_value: float,
        cash: float,
        price: float,
        commission_per_unit: float,
        is_buy: bool,
    ) -> int:

        if price <= 0:
            return 0

        effective_price = price + commission_per_unit
        target_value = (self.params.percents / 100.0) * portfolio_value

        size = target_value / effective_price

        if is_buy:
            max_affordable = cash / effective_price
            size = min(size, max_affordable)

        return max(0, math.floor(size))
