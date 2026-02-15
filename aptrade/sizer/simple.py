import math

from pydantic import BaseModel, ConfigDict, Field

from aptrade.sizer import AbstractSizer

# from aptrade.sizer import Sizer


class SimpleSizerParams(BaseModel):
    """Parameters for SimpleSizer with validation."""
    
    model_config = ConfigDict(frozen=True)  # Prevent accidental parameter changes
    
    percents: float = Field(
        default=95,
        ge=0.1,
        le=100,
        description="Percentage of portfolio to use (0.1-100)"
    )


class SimpleSizer(AbstractSizer):
    """Position sizer that uses a fixed percentage of portfolio value.
    
    Parameters:
        percents: Percentage of portfolio to use (0.1-100). Default: 99
    """

    def __init__(self, percents: float = 95):
        """Initialize with validated parameters.
        
        Args:
            percents: Percentage of portfolio to use
            
        Raises:
            ValidationError: If percents is not between 0.1 and 100
        """
        self.p = SimpleSizerParams(percents=percents)

    def _getsizing(self, comminfo, cash, data, isbuy) -> int:
        value = self.broker.getvalue()
        price = data.close[0] + comminfo.p.commission
        size = (self.p.percents / 100) * value / price
        return math.floor(size)
