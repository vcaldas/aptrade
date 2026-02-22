"""Tests for Pydantic-based sizers."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from aptrade.sizer.simple import SimpleSizer, SimpleSizerParams
from aptrade.domain.sizer import PercentTargetSizer


@given(pct=st.floats(0.1, 99.9), price=st.floats(0.1, 120))
def test_percent_target_sizer_basic(pct: float | int, price: float | int):
    sizer = PercentTargetSizer()
    equity = 10000.0
    target_percent = pct
    size = sizer.size(price, equity, target_percent)
    assert size == int(equity * (target_percent / 100) / price)


def test_percent_target_sizer_zero_price():
    sizer = PercentTargetSizer()
    price = 0.0
    equity = 10000.0
    target_percent = 90
    size = sizer.size(price, equity, target_percent)
    assert size == 0


## Initialization and boundaries
class TestSimpleSizerParams:
    """Test SimpleSizerParams Pydantic model."""

    def test_default_percents(self):
        """Test default percents value."""
        params = SimpleSizerParams()
        assert params.percents == 95

    def test_custom_percents(self):
        """Test setting custom percents value."""
        params = SimpleSizerParams(percents=50)
        assert params.percents == 50

    def test_percents_validation_min(self):
        """Test percents validation - minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSizerParams(percents=0)
        assert "greater than or equal to 0.1" in str(exc_info.value)

    def test_percents_validation_max(self):
        """Test percents validation - maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSizerParams(percents=101)
        assert "less than or equal to 100" in str(exc_info.value)


class TestSimpleSizerSizing:
    """Test SimpleSizer._getsizing calculation."""

    class MockBroker:
        def getvalue(self):
            return 10000  # $10k portfolio

        def getcommissioninfo(self, data):
            class MockCommInfo:
                class MockParams:
                    commission = 0

                p = MockParams()

            return MockCommInfo()

    class MockData:
        def __init__(self, close_price: float | int = 100):
            self.close = [close_price]

    class MockStrategy:
        pass

    @given(pct=st.floats(0.1, 99.9), price=st.floats(0.1, 120))
    def test_simple_sizer(self, pct: float | int, price: float | int):
        sizer = SimpleSizer(percents=pct)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=price)
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)

        assert size == math.floor((pct / 100 * 10000 / price))
