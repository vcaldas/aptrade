"""Tests for Pydantic-based sizers."""

import math
import pytest
from pydantic import ValidationError

from aptrade.sizer.simple import SimpleSizer, SimpleSizerParams
# contents of example.py
from hypothesis import given, strategies as st

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

    def test_percents_validation_boundary_low(self):
        """Test percents validation - valid minimum boundary."""
        params = SimpleSizerParams(percents=0.1)
        assert params.percents == 0.1

    def test_percents_validation_boundary_high(self):
        """Test percents validation - valid maximum boundary."""
        params = SimpleSizerParams(percents=100)
        assert params.percents == 100


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
        def __init__(self, close_price=100):
            self.close = [close_price]

    class MockStrategy:
        pass


    @given(pct=st.floats(0.1, 99.9), price=st.floats(0.1, 120))
    def test_simple_sizer(self, pct, price):
        sizer = SimpleSizer(percents=pct)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=price)
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)
        
        assert size == math.floor((pct / 100 * 10000 / price))

    def test_sizing_calculation_50_percent(self):
        """Test sizing calculation with 50% of portfolio."""
        sizer = SimpleSizer(percents=50)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=100)
        
        # 50% of 10000 = 5000, at $100/share = 50 shares
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)
        # n% of 10000 = 10000, at $100/share = 100 shares
        assert size == 50

    def test_sizing_calculation_100_percent(self):
        """Test sizing calculation with 100% of portfolio."""
        sizer = SimpleSizer(percents=100)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=100)
        
        # 100% of 10000 = 10000, at $100/share = 100 shares
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)
        
        assert size == 100

    def test_sizing_calculation_25_percent(self):
        """Test sizing calculation with 25% of portfolio."""
        sizer = SimpleSizer(percents=25)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=100)
        
        # 25% of 10000 = 2500, at $100/share = 25 shares
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)
        
        assert size == 25

    def test_sizing_with_different_price(self):
        """Test sizing calculation with different stock price."""
        sizer = SimpleSizer(percents=50)
        sizer.broker = self.MockBroker()
        sizer.strategy = self.MockStrategy()
        data = self.MockData(close_price=50)
        
        # 50% of 10000 = 5000, at $50/share = 100 shares
        comminfo = sizer.broker.getcommissioninfo(data)
        size = sizer._getsizing(comminfo, 10000, data, isbuy=True)
        
        assert size == 100
